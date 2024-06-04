import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter


class FilterLinear(nn.Module):
    def __init__(self, in_features, out_features, filter_square_matrix, bias=True):
        """
        filter_square_matrix : filter square matrix, whose each elements is 0 or 1.
        """
        super(FilterLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        use_gpu = torch.cuda.is_available()
        self.filter_square_matrix = None
        if use_gpu:
            self.filter_square_matrix = Variable(filter_square_matrix.cuda(), requires_grad=False)
        else:
            self.filter_square_matrix = Variable(filter_square_matrix, requires_grad=False)

        self.weight = Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    #         print(self.weight.data)
    #         print(self.bias.data)

    def forward(self, input):
        # フィルタ行列を使用して重み行列をフィルタリング
        filtered_weight = self.filter_square_matrix.mul(self.weight)
        output = F.linear(input, filtered_weight, self.bias)

        # 不要な変数を削除し、メモリ解放
        del filtered_weight
        torch.cuda.empty_cache()

        return output

    def __repr__(self):
        return (
            self.__class__.__name__
            + "("
            + "in_features="
            + str(self.in_features)
            + ", out_features="
            + str(self.out_features)
            + ", bias="
            + str(self.bias is not None)
            + ")"
        )


class MemoryModule(nn.Module):
    def __init__(self, input_dim, memory_size):
        super(MemoryModule, self).__init__()
        self.memory = nn.Parameter(torch.randn(memory_size, input_dim))

    def forward(self, local_stats):
        batch_size, num_features = local_stats.shape
        local_stats_flat = local_stats.view(batch_size, num_features)
        attention_weights = torch.matmul(local_stats_flat, self.memory.t())
        attention_weights = torch.softmax(attention_weights, dim=1)
        global_stats = torch.matmul(attention_weights, self.memory)
        global_stats = global_stats.view(batch_size, num_features)
        return global_stats


class LSTMModule(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(LSTMModule, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, h=None, c=None):
        if h is None or c is None:
            out, (h, c) = self.lstm(x)
        else:
            out, (h, c) = self.lstm(x, (h, c))
        out = self.fc(out)
        return out, h, c


class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(Discriminator, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # Take the output from the last time step
        out = self.sigmoid(out)
        return out


class LGnet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, X_mean, memory_size, num_layers, output_last=False):
        super(LGnet, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Define the LSTM gate layers
        self.memory_module = MemoryModule(input_size, memory_size)
        self.lstm_module = LSTMModule(input_size, hidden_size, output_size, num_layers)
        self.discriminator = Discriminator(input_size, hidden_size, num_layers)

        # Other initializations
        self.gamma_z_l = FilterLinear(input_size, input_size, torch.eye(input_size))
        self.gamma_z_prime_l = FilterLinear(input_size, input_size, torch.eye(input_size))

        self.q_for_memory = nn.Linear(2 * input_size + output_size, input_size)

        self.output_last = output_last

        use_gpu = torch.cuda.is_available()
        if use_gpu:
            self.X_mean = Variable(torch.Tensor(X_mean).cuda())
            self.zeros = Variable(torch.zeros(input_size).cuda())
        else:
            self.X_mean = Variable(torch.Tensor(X_mean))
            self.zeros = Variable(torch.zeros(input_size))

    def step(self, x, x_last_obsv, x_last_obsv_b, x_mean, h, c, mask, delta, delta_b):
        delta_z = torch.exp(-torch.max(self.zeros, self.gamma_z_l(delta)))
        delta_z_prime = torch.exp(-torch.max(self.zeros, self.gamma_z_prime_l(delta_b)))

        z = mask * x + (1 - mask) * (delta_z * x_last_obsv + (1 - delta_z) * x_mean)
        z_prime = mask * x + (1 - mask) * (delta_z_prime * x_last_obsv_b + (1 - delta_z_prime) * x_mean)

        x_i = self.lstm_module.fc(h.squeeze())

        local_statistics = self.q_for_memory(torch.cat((z, z_prime, x_i), 1))
        global_dynamics = self.memory_module(local_statistics)
        global_dynamics = global_dynamics.unsqueeze(1)

        outputs, h, c = self.lstm_module(global_dynamics, h, c)

        return outputs, h, c

    def forward(self, input):
        batch_size = input.size(0)
        step_size = input.size(2)

        Hidden_State, Cell_State = self.initHidden(batch_size)
        X = torch.squeeze(input[:, 0, :, :])
        X_last_obsv = torch.squeeze(input[:, 1, :, :])
        Mask = torch.squeeze(input[:, 2, :, :])
        Delta = torch.squeeze(input[:, 3, :, :])
        X_last_obsv_b = torch.squeeze(input[:, 4, :, :])
        Delta_b = torch.squeeze(input[:, 5, :, :])

        outputs = None
        for i in range(step_size):
            outputs, Hidden_State, Cell_State = self.step(
                torch.squeeze(X[:, i : i + 1, :]),
                torch.squeeze(X_last_obsv[:, i : i + 1, :]),
                torch.squeeze(X_last_obsv_b[:, i : i + 1, :]),
                torch.squeeze(self.X_mean[:, i : i + 1, :]),
                Hidden_State,
                Cell_State,
                torch.squeeze(Mask[:, i : i + 1, :]),
                torch.squeeze(Delta[:, i : i + 1, :]),
                torch.squeeze(Delta_b[:, i : i + 1, :]),
            )

        if self.output_last:
            return outputs[:, -1, :]
        else:
            return outputs

    def initHidden(self, batch_size):
        use_gpu = torch.cuda.is_available()
        if use_gpu:
            Hidden_State = Variable(torch.zeros(self.num_layers, batch_size, self.hidden_size).cuda())
            Cell_State = Variable(torch.zeros(self.num_layers, batch_size, self.hidden_size).cuda())
        else:
            Hidden_State = Variable(torch.zeros(self.num_layers, batch_size, self.hidden_size))
            Cell_State = Variable(torch.zeros(self.num_layers, batch_size, self.hidden_size))
        return Hidden_State, Cell_State
