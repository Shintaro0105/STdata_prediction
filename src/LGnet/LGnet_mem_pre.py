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


class LGnet_mem(nn.Module):
    def __init__(
        self, input_size, hidden_size, output_size, X_mean, memory_size, memory_dim, num_layers, output_last=False
    ):
        super(LGnet_mem, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.memory_size = memory_size
        self.memory_dim = memory_dim

        # Define the LSTM gate layers
        self.il = nn.Linear(memory_dim + hidden_size, hidden_size)
        self.fl = nn.Linear(memory_dim + hidden_size, hidden_size)
        self.ol = nn.Linear(memory_dim + hidden_size, hidden_size)
        self.cl = nn.Linear(memory_dim + hidden_size, hidden_size)

        self.fc = nn.Linear(hidden_size, output_size)

        # Other initializations
        self.gamma_z_l = FilterLinear(input_size, input_size, torch.eye(input_size))
        self.gamma_z_prime_l = FilterLinear(input_size, input_size, torch.eye(input_size))

        self.q_for_memory = nn.Linear(2 * input_size + output_size, memory_dim)

        # Initialize memory component
        # self.memory = nn.Parameter(torch.rand(memory_size, memory_dim))

        self.local_statistics = torch.Tensor(memory_dim)

        self.z = torch.Tensor(input_size)

        self.z_prime = torch.Tensor(input_size)

        self.x_i = torch.Tensor(output_size)

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

        x_i = self.fc(h)

        self.z = z
        self.z_prime = z_prime
        self.x_i = x_i

        # print("z")
        # print(z.shape)
        # print("z_prime")
        # print(z_prime.shape)
        # print("x_i")
        # print(x_i.shape)

        local_statistics = self.q_for_memory(torch.cat((z, z_prime, x_i), 1))

        self.local_statistics = local_statistics

        # print("local")
        # print(local_statistics.shape)
        # print("s_i")
        # print(s_i.shape)
        # print("global")
        # print(global_dynamics.shape)

        # print("h")
        # print(h.shape)

        combined = torch.cat((local_statistics, h), 1)

        # print("combined")
        # print(combined.shape)

        i = F.sigmoid(self.il(combined))
        f = F.sigmoid(self.fl(combined))
        o = F.sigmoid(self.ol(combined))
        c_tilde = F.tanh(self.cl(combined))
        c = f * c + i * c_tilde
        h = o * F.tanh(c)

        # print("outputs")
        # print(outputs.shape)

        return h, c

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

        # print("X shape:", X.shape)
        # print("X values:", X)
        # print("X_last_obsv shape:", X_last_obsv.shape)
        # print("X_last_obsv values:", X_last_obsv)
        # print("Mask shape:", Mask.shape)
        # print("Mask values:", Mask)
        # print("Delta shape:", Delta.shape)
        # print("Delta values:", Delta)
        # print("X_last_obsv_b shape:", X_last_obsv_b.shape)
        # print("X_last_obsv_b values:", X_last_obsv_b)
        # print("Delta_b shape:", Delta_b.shape)
        # print("Delta_b values:", Delta_b)

        outputs = None
        for i in range(step_size):
            Hidden_State, Cell_State = self.step(
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
            # print(outputs)

            if outputs is None:
                outputs = self.fc(Hidden_State).unsqueeze(1)
            else:
                outputs = torch.cat((outputs, self.fc(Hidden_State).unsqueeze(1)), 1)

        if self.output_last:
            h = Hidden_State
            c = Cell_State
            forecasts = outputs[:, -1, :].squeeze()

            local_statistics = self.q_for_memory(torch.cat((forecasts, forecasts, forecasts), 1))

            # print("local")
            # print(local_statistics.shape)
            # print("s_i")
            # print(s_i.shape)
            # print("global")
            # print(global_dynamics.shape)

            # print("h")
            # print(h.shape)

            combined = torch.cat((local_statistics, h), 1)

            # print("combined")
            # print(combined.shape)

            i = F.sigmoid(self.il(combined))
            f = F.sigmoid(self.fl(combined))
            o = F.sigmoid(self.ol(combined))
            c_tilde = F.tanh(self.cl(combined))
            c = f * c + i * c_tilde
            h = o * F.tanh(c)

            return outputs, self.fc(h).unsqueeze(1)
        else:
            return outputs

    def initHidden(self, batch_size):
        use_gpu = torch.cuda.is_available()
        if use_gpu:
            Hidden_State = Variable(torch.zeros(batch_size, self.hidden_size).cuda())
            Cell_State = Variable(torch.zeros(batch_size, self.hidden_size).cuda())
        else:
            Hidden_State = Variable(torch.zeros(batch_size, self.hidden_size))
            Cell_State = Variable(torch.zeros(batch_size, self.hidden_size))
        return Hidden_State, Cell_State
