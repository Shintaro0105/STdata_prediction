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


class LGnet(nn.Module):
    def __init__(self, input_size, hidden_size, mask_size, X_mean, output_last=False):
        super(LGnet, self).__init__()

        self.hidden_size = hidden_size
        self.mask_size = mask_size

        # Define the LSTM gate layers
        self.il = nn.Linear(input_size + hidden_size + mask_size, hidden_size)
        self.fl = nn.Linear(input_size + hidden_size + mask_size, hidden_size)
        self.ol = nn.Linear(input_size + hidden_size + mask_size, hidden_size)
        self.cl = nn.Linear(input_size + hidden_size + mask_size, hidden_size)

        # Other initializations
        self.gamma_x_l = FilterLinear(input_size, input_size, torch.eye(input_size))
        self.gamma_h_l = nn.Linear(input_size, input_size)

        self.output_last = output_last

        use_gpu = torch.cuda.is_available()
        self.X_mean = Variable(torch.Tensor(X_mean).cuda()) if use_gpu else Variable(torch.Tensor(X_mean))
        self.zeros = Variable(torch.zeros(input_size).cuda()) if use_gpu else Variable(torch.zeros(input_size))

    def step(self, x, x_last_obsv, x_mean, h, c, mask, delta):
        delta_x = torch.exp(-torch.max(self.zeros, self.gamma_x_l(delta)))
        delta_h = torch.exp(-torch.max(self.zeros, self.gamma_h_l(delta)))

        x = mask * x + (1 - mask) * (delta_x * x_last_obsv + (1 - delta_x) * x_mean)
        h = delta_h * h

        combined = torch.cat((x, h, mask), 1)
        i = F.sigmoid(self.il(combined))
        f = F.sigmoid(self.fl(combined))
        o = F.sigmoid(self.ol(combined))
        c_tilde = F.tanh(self.cl(combined))
        c = f * c + i * c_tilde
        h = o * F.tanh(c)

        return h, c

    def forward(self, input):
        batch_size = input.size(0)
        step_size = input.size(2)

        Hidden_State, Cell_State = self.initHidden(batch_size)
        X = torch.squeeze(input[:, 0, :, :])
        X_last_obsv = torch.squeeze(input[:, 1, :, :])
        Mask = torch.squeeze(input[:, 2, :, :])
        Delta = torch.squeeze(input[:, 3, :, :])

        outputs = None
        for i in range(step_size):
            Hidden_State, Cell_State = self.step(
                torch.squeeze(X[:, i : i + 1, :]),
                torch.squeeze(X_last_obsv[:, i : i + 1, :]),
                torch.squeeze(self.X_mean[:, i : i + 1, :]),
                Hidden_State,
                Cell_State,
                torch.squeeze(Mask[:, i : i + 1, :]),
                torch.squeeze(Delta[:, i : i + 1, :]),
            )
            if outputs is None:
                outputs = Hidden_State.unsqueeze(1)
            else:
                outputs = torch.cat((outputs, Hidden_State.unsqueeze(1)), 1)

        if self.output_last:
            return outputs[:, -1, :]
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
