import math, torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

class snowball_layer(Module):
    def __init__(self, in_features, out_features, amp = 0):
        super(snowball_layer, self).__init__()
        if amp:
            self.multiplication = torch.mm
        else:
            self.multiplication = torch.spmm
        self.in_features, self.out_features = in_features, out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features).cuda())
        self.bias = Parameter(torch.FloatTensor(out_features).cuda())
        self.reset_parameters()
    
    def reset_parameters(self):
        stdv_weight, stdv_bias = 1. / math.sqrt(self.weight.size(1)), 1. / math.sqrt(self.bias.size(0))
        self.weight.data.uniform_(-stdv_weight, stdv_weight); self.bias.data.uniform_(-stdv_bias, stdv_bias)
    
    def forward(self, input, adj, eye = False):
        if eye:
            return torch.mm(input, self.weight) + self.bias
        else:
            return self.multiplication(adj, torch.mm(input, self.weight) + self.bias)
            

class truncated_krylov_layer(Module):
    def __init__(self, in_features, n_blocks, out_features, amp = 0, ADJ_EXPONENTIALS = None):
        super(truncated_krylov_layer, self).__init__()
        if amp:
            self.multiplication = torch.mm
        else:
            self.multiplication = torch.spmm
        self.ADJ_EXPONENTIALS = ADJ_EXPONENTIALS
        self.in_features, self.out_features, self.n_blocks = in_features, out_features, n_blocks
        self.shared_weight = Parameter(torch.FloatTensor(in_features * n_blocks, out_features).cuda())
        self.output_bias = Parameter(torch.FloatTensor(out_features).cuda())
        self.reset_parameters()

    def reset_parameters(self):
        stdv_shared_weight, stdv_output_bias = 1. / math.sqrt(self.shared_weight.size(1)), 1. / math.sqrt(self.output_bias.size(0))
        self.shared_weight.data.uniform_(-stdv_shared_weight, stdv_shared_weight); self.output_bias.data.uniform_(-stdv_output_bias, stdv_output_bias)

    def forward(self, input, adj, eye = True):
        feature_output = []
        for i in range(self.n_blocks):
            AX = self.multiplication(self.ADJ_EXPONENTIALS[i], input)
            feature_output.append(AX)
        output = torch.mm(torch.cat(feature_output, 1), self.shared_weight)
        if eye:
            return output + self.output_bias
        else:
            return self.multiplication(adj, output) + self.output_bias
