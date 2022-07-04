import math

import torch

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


class GraphConvolution(Module):

    def __init__(self, in_features, out_features, bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input_data, adj):
        """
        support = torch.matmul(input_data, self.weight)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output
        """

        diag = torch.sum(adj.clone(), dim= 2)
        diag = torch.diag_embed(diag.clone(), dim1=1, dim2 = 2)
        diag_norm = torch.sqrt(torch.inverse(diag))
        
        interm = torch.matmul(torch.matmul(torch.matmul(diag_norm, adj),diag_norm),input_data)
        output = torch.nn.functional.relu(torch.matmul(interm, self.weight)),
        return output[0]

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
