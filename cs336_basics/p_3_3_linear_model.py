

import torch
from einops import rearrange, einsum
from math import sqrt
'''
y= W x
'''

class LinearModel(torch.nn.Module):
    def __init__(self, in_features,out_features,device = None,dtype = None):
        '''
        in_features: int final dimension of the input
        out_features: int final dimension of the output
        device: torch.device | None = None Device to store the parameters on
        dtype: torch.dtype | None = None Data type of the parameters
        '''
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.dtype = dtype

        self.W = torch.nn.Parameter(torch.empty(out_features,in_features,device = device,dtype = dtype))
        
        '''initialize the weights with torch.nn.init.trunc_normal_'''
        mean = 0.0
        std = sqrt(2.0 / (in_features + out_features))
        a = -3*std
        b = 3*std
        torch.nn.init.trunc_normal_(self.W,
                                     mean = mean,
                                     std = std,
                                     a = a,
                                     b = b)


    def forward(self,x:torch.Tensor)->torch.Tensor:
        '''
        x: torch.Tensor of shape (batch_size, in_features)
        Returns: torch.Tensor of shape (batch_size, out_features)
        '''
        y = einsum(x, self.W, "batch_size in_features, out_features in_features -> batch_size out_features")
        return y