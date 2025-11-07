

import torch
from jaxtyping import Float
from torch import nn, Tensor
from einops import einsum,einops

class SwiGLu(nn.Module):

    def __init__(self,
                 d_model:int,
                 d_ff:int,
                 device = None,
                 dtype = None,
                ):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.w1_weight = torch.nn.Parameter(torch.empty(d_ff,d_model,device = device,dtype = dtype))
        self.w2_weight = torch.nn.Parameter(torch.empty(d_model,d_ff,device = device,dtype = dtype))
        self.w3_weight = torch.nn.Parameter(torch.empty(d_ff,d_model,device = device,dtype = dtype))

    
    def forward(self,x:Float[Tensor,"... d_model"]) -> Float[Tensor,"... d_model"]:
        '''
        Process an input tensor of shape (batch_size, sequence_length, d_model) and return a tensor of the same shape
        '''
        in_type = x.dtype
        x = x.to(torch.float32)
        W1x = einsum(x,self.w1_weight,"... d_model, d_ff d_model -> ... d_ff")
        Silu_W1x = W1x * torch.sigmoid(W1x)
        W3x = einsum(x,self.w3_weight,"... d_model, d_ff d_model -> ... d_ff")
        Silu_W1x_W3x = Silu_W1x * W3x
        SwiGLu = einsum(Silu_W1x_W3x,self.w2_weight,"... d_ff, d_model d_ff -> ... d_model")

        return SwiGLu.to(in_type)