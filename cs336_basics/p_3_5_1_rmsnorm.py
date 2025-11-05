


import torch
from math import sqrt
from einops import einsum

class RMSNorm(torch.nn.Module):
    def __init__(self,d_model:int,eps:float = 1e-5,device = None,dtype = None):
        super().__init__()
        '''
        d_model: int Hidden dimension of the model
        eps: float = 1e-5 Epsilon value for numerical stability
        device: torch.device | None = None Device to store the parameters on
        dtype: torch.dtype | None = None Data type of the parameters
        '''
        self.eps = eps
        self.d_model = d_model
        self.weight = torch.nn.Parameter(torch.empty(d_model,device = device,dtype = dtype))
        # torch.nn.init.ones_(self.weight)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Process an input tensor of shape (batch_size, sequence_length, d_model) and return a tensor of the same shape
        '''
        in_dtype = x.dtype
        x = x.to(torch.float32)
        RMS = torch.sqrt((1/self.d_model) * torch.sum(x**2,dim  = -1,keepdim = True) + self.eps)    # 沿着最后一个纬度进行平方
        # RMS_Norm = (x / RMS) * self.weight
        RMS_Norm = einsum((x / RMS),self.weight,"... d_model, d_model -> ... d_model")
        return RMS_Norm.to(in_dtype)
        



