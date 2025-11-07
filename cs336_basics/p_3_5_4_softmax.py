



import torch

class Softmax:
    def __init__(self):
        pass

    def forward(self, x):
        exp_x = torch.exp(x - torch.max(x, dim=-1, keepdim=True).values)
        
        return exp_x / torch.sum(exp_x, dim=-1, keepdim=True)
    


