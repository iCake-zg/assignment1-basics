


import torch
from einops import rearrange, einsum

D = [[1,2],[3,4]]
A = [[1,2],[3,4]]
D = torch.tensor(D)
A = torch.tensor(A)
# Y = D@A.T

Y = einsum(D, A, "batch sequence d_in, d_out d_in -> batch sequence d_out")



