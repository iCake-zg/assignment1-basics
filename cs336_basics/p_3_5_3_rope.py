

import torch

from einops import einsum
class RotaryPositionEmbedding(torch.nn.Module):

    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        '''
        Construct the RoPE module and create buffers if needed.
        theta: float Θ value for the RoPE
        d_k: int dimension of query and key vectors
        max_seq_len: int Maximum sequence length that will be inputted
        device: torch.device | None = None Device to store the buffer on  
        '''
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len

        print("d_k:", d_k)
        print("max_seq_len:", max_seq_len)
        print("theta:", theta)
        # 预计算cos 和 sin值
        # 为每个位置i 和每个维度对k 计算角度 theta—{i,k}
        k_indices = torch.arange(0,d_k //2, device = device)  
        freqs = 1/(theta ** (((2*k_indices)-2) / d_k))
        freqs = 1.0 / (theta ** (2 * k_indices / d_k))
        
        print("k_indices:", k_indices)
        print("freqs:", freqs)
        # 计算位置索引
        positions = torch.arange(max_seq_len, device = device)
        print(positions)
        # 计算角度
        angles = torch.outer(positions, freqs)  # shape = (max_seq_len,dk//2)
        # angles = einsum(positions, freqs, 'i d_k, i d_k -> i d_k')
        print(angles)

        # 计算cos 和 sin值
        cos_angles = torch.cos(angles)
        sin_angles = torch.sin(angles)  
        
        # 存储cos 和 sin值为缓冲区
        self.register_buffer('cos_cashed', cos_angles,persistent=False)
        self.register_buffer('sin_cashed', sin_angles,persistent=False)



    
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        '''
        Process an input tensor of shape (..., seq_len, d_k) and return a tensor of the same shape.
        Note that you should tolerate x with an arbitrary number of batch dimensions. You should
        assume that the token positions are a tensor of shape (..., seq_len) specifying the token
        positions of x along the sequence dimension.
        '''

        seq_len = x.shape[-2]
        
        print("token_positions:", token_positions)
        positions = token_positions.long()
        cos = self.cos_cashed[positions]
        print(cos.shape)
        sin = self.sin_cashed[positions]
        
        # cos = self.cos_cashed
        # sin = self.sin_cashed
        # 将 x 分成两部分：偶数索引和奇数索引
        # x 的形状是 (..., seq_len, d_k)
        # 我们需要将其重塑为 (..., seq_len, d_k//2, 2) 然后分离
        x_reshaped = x.reshape(*x.shape[:-1], -1, 2)
        x_even = x_reshaped[..., 0]
        x_odd = x_reshaped[..., 1]

        # print(cos)
        # print(sin)
        # 应用旋转
        x_even_rotated = x_even * cos - x_odd * sin
        x_odd_rotated = x_even * sin + x_odd * cos
        # print(x_even_rotated)
        # print(x_odd_rotated)
        # 合并两部分
        rotated = torch.stack([x_even_rotated, x_odd_rotated], dim=-1).reshape(*x.shape)

        return rotated
