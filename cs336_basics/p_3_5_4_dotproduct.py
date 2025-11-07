

import torch
from einops import einsum


class DotProductAttention():


    def __init__(self) -> None:
        pass

    def forward(self, query, key, value, mask=None):
        """
        Args:
            Q (Float[Tensor, " ... queries d_k"]): Query tensor
            K (Float[Tensor, " ... keys d_k"]): Key tensor
            V (Float[Tensor, " ... values d_v"]): Values tensor
            mask (Bool[Tensor, " ... queries keys"] | None): Mask tensor
        Returns:
            Float[Tensor, " ... queries d_v"]: Output of SDPA
        """
        # Q*K^T
        scores = einsum(query, key, " ... queries d_k, ... keys d_k -> ... queries keys")
        # Scale scores
        scores = scores / (query.shape[-1] ** 0.5)
        # Apply mask
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        return scores



        
    


