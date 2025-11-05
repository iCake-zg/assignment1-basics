




from typing import Optional
import torch
from torch import Tensor
from math import sqrt

class EmbeddingModel(torch.nn.Module):

    def __init__(self, num_embeddings,embedding_dim,device = None,dtype = None):
        super().__init__()

        '''
        num_embeddings: int Size of the vocabulary
        embedding_dim: int Dimension of the embedding vectors, i.e., dmodel
        device: torch.device | None = None Device to store the parameters on
        dtype: torch.dtype | None = None Data type of the parameters
        '''
        self.weight = torch.nn.Parameter(torch.empty(num_embeddings,embedding_dim,device = device,dtype = dtype))  # (num_embeddings, embedding_dim)

        std = sqrt(2.0 / (num_embeddings + embedding_dim))
        mean = 0.0
        torch.nn.init.trunc_normal_(self.weight,
                                     mean = mean,
                                     std = std,
                                     a = -3*std,
                                     b = 3*std)


    def forward(self,token_ids:torch.Tensor)->torch.Tensor:
        '''
        Lookup the embedding vectors for the given token IDs.
        '''
        return self.weight[token_ids]





