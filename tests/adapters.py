from __future__ import annotations

import os
from collections.abc import Iterable
from typing import IO, Any, BinaryIO

import numpy.typing as npt
import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor


def run_linear(
    d_in: int,
    d_out: int,
    weights: Float[Tensor, " d_out d_in"],
    in_features: Float[Tensor, " ... d_in"],
) -> Float[Tensor, " ... d_out"]:
    """
    Given the weights of a Linear layer, compute the transformation of a batched input.

    Args:
        in_dim (int): The size of the input dimension
        out_dim (int): The size of the output dimension
        weights (Float[Tensor, "d_out d_in"]): The linear weights to use
        in_features (Float[Tensor, "... d_in"]): The output tensor to apply the function to

    Returns:
        Float[Tensor, "... d_out"]: The transformed output of your linear module.
    """
    from cs336_basics.p_3_3_linear_model import LinearModel
    linear = LinearModel(d_in, d_out, device=weights.device, dtype=weights.dtype)
    linear.load_state_dict({'W': weights})
    output = linear.forward(in_features)
    return output
    # raise NotImplementedError


def run_embedding(
    vocab_size: int,
    d_model: int,
    weights: Float[Tensor, " vocab_size d_model"],
    token_ids: Int[Tensor, " ..."],
) -> Float[Tensor, " ... d_model"]:
    """
    Given the weights of an Embedding layer, get the embeddings for a batch of token ids.

    Args:
        vocab_size (int): The number of embeddings in the vocabulary
        d_model (int): The size of the embedding dimension
        weights (Float[Tensor, "vocab_size d_model"]): The embedding vectors to fetch from
        token_ids (Int[Tensor, "..."]): The set of token ids to fetch from the Embedding layer

    Returns:
        Float[Tensor, "... d_model"]: Batch of embeddings returned by your Embedding layer.
    """
    from cs336_basics.p_3_4_embedding_model import EmbeddingModel
    embedding = EmbeddingModel(vocab_size, d_model, device=weights.device, dtype=weights.dtype)
    embedding.load_state_dict({'weight': weights})
    output = embedding.forward(token_ids)
    return output

    raise NotImplementedError


def run_swiglu(
    d_model: int,
    d_ff: int,
    w1_weight: Float[Tensor, " d_ff d_model"],
    w2_weight: Float[Tensor, " d_model d_ff"],
    w3_weight: Float[Tensor, " d_ff d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    """Given the weights of a SwiGLU network, return
    the output of your implementation with these weights.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        d_ff (int): Dimensionality of the up-project happening internally to your swiglu.
        w1_weight (Float[Tensor, "d_ff d_model"]): Stored weights for W1
        w2_weight (Float[Tensor, "d_model d_ff"]): Stored weights for W2
        w3_weight (Float[Tensor, "d_ff d_model"]): Stored weights for W3
        in_features (Float[Tensor, "... d_model"]): Input embeddings to the feed-forward layer.

    Returns:
        Float[Tensor, "... d_model"]: Output embeddings of the same shape as the input embeddings.
    """
    # Example:
    # If your state dict keys match, you can use `load_state_dict()`
    # swiglu.load_state_dict(weights)
    # You can also manually assign the weights
    # swiglu.w1.weight.data = w1_weight
    # swiglu.w2.weight.data = w2_weight
    # swiglu.w3.weight.data = w3_weight
    from cs336_basics.p_3_5_2_swiglu import SwiGLu
    swiglu = SwiGLu(d_model, d_ff)
    swiglu.load_state_dict({'w1_weight': w1_weight, 'w2_weight': w2_weight, 'w3_weight': w3_weight})
    output = swiglu.forward(in_features)
    return output
    raise NotImplementedError


def run_scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... values d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    """
    Given key (K), query (Q), and value (V) tensors, return
    the output of your scaled dot product attention implementation.

    Args:
        Q (Float[Tensor, " ... queries d_k"]): Query tensor
        K (Float[Tensor, " ... keys d_k"]): Key tensor
        V (Float[Tensor, " ... values d_v"]): Values tensor
        mask (Bool[Tensor, " ... queries keys"] | None): Mask tensor
    Returns:
        Float[Tensor, " ... queries d_v"]: Output of SDPA
    """
    from einops import einsum
    # QK
    scores = einsum(Q, K, " ... q d_k, ... k d_k -> ... q k")
    # Scale scores
    scores = scores / (Q.shape[-1] ** 0.5)
    # Apply mask
    if mask is not None:
        scores = scores.masked_fill(~mask,float('-inf'))
        # scores = scores.masked_fill(~mask, float('-inf'))
    # softmax
    from cs336_basics.p_3_5_4_softmax import Softmax
    softmax = Softmax()
    scores = softmax.forward(scores)
    # V
    output = einsum(scores, V, " ... q k, ... k d_v -> ... q d_v")

    return output


    # raise NotImplementedError


def run_multihead_self_attention(
    d_model: int,
    num_heads: int,
    q_proj_weight: Float[Tensor, " d_k d_in"],
    k_proj_weight: Float[Tensor, " d_k d_in"],
    v_proj_weight: Float[Tensor, " d_v d_in"],
    o_proj_weight: Float[Tensor, " d_model d_v"],
    in_features: Float[Tensor, " ... sequence_length d_in"],
) -> Float[Tensor, " ... sequence_length d_out"]:
    """
    Given the key, query, and value projection weights of a naive unbatched
    implementation of multi-head attention, return the output of an optimized batched
    implementation. This implementation should handle the key, query, and value projections
    for all heads in a single matrix multiply.
    This function should not use RoPE.
    See section 3.2.2 of Vaswani et al., 2017.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        num_heads (int): Number of heads to use in multi-headed attention.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        q_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the Q projection
        k_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the K projection
        v_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the V projection
        o_proj_weight (Float[Tensor, "d_model d_v"]): Weights for the output projection
        in_features (Float[Tensor, "... sequence_length d_in"]): Tensor to run your implementation on.

    Returns:
        Float[Tensor, " ... sequence_length d_out"]: Tensor with the output of running your optimized, batched multi-headed attention
        implementation with the given QKV projection weights and input features.
    """
    import torch
    from einops import einsum


    
    # Get dimensions
    *batch_dims, seq_len, d_in = in_features.shape
    d_k = q_proj_weight.shape[0]
    d_v = v_proj_weight.shape[0]
    head_dim_k = d_k // num_heads
    head_dim_v = d_v // num_heads

    batch_size = 1
    for dim in batch_dims:
        batch_size *= dim
    
    x = in_features.reshape(batch_size,seq_len,d_in)

    # x: [batch_size, seq_len, d_in]
    Q = einsum(x, q_proj_weight ,"... s d_in,  d_k d_in -> ... s d_k")
    K = einsum(x, k_proj_weight ,"... s d_in,  d_k d_in -> ... s d_k")
    V = einsum(x, v_proj_weight ,"... s d_in,  d_v d_in -> ... s d_v")
    
    # Reshape for multi-head attention
    # [batch_size, seq_len, d_k] -> [batch_size, seq_len, num_heads, head_dim_k]
    # -> [batch_size, num_heads, seq_len, head_dim_k]
    Q = Q.view(batch_size, seq_len, num_heads, head_dim_k).transpose(1, 2)
    K = K.view(batch_size, seq_len, num_heads, head_dim_k).transpose(1, 2)
    V = V.view(batch_size, seq_len, num_heads, head_dim_v).transpose(1, 2)
    
    # Scaled dot-product attention
    # Q: [batch_size, num_heads, seq_len, head_dim_k]
    # K: [batch_size, num_heads, seq_len, head_dim_k]
    # Attention scores: [batch_size, num_heads, seq_len, seq_len]
    scores = torch.matmul(Q, K.transpose(-2, -1)) / (head_dim_k ** 0.5)
    
    # Apply causal mask (for autoregressive attention)
    causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=scores.device), diagonal=1).bool()
    scores = scores.masked_fill(causal_mask, float('-inf'))
    
    # Apply softmax
    from cs336_basics.p_3_5_4_softmax import Softmax
    softmax = Softmax()
    attn_weights = softmax.forward(scores)  # [batch_size, num_heads, seq_len, seq_len]
    
    # Apply attention to values
    # attn_weights: [batch_size, num_heads, seq_len, seq_len]
    # V: [batch_size, num_heads, seq_len, head_dim_v]
    # output: [batch_size, num_heads, seq_len, head_dim_v]
    attn_output = torch.matmul(attn_weights, V)
    
    # Concatenate heads
    # [batch_size, num_heads, seq_len, head_dim_v] -> [batch_size, seq_len, num_heads, head_dim_v]
    # -> [batch_size, seq_len, d_v]
    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.view(batch_size, seq_len, d_v)
    
    # Apply output projection
    # attn_output: [batch_size, seq_len, d_v]
    # o_proj_weight: [d_model, d_v]
    # output: [batch_size, seq_len, d_model]
    output = torch.matmul(attn_output, o_proj_weight.T)
    
    # Reshape back to original batch dimensions
    output = output.view(*batch_dims, seq_len, d_model)
    
    return output
    

    raise NotImplementedError


def run_multihead_self_attention_with_rope(
    d_model: int,
    num_heads: int,
    max_seq_len: int,
    theta: float,
    q_proj_weight: Float[Tensor, " d_k d_in"],
    k_proj_weight: Float[Tensor, " d_k d_in"],
    v_proj_weight: Float[Tensor, " d_v d_in"],
    o_proj_weight: Float[Tensor, " d_model d_v"],
    in_features: Float[Tensor, " ... sequence_length d_in"],
    token_positions: Int[Tensor, " ... sequence_length"] | None = None,
) -> Float[Tensor, " ... sequence_length d_out"]:
    """
    Given the key, query, and value projection weights of a naive unbatched
    implementation of multi-head attention, return the output of an optimized batched
    implementation. This implementation should handle the key, query, and value projections
    for all heads in a single matrix multiply.
    This version of MHA should include RoPE.
    In this case, the RoPE embedding dimension must be the head embedding dimension (d_model // num_heads).
    See section 3.2.2 of Vaswani et al., 2017.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        num_heads (int): Number of heads to use in multi-headed attention.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        theta (float): RoPE parameter.
        q_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the Q projection
        k_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the K projection
        v_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the V projection
        o_proj_weight (Float[Tensor, "d_model d_v"]): Weights for the output projection
        in_features (Float[Tensor, "... sequence_length d_in"]): Tensor to run your implementation on.
        token_positions (Int[Tensor, " ... sequence_length"] | None): Optional tensor with the positions of the tokens

    Returns:
        Float[Tensor, " ... sequence_length d_out"]: Tensor with the output of running your optimized, batched multi-headed attention
        implementation with the given QKV projection weights and input features.
    """
    from einops import einsum
    *batch_dims, seq_len, d_in = in_features.shape
    d_k = q_proj_weight.shape[0]
    d_v = v_proj_weight.shape[0]
    head_dim_k = d_k // num_heads
    head_dim_v = d_v // num_heads

    batch_size = 1
    for dim in batch_dims:
        batch_size *= dim
    
    x = in_features.reshape(batch_size,seq_len,d_in)

    # x: [batch_size, seq_len, d_in]
    Q = einsum(x, q_proj_weight ,"... s d_in,  d_k d_in -> ... s d_k")
    K = einsum(x, k_proj_weight ,"... s d_in,  d_k d_in -> ... s d_k")
    V = einsum(x, v_proj_weight ,"... s d_in,  d_v d_in -> ... s d_v")
    
    # Reshape for multi-head attention
    # [batch_size, seq_len, d_k] -> [batch_size, seq_len, num_heads, head_dim_k]
    # -> [batch_size, num_heads, seq_len, head_dim_k]
    Q = Q.view(batch_size, seq_len, num_heads, head_dim_k).transpose(1, 2)
    K = K.view(batch_size, seq_len, num_heads, head_dim_k).transpose(1, 2)
    V = V.view(batch_size, seq_len, num_heads, head_dim_v).transpose(1, 2)

    from cs336_basics.p_3_5_3_rope import RotaryPositionEmbedding
    rope = RotaryPositionEmbedding(theta, head_dim_k, max_seq_len)
    # Apply RoPE to Q and K
    Q = rope.forward(Q, token_positions)
    K = rope.forward(K, token_positions)
    
    # Scaled dot-product attention
    # Q: [batch_size, num_heads, seq_len, head_dim_k]
    # K: [batch_size, num_heads, seq_len, head_dim_k]
    # Attention scores: [batch_size, num_heads, seq_len, seq_len]
    scores = torch.matmul(Q, K.transpose(-2, -1)) / (head_dim_k ** 0.5)
    
    # Apply causal mask (for autoregressive attention)
    causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=scores.device), diagonal=1).bool()
    scores = scores.masked_fill(causal_mask, float('-inf'))
    
    # Apply softmax
    from cs336_basics.p_3_5_4_softmax import Softmax
    softmax = Softmax()
    attn_weights = softmax.forward(scores)  # [batch_size, num_heads, seq_len, seq_len]
    
    # Apply attention to values
    # attn_weights: [batch_size, num_heads, seq_len, seq_len]
    # V: [batch_size, num_heads, seq_len, head_dim_v]
    # output: [batch_size, num_heads, seq_len, head_dim_v]
    attn_output = torch.matmul(attn_weights, V)
    
    # Concatenate heads
    # [batch_size, num_heads, seq_len, head_dim_v] -> [batch_size, seq_len, num_heads, head_dim_v]
    # -> [batch_size, seq_len, d_v]
    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.view(batch_size, seq_len, d_v)
    
    # Apply output projection
    # attn_output: [batch_size, seq_len, d_v]
    # o_proj_weight: [d_model, d_v]
    # output: [batch_size, seq_len, d_model]
    output = torch.matmul(attn_output, o_proj_weight.T)
    
    # Reshape back to original batch dimensions
    output = output.view(*batch_dims, seq_len, d_model)
    
    return output





    # raise NotImplementedError


def run_rope(
    d_k: int,
    theta: float,
    max_seq_len: int,
    in_query_or_key: Float[Tensor, " ... sequence_length d_k"],
    token_positions: Int[Tensor, " ... sequence_length"],
) -> Float[Tensor, " ... sequence_length d_k"]:
    """
    Run RoPE for a given input tensor.

    Args:
        d_k (int): Embedding dimension size for the query or key tensor.
        theta (float): RoPE parameter.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        in_query_or_key (Float[Tensor, "... sequence_length d_k"]): Input tensor to run RoPE on.
        token_positions (Int[Tensor, "... sequence_length"]): Tensor of shape (batch_size, sequence_length) with the token positions
    Returns:
        Float[Tensor, " ... sequence_length d_k"]: Tensor with RoPEd input.
    """
    from cs336_basics.p_3_5_3_rope import RotaryPositionEmbedding
    rope = RotaryPositionEmbedding(theta, d_k, max_seq_len)
    output = rope.forward(in_query_or_key, token_positions)

    return output
    # raise NotImplementedError


def run_transformer_block(
    d_model: int,
    num_heads: int,
    d_ff: int,
    max_seq_len: int,
    theta: float,
    weights: dict[str, Tensor],
    in_features: Float[Tensor, " batch sequence_length d_model"],
) -> Float[Tensor, " batch sequence_length d_model"]:
    """
    Given the weights of a pre-norm Transformer block and input features,
    return the output of running the Transformer block on the input features.

    This function should use RoPE.
    Depending on your implementation, you may simply need to pass the relevant args
    to your TransformerBlock constructor, or you may need to initialize your own RoPE
    class and pass that instead.

    Args:
        d_model (int): The dimensionality of the Transformer block input.
        num_heads (int): Number of heads to use in multi-headed attention. `d_model` must be
            evenly divisible by `num_heads`.
        d_ff (int): Dimensionality of the feed-forward inner layer.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        theta (float): RoPE parameter.
        weights (dict[str, Tensor]):
            State dict of our reference implementation.
            The keys of this dictionary are:
            - `attn.q_proj.weight`
                The query projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.q_proj.weight == torch.cat([q_heads.0.weight, ..., q_heads.N.weight], dim=0)`.
            - `attn.k_proj.weight`
                The key projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.k_proj.weight == torch.cat([k_heads.0.weight, ..., k_heads.N.weight], dim=0)`.
            - `attn.v_proj.weight`
                The value projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_v),
                so `attn.v_proj.weight == torch.cat([v_heads.0.weight, ..., v_heads.N.weight], dim=0)`.
            - `attn.output_proj.weight`
                Weight of the multi-head self-attention output projection
                Shape is (d_model, d_model).
            - `ln1.weight`
                Weights of affine transform for the first RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `ffn.w1.weight`
                Weight of the first linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `ffn.w2.weight`
                Weight of the second linear transformation in the FFN.
                Shape is (d_ff, d_model).
            - `ffn.w3.weight`
                Weight of the third linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `ln2.weight`
                Weights of affine transform for the second RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
        in_features (Float[Tensor, "batch sequence_length d_model"]):
            Tensor to run your implementation on.

    Returns:
        Float[Tensor, "batch sequence_length d_model"] Tensor with the output of
        running the Transformer block on the input features while using RoPE.
    """
    raise NotImplementedError


def run_transformer_lm(
    vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    rope_theta: float,
    weights: dict[str, Tensor],
    in_indices: Int[Tensor, " batch_size sequence_length"],
) -> Float[Tensor, " batch_size sequence_length vocab_size"]:
    """Given the weights of a Transformer language model and input indices,
    return the output of running a forward pass on the input indices.

    This function should use RoPE.

    Args:
        vocab_size (int): The number of unique items in the output vocabulary to be predicted.
        context_length (int): The maximum number of tokens to process at once.
        d_model (int): The dimensionality of the model embeddings and sublayer outputs.
        num_layers (int): The number of Transformer layers to use.
        num_heads (int): Number of heads to use in multi-headed attention. `d_model` must be
            evenly divisible by `num_heads`.
        d_ff (int): Dimensionality of the feed-forward inner layer (section 3.3).
        rope_theta (float): The RoPE $\Theta$ parameter.
        weights (dict[str, Tensor]):
            State dict of our reference implementation. {num_layers} refers to an
            integer between `0` and `num_layers - 1` (the layer index).
            The keys of this dictionary are:
            - `token_embeddings.weight`
                Token embedding matrix. Shape is (vocab_size, d_model).
            - `layers.{num_layers}.attn.q_proj.weight`
                The query projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.q_proj.weight == torch.cat([q_heads.0.weight, ..., q_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.k_proj.weight`
                The key projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.k_proj.weight == torch.cat([k_heads.0.weight, ..., k_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.v_proj.weight`
                The value projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_v),
                so `attn.v_proj.weight == torch.cat([v_heads.0.weight, ..., v_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.output_proj.weight`
                Weight of the multi-head self-attention output projection
                Shape is ((d_model / num_heads) * num_heads, d_model).
            - `layers.{num_layers}.ln1.weight`
                Weights of affine transform for the first RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `layers.{num_layers}.ffn.w1.weight`
                Weight of the first linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `layers.{num_layers}.ffn.w2.weight`
                Weight of the second linear transformation in the FFN.
                Shape is (d_ff, d_model).
            - `layers.{num_layers}.ffn.w3.weight`
                Weight of the third linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `layers.{num_layers}.ln2.weight`
                Weights of affine transform for the second RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `ln_final.weight`
                Weights of affine transform for RMSNorm applied to the output of the final transformer block.
                Shape is (d_model, ).
            - `lm_head.weight`
                Weights of the language model output embedding.
                Shape is (vocab_size, d_model).
        in_indices (Int[Tensor, "batch_size sequence_length"]) Tensor with input indices to run the language model on. Shape is (batch_size, sequence_length), where
            `sequence_length` is at most `context_length`.

    Returns:
        Float[Tensor, "batch_size sequence_length vocab_size"]: Tensor with the predicted unnormalized
        next-word distribution for each token.
    """
    raise NotImplementedError


def run_rmsnorm(
    d_model: int,
    eps: float,
    weights: Float[Tensor, " d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    """Given the weights of a RMSNorm affine transform,
    return the output of running RMSNorm on the input features.

    Args:
        d_model (int): The dimensionality of the RMSNorm input.
        eps: (float): A value added to the denominator for numerical stability.
        weights (Float[Tensor, "d_model"]): RMSNorm weights.
        in_features (Float[Tensor, "... d_model"]): Input features to run RMSNorm on. Can have arbitrary leading
            dimensions.

    Returns:
        Float[Tensor,"... d_model"]: Tensor of with the same shape as `in_features` with the output of running
        RMSNorm of the `in_features`.
    """
    from cs336_basics.p_3_5_1_rmsnorm import RMSNorm
    rms = RMSNorm(d_model,eps)
    rms.load_state_dict({"weight": weights})
    rms_res = rms.forward(in_features)
    return rms_res
    # raise NotImplementedError


def run_silu(in_features: Float[Tensor, " ..."]) -> Float[Tensor, " ..."]:
    """Given a tensor of inputs, return the output of applying SiLU
    to each element.

    Args:
        in_features(Float[Tensor, "..."]): Input features to run SiLU on. Shape is arbitrary.

    Returns:
        Float[Tensor,"..."]: of with the same shape as `in_features` with the output of applying
        SiLU to each element.
    """
    raise NotImplementedError


def run_get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Given a dataset (a 1D numpy array of integers) and a desired batch size and
    context length, sample language modeling input sequences and their corresponding
    labels from the dataset.

    Args:
        dataset (np.array): 1D numpy array of integer token IDs in the dataset.
        batch_size (int): Desired batch size to sample.
        context_length (int): Desired context length of each sampled example.
        device (str): PyTorch device string (e.g., 'cpu' or 'cuda:0') indicating the device
            to place the sampled input sequences and labels on.

    Returns:
        Tuple of torch.LongTensors of shape (batch_size, context_length). The first tuple item
        is the sampled input sequences, and the second tuple item is the corresponding
        language modeling labels.
    """
    raise NotImplementedError


def run_softmax(in_features: Float[Tensor, " ..."], dim: int) -> Float[Tensor, " ..."]:
    """
    Given a tensor of inputs, return the output of softmaxing the given `dim`
    of the input.

    Args:
        in_features (Float[Tensor, "..."]): Input features to softmax. Shape is arbitrary.
        dim (int): Dimension of the `in_features` to apply softmax to.

    Returns:
        Float[Tensor, "..."]: Tensor of with the same shape as `in_features` with the output of
        softmax normalizing the specified `dim`.
    """
    from cs336_basics.p_3_5_4_softmax import Softmax
    softmax = Softmax()
    softmax_res = softmax.forward(in_features)
    return softmax_res
    raise NotImplementedError


def run_cross_entropy(
    inputs: Float[Tensor, " batch_size vocab_size"], targets: Int[Tensor, " batch_size"]
) -> Float[Tensor, ""]:
    """Given a tensor of inputs and targets, compute the average cross-entropy
    loss across examples.

    Args:
        inputs (Float[Tensor, "batch_size vocab_size"]): inputs[i][j] is the
            unnormalized logit of jth class for the ith example.
        targets (Int[Tensor, "batch_size"]): Tensor of shape (batch_size,) with the index of the correct class.
            Each value must be between 0 and `num_classes - 1`.

    Returns:
        Float[Tensor, ""]: The average cross-entropy loss across examples.
    """
    raise NotImplementedError


def run_gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    """Given a set of parameters, clip their combined gradients to have l2 norm at most max_l2_norm.

    Args:
        parameters (Iterable[torch.nn.Parameter]): collection of trainable parameters.
        max_l2_norm (float): a positive value containing the maximum l2-norm.

    The gradients of the parameters (parameter.grad) should be modified in-place.
    """
    raise NotImplementedError


def get_adamw_cls() -> Any:
    """
    Returns a torch.optim.Optimizer that implements AdamW.
    """
    raise NotImplementedError


def run_get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):
    """
    Given the parameters of a cosine learning rate decay schedule (with linear
    warmup) and an iteration number, return the learning rate at the given
    iteration under the specified schedule.

    Args:
        it (int): Iteration number to get learning rate for.
        max_learning_rate (float): alpha_max, the maximum learning rate for
            cosine learning rate schedule (with warmup).
        min_learning_rate (float): alpha_min, the minimum / final learning rate for
            the cosine learning rate schedule (with warmup).
        warmup_iters (int): T_w, the number of iterations to linearly warm-up
            the learning rate.
        cosine_cycle_iters (int): T_c, the number of cosine annealing iterations.

    Returns:
        Learning rate at the given iteration under the specified schedule.
    """
    raise NotImplementedError


def run_save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
):
    """
    Given a model, optimizer, and an iteration number, serialize them to disk.

    Args:
        model (torch.nn.Module): Serialize the state of this model.
        optimizer (torch.optim.Optimizer): Serialize the state of this optimizer.
        iteration (int): Serialize this value, which represents the number of training iterations
            we've completed.
        out (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to serialize the model, optimizer, and iteration to.
    """
    raise NotImplementedError


def run_load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    """
    Given a serialized checkpoint (path or file-like object), restore the
    serialized state to the given model and optimizer.
    Return the number of iterations that we previously serialized in
    the checkpoint.

    Args:
        src (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to serialized checkpoint.
        model (torch.nn.Module): Restore the state of this model.
        optimizer (torch.optim.Optimizer): Restore the state of this optimizer.
    Returns:
        int: the previously-serialized number of iterations.
    """
    raise NotImplementedError


def get_tokenizer(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    special_tokens: list[str] | None = None,
) -> Any:
    """Given a vocabulary, a list of merges, and a list of special tokens,
    return a BPE tokenizer that uses the provided vocab, merges, and special tokens.

    Args:
        vocab (dict[int, bytes]): The tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
            to bytes (token bytes)
        merges (list[tuple[bytes, bytes]]): BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
            representing that <token1> was merged with <token2>.
            Merges are ordered by order of creation.
        special_tokens (list[str] | None): A list of string special tokens for the tokenizer. These strings will never
            be split into multiple tokens, and will always be kept as a single token.

    Returns:
        A BPE tokenizer that uses the provided vocab, merges, and special tokens.
    """
    from cs336_basics.p_2_6_decoding import Tokenizer
    return Tokenizer(vocab,merges,special_tokens)

    raise NotImplementedError


def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """
    from multiprocessing import Pool
    import regex as re
    from typing import Union,BinaryIO
    from collections import defaultdict,Counter
    import os
    from cs336_basics.pretokenization_example import find_chunk_boundaries
    base_PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    initialize_vocab = {1: b'!', 2: b'"', 3: b'#', 4: b'$', 5: b'%', 6: b'&', 7: b"'", 8: b'(', 9: b')', 10: b'*', 11: b'+', 12: b',', 13: b'-', 14: b'.', 15: b'/', 16: b'0', 17: b'1', 18: b'2', 19: b'3', 20: b'4', 21: b'5', 22: b'6', 23: b'7', 24: b'8', 25: b'9', 26: b':', 27: b';', 28: b'<', 29: b'=', 30: b'>', 31: b'?', 32: b'@', 33: b'A', 34: b'B', 35: b'C', 36: b'D', 37: b'E', 38: b'F', 39: b'G', 40: b'H', 41: b'I', 42: b'J', 43: b'K', 44: b'L', 45: b'M', 46: b'N', 47: b'O', 48: b'P', 49: b'Q', 50: b'R', 51: b'S', 52: b'T', 53: b'U', 54: b'V', 55: b'W', 56: b'X', 57: b'Y', 58: b'Z', 59: b'[', 60: b'\\', 61: b']', 62: b'^', 63: b'_', 64: b'`', 65: b'a', 66: b'b', 67: b'c', 68: b'd', 69: b'e', 70: b'f', 71: b'g', 72: b'h', 73: b'i', 74: b'j', 75: b'k', 76: b'l', 77: b'm', 78: b'n', 79: b'o', 80: b'p', 81: b'q', 82: b'r', 83: b's', 84: b't', 85: b'u', 86: b'v', 87: b'w', 88: b'x', 89: b'y', 90: b'z', 91: b'{', 92: b'|', 93: b'}', 94: b'~', 95: b'\xa1', 96: b'\xa2', 97: b'\xa3', 98: b'\xa4', 99: b'\xa5', 100: b'\xa6', 101: b'\xa7', 102: b'\xa8', 103: b'\xa9', 104: b'\xaa', 105: b'\xab', 106: b'\xac', 107: b'\xae', 108: b'\xaf', 109: b'\xb0', 110: b'\xb1', 111: b'\xb2', 112: b'\xb3', 113: b'\xb4', 114: b'\xb5', 115: b'\xb6', 116: b'\xb7', 117: b'\xb8', 118: b'\xb9', 119: b'\xba', 120: b'\xbb', 121: b'\xbc', 122: b'\xbd', 123: b'\xbe', 124: b'\xbf', 125: b'\xc0', 126: b'\xc1', 127: b'\xc2', 128: b'\xc3', 129: b'\xc4', 130: b'\xc5', 131: b'\xc6', 132: b'\xc7', 133: b'\xc8', 134: b'\xc9', 135: b'\xca', 136: b'\xcb', 137: b'\xcc', 138: b'\xcd', 139: b'\xce', 140: b'\xcf', 141: b'\xd0', 142: b'\xd1', 143: b'\xd2', 144: b'\xd3', 145: b'\xd4', 146: b'\xd5', 147: b'\xd6', 148: b'\xd7', 149: b'\xd8', 150: b'\xd9', 151: b'\xda', 152: b'\xdb', 153: b'\xdc', 154: b'\xdd', 155: b'\xde', 156: b'\xdf', 157: b'\xe0', 158: b'\xe1', 159: b'\xe2', 160: b'\xe3', 161: b'\xe4', 162: b'\xe5', 163: b'\xe6', 164: b'\xe7', 165: b'\xe8', 166: b'\xe9', 167: b'\xea', 168: b'\xeb', 169: b'\xec', 170: b'\xed', 171: b'\xee', 172: b'\xef', 173: b'\xf0', 174: b'\xf1', 175: b'\xf2', 176: b'\xf3', 177: b'\xf4', 178: b'\xf5', 179: b'\xf6', 180: b'\xf7', 181: b'\xf8', 182: b'\xf9', 183: b'\xfa', 184: b'\xfb', 185: b'\xfc', 186: b'\xfd', 187: b'\xfe', 188: b'\xff', 189: b'\x00', 190: b'\x01', 191: b'\x02', 192: b'\x03', 193: b'\x04', 194: b'\x05', 195: b'\x06', 196: b'\x07', 197: b'\x08', 198: b'\t', 199: b'\n', 200: b'\x0b', 201: b'\x0c', 202: b'\r', 203: b'\x0e', 204: b'\x0f', 205: b'\x10', 206: b'\x11', 207: b'\x12', 208: b'\x13', 209: b'\x14', 210: b'\x15', 211: b'\x16', 212: b'\x17', 213: b'\x18', 214: b'\x19', 215: b'\x1a', 216: b'\x1b', 217: b'\x1c', 218: b'\x1d', 219: b'\x1e', 220: b'\x1f', 221: b' ', 222: b'\x7f', 223: b'\x80', 224: b'\x81', 225: b'\x82', 226: b'\x83', 227: b'\x84', 228: b'\x85', 229: b'\x86', 230: b'\x87', 231: b'\x88', 232: b'\x89', 233: b'\x8a', 234: b'\x8b', 235: b'\x8c', 236: b'\x8d', 237: b'\x8e', 238: b'\x8f', 239: b'\x90', 240: b'\x91', 241: b'\x92', 242: b'\x93', 243: b'\x94', 244: b'\x95', 245: b'\x96', 246: b'\x97', 247: b'\x98', 248: b'\x99', 249: b'\x9a', 250: b'\x9b', 251: b'\x9c', 252: b'\x9d', 253: b'\x9e', 254: b'\x9f', 255: b'\xa0', 256: b'\xad'}
    special_pattern = '|'.join(re.escape(token) for token in special_tokens)
    # 将 special tokens 模式放在最前面，优先匹配
    PAT = f"{special_pattern}|{base_PAT}"

    # 异常处理
    if not os.path.exists(input_path):
        raise FileExistsError("Do not exist this file,Please check it agagin")
    if vocab_size <= len(special_tokens):
        raise ValueError("Vocab size must greatter than the number of special token")
    
    str_chunks_list = []
    # 文本分段
    with open(input_path,'rb') as f:
        boundaries = find_chunk_boundaries(f, 4, b"<|endoftext|>")
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            str_chunk_list:list[str] = re.findall(PAT, chunk)
            str_chunks_list.extend(str_chunk_list)

    vocab:dict = {}
    next_id = 0
    merger:tuple[list(bytes,bytes)] = []
    int_words_list:list[list[int]] = []

    for special_token in special_tokens:
        vocab[next_id] = special_token.encode('utf-8')
        next_id += 1
    vocab.update(initialize_vocab)
    next_id = 257

    # 反向词汇表
    reverse_vocab = {v:k for k,v in vocab.items()}
    # 最大添加步数
    max_merge = vocab_size - len(vocab)


    for str_word in str_chunks_list:
        if str_word == '<|endoftext|>':
            continue
        byte_word = str_word.encode('utf-8')
        int_word = list(byte_word)
        int_words_list.append(int_word)



    for _ in range(max_merge):
        # 计算最大pair
        pair_dict = defaultdict(int)
        for int_word in int_words_list:
            if len(int_word) <= 1:
                continue
            for i in range(len(int_word)-1):
                pair = (int_word[i],int_word[i+1])
                pair_dict[pair] += 1

        if not pair_dict:
            break
        
        pair_dict = Counter(pair_dict)
        max_freq = max(pair_dict.values())
        same_freq_pair = [k for k,v in pair_dict.items() if v== max_freq]
        
        if len(same_freq_pair) != 1:    # [(258,269),(32,100)]
            # print(same_freq_pair)
            comparable_pairs_int_byte = []
            for left_int,right_int in same_freq_pair:
                if left_int > 256:
                    left_byte = vocab[left_int]                 
                else: 
                    left_byte  = bytes([left_int])

                if right_int > 256:
                    right_byte = vocab[right_int] 
                else:
                    right_byte = bytes([right_int])
                comparable_pairs_int_byte.append(((left_int,right_int),(left_byte,right_byte)))
            def get_sorted_key(item):
                max_byte = item[1]
                return max_byte

            comparable_pairs_int_byte_sorted = sorted(comparable_pairs_int_byte,key = get_sorted_key,reverse=True)
            # print(comparable_pairs_int_byte_sorted)
            res = [item[0] for item in comparable_pairs_int_byte_sorted ][0]
            # print(f"res{res}")
            most_common_pair = res
        else:
            most_common_pair = pair_dict.most_common(1)[0][0]


        if most_common_pair[0] > 256:
            left_byte = vocab[most_common_pair[0]]
        else:
            left_byte = bytes([most_common_pair[0]])
            # left_byte = vocab[most_common_pair[0]]
        if most_common_pair[1] > 256:
            right_byte = vocab[most_common_pair[1]]
        else:
            right_byte = bytes([most_common_pair[1]])


        new_pair = left_byte+right_byte
        merger.append((left_byte,right_byte))
        vocab[next_id] = new_pair
        reverse_vocab[new_pair] = next_id


        # 合并：
        new_int_words_list = []
        for int_word in int_words_list:
            if len(int_word) <= 1:
                continue
            i = 0
            new_word = []
            while i < len(int_word):
                if i < len(int_word) -1 and int_word[i] == most_common_pair[0] and int_word[i+1] == most_common_pair[1]:
                    new_word.append(next_id)
                    i += 2
                else:
                    new_word.append(int_word[i])
                    i += 1
            new_int_words_list.append(new_word)
        
        int_words_list = new_int_words_list
        next_id += 1


    return vocab,merger

    raise NotImplementedError
