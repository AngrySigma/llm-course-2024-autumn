import torch


def compute_alibi(num_heads: int, seq_len: int) -> torch.Tensor:
    """
    Compute ALiBi for a sequence.

    ALiBi can be used not only with causal models.
    In this case, the biases will be symmetrical about the diagonal up to the sign.

    Args:
        num_heads (int): Number of attention heads.
        seq_len (int): Sequence length.

    Returns:
        torch.Tensor: A tensor containing ALiBi to be added to attention scores.
    """
    start = 2 ** (-8 / num_heads)
    slopes = start ** torch.arange(1, num_heads + 1)
    pos_diff = torch.arange(seq_len).unsqueeze(0) - torch.arange(seq_len).unsqueeze(1)
    return torch.einsum("h,jk->hjk", slopes, pos_diff)


if __name__ == "__main__":
    bias = compute_alibi(4, 4)
    print(bias)
