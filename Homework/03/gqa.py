import torch
import torch.nn.functional as F


def scaled_dot_product_gqa(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    is_causal: bool = True,
    need_weights: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Scaled Dot-Product attention in grouped manner.

    Args:
        query (torch.Tensor): Query tensor of shape [batch size; seq len; num heads; hidden dim]
        key (torch.Tensor): Key tensor of shape [batch size; kv seq len; num kv heads; hidden dim]
        value (torch.Tensor): Value tensor of shape [batch size; kv seq len; num kv heads; hidden dim]
        is_causal (bool): Whether causal mask of attention should be used
        need_weights (bool): Whether attention weights should be returned

    Returns:
        2-tuple of torch.Tensor:
            - Attention output with shape [batch size; seq len; num heads; hidden dim]
            - (Optional) Attention weights with shape [batch size; num heads; seq len; kv seq len].
                Only returned if 'need_weights' is True.
    """
    batch_size, seq_len, num_heads, hidden_dim = query.shape
    kv_seq_len = key.size(1)
    kv_num_heads = key.size(2)
    if kv_num_heads > num_heads:
        raise ValueError(
            "Number of key heads should be greater or equal to number of query heads"
        )

    query = query.permute(0, 2, 1, 3)
    key = key.permute(0, 2, 1, 3)
    value = value.permute(0, 2, 1, 3)
    query = query / hidden_dim**0.5

    query = query.view(
        batch_size, kv_num_heads, num_heads // kv_num_heads, seq_len, hidden_dim
    ).permute(0, 2, 1, 3, 4)

    similarity = torch.einsum("b g h n d, b h s d -> b g h n s", query, key)

    if is_causal:
        mask = torch.ones((batch_size, seq_len, kv_seq_len), dtype=torch.bool).tril_()
        similarity.masked_fill_(~mask, -torch.inf)

    attention = F.softmax(similarity, dim=-1)
    out = torch.einsum(
        "b g h n s, b h s d -> b g h n d",
        attention,
        value,
    )
    out = (
        out.permute(0, 3, 2, 1, 4).contiguous().view(-1, seq_len, num_heads, hidden_dim)
    )

    attention = (
        attention.permute(0, 2, 1, 3, 4)
        .contiguous()
        .view(batch_size, num_heads, seq_len, kv_seq_len)
    )

    return (out, attention) if need_weights else out
