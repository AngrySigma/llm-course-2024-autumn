import torch


def compute_attention(queries, keys, values) -> torch.Tensor:
    """
    queries- (BATCH_SIZE, SEQ_LENGTH, HIDDEN_DIM)
    keys- (BATCH_SIZE, SEQ_LENGTH, HIDDEN_DIM)
    values- (BATCH_SIZE, SEQ_LENGTH, HIDDEN_DIM)
    """
    return (
        torch.nn.functional.softmax(
            queries
            @ keys.transpose(-1, -2)
            / torch.sqrt(torch.tensor(queries.size()[-1])),
            dim=-1,
        )
        @ values
    )


def compute_multihead_attention(
    queries, keys, values, projection_matrix
) -> torch.Tensor:
    """
    queries- (BATCH_SIZE, N_HEADS, SEQ_LENGTH, DIM_PER_HEAD)
    keys- (BATCH_SIZE, N_HEADS, SEQ_LENGTH, DIM_PER_HEAD)
    values- (BATCH_SIZE, N_HEADS, SEQ_LENGTH, DIM_PER_HEAD)
    projection_matrix- (N_HEADS*DIM_PER_HEAD, N_HEADS*DIM_PER_HEAD)
    """
    batch_size, num_heads, seq_len, hidden_dim = queries.size()
    qks = (
        compute_attention(queries, keys, values)
        .transpose(1, 2)
        .contiguous()
        .view(batch_size, seq_len, num_heads * hidden_dim)
    )
    return qks @ projection_matrix.T


def compute_rotary_embeddings(x) -> torch.Tensor:
    batch, seq_len, n_heads, dim = x.size()
    base = 10000
    theta = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
    theta_idx = torch.arange(0, seq_len, dtype=theta.dtype)
    idx_theta = torch.outer(theta_idx, theta)
    complex_theta = torch.polar(torch.ones_like(idx_theta), idx_theta).reshape(
        1, seq_len, 1, dim // 2
    )

    x_reshaped = x.view(batch, seq_len, n_heads, dim // 2, 2)
    x_complex = torch.view_as_complex(x_reshaped)

    x_rotated = x_complex * complex_theta

    x_rotated = torch.view_as_real(x_rotated)
    x_rotated = x_rotated.view(batch, seq_len, n_heads, dim)
    return x_rotated
