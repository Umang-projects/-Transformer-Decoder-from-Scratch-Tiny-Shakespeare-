class MultiheadAttentionPerHead(nn.Module):
    def __init__(self, d_model: int, heads: int):
        super().__init__()
        assert d_model % heads == 0
        self.heads = heads
        self.d_model = d_model
        self.dk = d_model // heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor = None):
        b, l, _ = x.shape
        # Project and reshape
        q = self.q_proj(x).view(b, l, self.heads, self.dk).transpose(1, 2)
        k = self.k_proj(x).view(b, l, self.heads, self.dk).transpose(1, 2)
        v = self.v_proj(x).view(b, l, self.heads, self.dk).transpose(1, 2)

        # Calculate scores
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.dk)
        if attn_mask is not None:
            scores = scores + attn_mask
        
        # This is the weights tensor. Its shape is [B, h, L, L] -- a 4D tensor!
        weights = torch.softmax(scores, dim=-1)

        # Apply attention and reshape for output projection
        head_out = weights @ v
        head_out = head_out.transpose(1, 2).contiguous().view(b, l, self.d_model)
        
        out = self.out_proj(head_out)
        return out, weights
