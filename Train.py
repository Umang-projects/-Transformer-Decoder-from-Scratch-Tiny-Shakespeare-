class Decoder_Block(nn.Module):
  def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
    super().__init__()
    self.self_attn = MultiheadAttentionPerHead(d_model, num_heads)
    self.layernorm1 = nn.LayerNorm(d_model)
    self.dropout1 = nn.Dropout(dropout)
    self.ffn = FFN(d_model, d_ff)
    self.layernorm2 = nn.LayerNorm(d_model)
    self.dropout2 = nn.Dropout(dropout)

  def forward(self, X, mask):
    attn_output, attn_weights = self.self_attn(X, mask)
    
    X_plus_attn = X + self.dropout1(attn_output)
    X_norm1 = self.layernorm1(X_plus_attn)

    ffn_output = self.ffn(X_norm1)
    
    X_plus_ffn = X_norm1 + self.dropout2(ffn_output)
    out = self.layernorm2(X_plus_ffn)
    return out, attn_weights