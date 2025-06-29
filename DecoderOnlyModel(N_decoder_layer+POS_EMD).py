class DecoderOnlyModel(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, d_ff, n_layers, seq_len):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed   = nn.Embedding(seq_len, d_model)
        self.layers      = nn.ModuleList([
            Decoder_Block(d_model, n_heads, d_ff,dropout=0.1) for _ in range(n_layers)
        ])
        self.proj        = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids):
        B, L = input_ids.shape
        device = input_ids.device
        
        token_embeds = self.token_embed(input_ids)
        pos_ids = torch.arange(L, device=device).unsqueeze(0)
        pos_embeds = self.pos_embed(pos_ids)
        x = token_embeds + pos_embeds

        mask_bool = torch.triu(torch.ones(L, L, device=device), diagonal=1).bool()
        attn_mask = torch.zeros(L, L, device=device).masked_fill(mask_bool, float('-inf'))
        
        # using for visualizing the attention weights
        attention_scores_list = []
        
        for layer in self.layers:
            x, attn_scores = layer(x, attn_mask)  # Capture the weights here
            attention_scores_list.append(attn_scores)
            
        logits = self.proj(x)
        return logits, attention_scores_list


model = DecoderOnlyModel(
    vocab_size=tokenizer.vocab_size,
    d_model=d_model,
    n_heads=n_heads,
    d_ff=d_ff,
    n_layers=n_layers,
    seq_len=seq_len
)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)