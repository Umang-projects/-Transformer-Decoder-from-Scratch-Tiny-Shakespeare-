class FinalProjectionFromScratch(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        # 1) Create weight & bias as nn.Parameter so they’re learnable
        self.W_out = nn.Parameter(torch.empty(d_model, vocab_size))
        self.b_out = nn.Parameter(torch.empty(vocab_size))

        # 2) Initialize (Xavier for the weight, zeros for the bias)
        nn.init.xavier_uniform_(self.W_out)
        nn.init.zeros_(self.b_out)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        logits = X.matmul(self.W_out) + self.b_out
        return logits