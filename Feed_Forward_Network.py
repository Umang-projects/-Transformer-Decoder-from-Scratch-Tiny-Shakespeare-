
class FFN(nn.Module):
  def __init__(self,d_model,d_ff):
    super().__init__()
    #1 LinearLayer 1
    self.W1=nn.Parameter(torch.empty(d_model,d_ff))#(d_model*d_ff)
    self.b1=nn.Parameter(torch.empty(d_ff))

    #2 LinearLayer 2
    self.W2=nn.Parameter(torch.empty(d_ff,d_model))#(d_model*d_ff)
    self.b2=nn.Parameter(torch.empty(d_model))

    # 3) Initialize weights (Xavier) and biases (zeros)
    nn.init.xavier_uniform_(self.W1)
    nn.init.zeros_(self.b1)
    nn.init.xavier_uniform_(self.W2)
    nn.init.zeros_(self.b2)

    self.relu=nn.ReLU()

  def forward(self,X:torch.Tensor)->torch.Tensor:
    # print(X.shape)
    # print(self.W1.shape)
    # print(self.W2.shape)
    hidden=X@self.W1+self.b1
    hidden_relu=self.relu(hidden)
    out=hidden_relu@self.W2+self.b2
    return out