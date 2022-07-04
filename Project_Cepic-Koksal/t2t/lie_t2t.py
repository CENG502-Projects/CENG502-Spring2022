import torch
import torch.nn as nn
import numpy as np
from .token_transformer import Token_transformer
from .token_performer import Token_performer

class LIE(nn.Module):
    def __init__(self, token_dim):
        super().__init__()

        self.unfold = nn.Unfold(kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.lin1 = nn.Linear(token_dim*3*3, token_dim)
        self.relu = nn.ReLU()
        self.lin2 = nn.Linear(token_dim, token_dim*3*3)
        self.sigmoid = nn.Sigmoid()
        self.lin3 = nn.Linear(token_dim*3*3, token_dim)

    def forward(self, x):
        t = self.unfold(x).transpose(1,2)
        s = self.sigmoid(self.lin2(self.relu(self.lin1(t))))
        return self.lin3(t*s)

class LIE_T2T(nn.Module):
    def __init__(self, tokens_type='performer', k=3, s=2, in_chans=3, token_dim=64, lie=True):
        super().__init__()

        self.unfold = nn.Unfold(kernel_size=(k, k), stride=(s, s), padding=(1, 1))
        if tokens_type == 'transformer':
            self.attention = Token_transformer(dim=in_chans * k * k, in_dim=token_dim, num_heads=1, mlp_ratio=1.0)
        elif tokens_type == 'performer':
            self.attention = Token_performer(dim=in_chans * k * k, in_dim=token_dim, kernel_ratio=0.5)
        if lie == True:
            self.lie = LIE(token_dim=token_dim)
        else:
            self.lie = None
    def forward(self, x, W, H):

        x = self.unfold(x).transpose(1, 2)

        W = int(np.ceil(W/self.unfold.stride[0]))
        H = int(np.ceil(H/self.unfold.stride[1]))

        x = self.attention(x)

        B, _, C = x.shape

        if self.lie is not None:
            x = x.transpose(1,2).reshape(B, C, W, H)
            x = self.lie(x)
            W = int(np.ceil(W/self.lie.unfold.stride[0]))
            H = int(np.ceil(H/self.lie.unfold.stride[1]))

        x = x.transpose(1, 2).reshape(B, C, W, H)

        return x, W, H

class LIE_module(nn.Module):
    def __init__(self, N=2, K=4, tokens_type='performer', embed_dim=256, token_dim=32, lie=True):
        super().__init__()

        lie_ch = [3] + (N-1) * [token_dim]

        stride = [2, 2, 2] + (K-3) * [1]

        self.blocks = nn.ModuleList([
            LIE_T2T(tokens_type=tokens_type, k=3, s=2, in_chans=lie_ch[i], token_dim=token_dim, lie=lie) for i in range(N)] +
            [LIE_T2T(tokens_type=tokens_type, k=3, s=stride[i], in_chans=token_dim, token_dim=token_dim, lie=lie) for i in range(K)])

        self.final_unfold = nn.Unfold(kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.project = nn.Linear(token_dim * 3 * 3, embed_dim)

    def forward(self, x):
        _, _, W, H = x.shape

        for blk in self.blocks:
            x, W, H = blk(x, W, H)

        x = self.final_unfold(x).transpose(1, 2)

        x = self.project(x).permute(0,2,1)

        return x

