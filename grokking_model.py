"""
Desc:       Reproduce the Neel Nanda Grokking (modulo addition) Paper: https://openreview.net/pdf?id=9XFSbDPmdW
            Their Code: https://github.com/mechanistic-interpretability-grokking/progress-measures-paper/blob/main/Grokking_Analysis.ipynb
            NOTE: A left out a bunch of stuff that ended up being irrelevant (from their paper)

Date:       2025, Nov 5 started
Author:     ryan.rtjj@gmail.com
"""
import einops
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

# +-----------+
# | Model def |
# +-----------+
D_VOCAB = 113
D_MODEL = 128
D_HEAD = 32
D_MLP = 512
ACT_TYPE = 'ReLU'
NUM_HEADS = 4
N_CTX = 3

class Embed(nn.Module):
    """
    Simple dictionary lookup.
    """
    def __init__(self, d_vocab: int = D_VOCAB, d_model: int = D_MODEL):
        super().__init__()
        self.W_E = nn.Parameter(torch.randn(d_model, d_vocab) / np.sqrt(d_model))

    def forward(self, x: torch.Tensor):
        """
        Note that x is shape (batch, n_ctx)
        """
        return torch.einsum('dbp -> bpd', self.W_E[:, x])

class Unembed(nn.Module):
    """
    Simple dictionary lookup
    """
    def __init__(self, d_vocab: int = D_VOCAB, d_model: int = D_MODEL):
        super().__init__()
        self.W_U = nn.Parameter(torch.randn(d_model, d_vocab) / np.sqrt(D_VOCAB))

    def forward(self, x: torch.Tensor):
        """
        Note that x is shape (batch, pos, d_model)
        """
        return x @ self.W_U

class Attention(nn.Module):
    def __init__(
            self,
            d_model: int = D_MODEL,
            num_heads: int = NUM_HEADS,
            d_head: int = D_HEAD,
            n_ctx: int = N_CTX,
    ):
        """
        The fact that W_K, W_Q, W_V weights are normalized by sqrt(d_model) seems kind of wonky.
        Shouldn't it be normalized by sqrt(d_head)?
        """
        super().__init__()
        self.d_head = d_head
        self.W_K = nn.Parameter(torch.randn(num_heads, d_head, d_model) / np.sqrt(d_model))
        self.W_Q = nn.Parameter(torch.randn(num_heads, d_head, d_model) / np.sqrt(d_model))
        self.W_V = nn.Parameter(torch.randn(num_heads, d_head, d_model) / np.sqrt(d_model))
        self.W_O = nn.Parameter(torch.randn(d_model, d_head * num_heads) / np.sqrt(d_model))

        self.register_buffer('mask', torch.tril(torch.ones((n_ctx, n_ctx))))

    def forward(self, x: torch.Tensor):
        """
        @param x:   Tensor of shape (batch, positions, d_model)
        """
        pos = x.shape[-2]
        k = torch.einsum('ihd,bpd -> biph', self.W_K, x)
        q = torch.einsum('ihd,bpd -> biph', self.W_Q, x)
        v = torch.einsum('ihd,bpd -> biph', self.W_V, x)
        attn_scores_pre = torch.einsum('biph,biqh -> biqp', k, q)

        # This is basically the upper triangle having negative infinity
        attn_scores_masked = torch.tril(attn_scores_pre) - 1e10 * (1 - self.mask[:pos, :pos])

        # Another weird normalization
        attn_matrix = F.softmax(attn_scores_masked / np.sqrt(self.d_head), dim=-1)
        z = torch.einsum('biph,biqh->biqh', v, attn_matrix)
        z_flat = einops.rearrange(z, 'b i q h -> b q (i h)')
        out = torch.einsum('df,bqf->bqd', self.W_O, z_flat)
        return out
    
class MLP(nn.Module):
    def __init__(
            self,
            d_model: int = D_MODEL,
            d_mlp: int = D_MLP,
            act_type: str = ACT_TYPE,
    ):
        super().__init__()

        self.W_up = nn.Parameter(torch.randn(d_mlp, d_model) / np.sqrt(d_model))
        self.b_up = nn.Parameter(torch.zeros(d_mlp))

        self.W_down = nn.Parameter(torch.randn(d_model, d_mlp) / np.sqrt(d_model))
        self.b_down = nn.Parameter(torch.zeros(d_model))    # <-- I think this might become 0

        assert act_type in ['ReLU', 'GeLU'], 'Expect act_type to be one of ReLU / GeLU'
        self.act_type = act_type

    def forward(self, x: torch.Tensor):
        x = torch.einsum('md,bpd->bpm', self.W_up, x) + self.b_up

        if self.act_type == 'ReLU':
            x = F.relu(x)
        elif self.act_type == 'GeLU':
            x = F.gelu(x)
        else:
            raise ValueError('self.act_type not ReLU / GeLU')

        x = torch.einsum('dm,bpm->bpd', self.W_down, x) + self.b_down
        return x

class TransformerBlock(nn.Module):
    """
    Just a simple chaining of Attention and MLP
    """
    def __init__(
            self,
            d_model: int = D_MODEL,
            d_mlp: int = D_MLP,
            d_head: int = D_HEAD,
            num_heads: int = NUM_HEADS,
            n_ctx: int = N_CTX,
            act_type: str = ACT_TYPE,
    ):
        super().__init__()
        self.attn = Attention(d_model, num_heads, d_head, n_ctx)
        self.mlp = MLP(d_model, d_mlp, act_type)

    def forward(self, x):
        x = x + self.attn(x)    # resid_mid
        x = x + self.mlp(x)     # resid_post
        return x

class Transformer(nn.Module):
    """
    Missing parts:
    -   PosEmbed:   Irrelevant; makes sense. a + b is the same as b + a;
                    model must just learn to ignore the '+' and only attend to a, b.

    Model Architecture:

    R^113 (1-hot)         R^128                  R^128                    R^512                      R^128                   R^113
    INPUT --- embed --> EMBEDDING --- attn --> MLP_INPUT --- mlp_up --> MLP_ACTS --- mlp_down --> EMBEDDING --- unembed --> LOGITS
                W_E              attention block               W_up         ReLU        W_down                     W_U
    """
    def __init__(
            self,
            num_layers: int = 1,
            d_vocab: int = D_VOCAB,
            d_model: int = D_MODEL,
            d_mlp: int = D_MLP,
            d_head: int = D_HEAD,
            num_heads: int = NUM_HEADS,
            n_ctx: int = N_CTX,
            act_type: str = ACT_TYPE,
    ):
        """
        Missing params:
        -   use_cache:  It's never True anyway
        -   use_ln:     Irrelevant, also makes it harder

        """
        super().__init__()
        self.embed = Embed(d_vocab, d_model)
        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model,
                d_mlp,
                d_head,
                num_heads,
                n_ctx,
                act_type
            ) for _ in range(num_layers)
        ])
        self.unembed = Unembed(d_vocab, d_model)

    def forward(self, x: torch.Tensor):
        x = self.embed(x)
        x = self.pos_embed(x)
        for block in self.blocks:
            x = block(x)
        x = self.unembed(x)
        return x

if __name__ == '__main__':
    model = Transformer()
    print(model)