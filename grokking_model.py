"""
Desc:       Reproduce the Neel Nanda Grokking (modulo addition) Paper: https://openreview.net/pdf?id=9XFSbDPmdW
            Their Code: https://github.com/mechanistic-interpretability-grokking/progress-measures-paper/blob/main/Grokking_Analysis.ipynb
            NOTE: A left out a bunch of stuff that ended up being irrelevant (from their paper)

Date:       2025, Nov 5 started
Author:     ryan.rtjj@gmail.com
"""
import glob
import os
from pathlib import Path
import pickle
import random

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
PRIME = 113

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

def generate_data(
        frac_train: float,
        P: int,
        seed: int = 0,
        save_filename: str | None = None,
    ) -> tuple[list[tuple], list[tuple], list[tuple], list[bool], list[bool]]:
    """
    Generates all permutations in [0, P) and splits into train and test sets

    @return:    [0]: All data, e.g. [(0, 0, 113), (0, 1, 113), ...]
                [1]: list of train tuples, e.g. [(0, 0, 113), (0, 7, 113), ...]
                [2]: list of test tuples, e.g. [(0, 1, 113), (0, 2, 113), ...] 
                [3]: bool mask representing train samples, e.g. [True, False, ...]
                [4]: bool mask representing test samples, e.g. [False, True, ...]
    """
    pairs = [(i, j, P) for i in range(P) for j in range(P)]
    random.seed(seed)
    random.shuffle(pairs)
    num_train = int(frac_train * len(pairs))

    train_pairs = pairs[:num_train]
    test_pairs = pairs[num_train:]

    is_train = []
    is_test = []

    for x in range(P):
        for y in range(P):
            if (x, y, P) in train_pairs:
                is_train.append(True)
                is_test.append(False)
            else:
                is_train.append(False)
                is_test.append(True)

    if save_filename:
        if not save_filename.suffix == '.pkl':
            save_filename = save_filename + '.pkl'

        with open(save_filename, 'wb') as f:
            pickle.dump({
                'pairs': pairs,
                'train_pairs': train_pairs,
                'test_pairs': test_pairs,
                'is_train': is_train,
                'is_test': is_test
            }, f)

        file_size = os.path.getsize(save_filename)
        file_size_mb = file_size / (1024 * 1024)

        print(f'Saved generated data to {save_filename}. File size: {file_size_mb:.3f} MB')

    return pairs, train_pairs, test_pairs, is_train, is_test

def prepare_dirs(dirs: list[Path], clear: bool = True):
    for _dir in dirs:
        try:
            os.mkdir(_dir)
        except:
            pass

        if clear:
            files = glob.glob(f'{_dir}/*')
            for f in files:
                os.remove(f)

def train(
        model: nn.Module,
        data_file: Path,
        checkpoints_dir: Path,
        batch_size: int = -1
    ):
    """
    Simple training loop
    """
    # Read the data
    with open(data_file, 'rb') as f:
        all_data = pickle.load(f)
    
    pairs = all_data['pairs']
    train_pairs = all_data['train_pairs']
    test_pairs = all_data['test_pairs']

    # Generate labels
    labels = [(a + b) % P for (a, b, P) in pairs]

    # Compute batch_size
    if batch_size == -1:
        


if __name__ == '__main__':
    CHECKPOINTS_DIR = Path('checkpoints')
    DATA_DIR = Path('datasets')
    prepare_dirs([CHECKPOINTS_DIR, DATA_DIR], True)
    generate_data(0.3, P=PRIME, save_filename=DATA_DIR / 'debug.pkl')
    # model = Transformer()