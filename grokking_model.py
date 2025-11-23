"""
Desc:       Reproduce the Neel Nanda Grokking (modulo addition) Paper: https://openreview.net/pdf?id=9XFSbDPmdW
            Their Code: https://github.com/mechanistic-interpretability-grokking/progress-measures-paper/blob/main/Grokking_Analysis.ipynb
            NOTE: A left out a bunch of stuff that ended up being irrelevant (from their paper)

Date:       2025, Nov 5 started
Author:     ryan.rtjj@gmail.com
"""
import glob
import io
import os
from pathlib import Path
import pickle
import random

import einops
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
from PIL import Image

# +-------+
# | utils |
# +-------+
BACKGROUND_COLOR = '#FCFBF8'

# +-----------+
# | Model def |
# +-----------+
PRIME = 113
D_VOCAB = 113 + 1 # The last token is for the '=' sign (where we do read-out)
D_MODEL = 128
D_HEAD = 32
D_MLP = 512
ACT_TYPE = 'ReLU'
NUM_HEADS = 4
N_CTX = 3

class HookPoint(nn.Module):
    """
    Wraps any nn.Module's forward() call in an identity function for
    the sake of capturing the value of the any intermediate calculation
    """
    def __init__(self):
        super().__init__()
        self.fwd_hooks = []

    def give_name(self, name):
        self.name = name

    def add_hook(self, hook: callable):
        """
        @param hook:        Accepts parameters (activation, hook_name).
                            We have to change it into PyTorch hook format,
                            which is (module, input, output)
        """
        def torch_hook(module: nn.Module, input: torch.Tensor, output: torch.Tensor):
            """
            torch-compatible interface. Implements our actual hook under the hood.
            """
            # Doesn't matter if it's output or input; they're the same
            return hook(output, name=self.name)

        handle = self.register_forward_hook(torch_hook)
        self.fwd_hooks.append(handle)

    def remove_hooks(self):
        for handle in self.fwd_hooks:
            handle.remove()
        self.fwd_hooks = []
    
    def forward(self, x: torch.Tensor):
        return x

class Embed(nn.Module):
    """
    Simple dictionary lookup.
    """
    def __init__(self, d_vocab: int = D_VOCAB, d_model: int = D_MODEL):
        super().__init__()
        self.W_E = nn.Parameter(torch.randn(d_model, d_vocab) / np.sqrt(d_model))

    def forward(self, x: torch.Tensor):
        """
        Note that x is shape (batch, n_ctx), where each value is the token number
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

        # Hooks
        self.hook_attn = HookPoint()
        self.hook_z = HookPoint()       # Caches the mixed v vectors

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
        attn_matrix = self.hook_attn(
            F.softmax(attn_scores_masked / np.sqrt(self.d_head), dim=-1)
        )

        # rmb that v is shape (batch, num_heads, num_tokens, d_head)
        z = self.hook_z(torch.einsum('biph,biqp->biqh', v, attn_matrix))
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

    R^1 (idxs)           R^128                  R^128                    R^512                      R^128                   R^114
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

        # Give all hooks a name
        for name, module in self.named_modules():
            if isinstance(module, HookPoint):
                module.give_name(name)

    def hook_points(self):
        return [module for name, module in self.named_modules() if 'hook' in name]
    
    def remove_all_hooks(self):
        """
        Enumerates over all hooks in all components of the model and removes them
        """
        for hp in self.hook_points():
            hp.remove_hooks()

    def cache_all(self, cache: dict[str, torch.Tensor]):
        def save_hook(activation_tensor: torch.Tensor, name: str):
            cache[name] = activation_tensor.detach()

        for hook_point in self.hook_points():
            hook_point.add_hook(save_hook)

    def forward(self, x: torch.Tensor):
        x = self.embed(x)
        for block in self.blocks:
            x = block(x)
        x = self.unembed(x)
        return x

class MyDataset(Dataset):
    def __init__(self, pairs: list[tuple]):
        """
        @param pairs:     A list of triplets actually, lol. (a, b, P)
        """
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx: int):
        # Convert to tensors
        x = torch.tensor(self.pairs[idx], dtype=torch.long)
        y = (self.pairs[idx][0] + self.pairs[idx][1]) % self.pairs[idx][2]
        y = torch.tensor(y, dtype=torch.long)
        return x, y

def beautify_ax(ax: plt.Axes, xmin: float, xmax: float, ymin: float, ymax: float, ignore_aspect_ratio=False):
    ax.set_facecolor(BACKGROUND_COLOR)
    ax.set_xlim((xmin, xmax))
    ax.set_ylim((ymin, ymax))
    if not ignore_aspect_ratio:
        aspect_ratio = (xmax - xmin) / (ymax - ymin)
        ax.set_aspect(aspect_ratio)
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)

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
        _dir.mkdir(parents=True, exist_ok=True)

        if clear:
            files = glob.glob(f'{_dir}/*')
            for f in files:
                os.remove(f)

def train(
        model: nn.Module,
        data_file: Path,
        checkpoints_dir: Path,
        tensorboards_dir: Path,
        batch_size: int = -1,
        epochs: int = 2,
        lr: float = 0.001,
        weight_decay: float = 1.0,
        model_save_freq: int = 200,
    ):
    """
    Simple training loop
    """
    # Read the data
    with open(data_file, 'rb') as f:
        all_data = pickle.load(f)

    train_pairs = all_data['train_pairs']
    test_pairs = all_data['test_pairs']

    # Compute batch_size
    if batch_size == -1:
        batch_size = len(train_pairs)
    else:
        batch_size = min(len(train_pairs), batch_size)

    # Generate dataset / dataloader
    train_dataset = MyDataset(pairs=train_pairs)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
    test_dataset = MyDataset(pairs=test_pairs)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)
    total_batches = epochs * len(train_dataloader)

    # Create utils stuff
    pbar = tqdm(total=total_batches, desc='Training')
    writer = SummaryWriter(tensorboards_dir)

    # Define criterion
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    epoch_train_loss_history = []
    epoch_test_loss_history = []
    epoch_test_acc_history = []
    epoch_W_E_ss_history = []

    for epoch in range(epochs):
        model.train()
        epoch_train_loss = 0.0

        for batch_x, batch_y in train_dataloader:
            # forward pass
            logits = model(batch_x)[:,-1,:] # only take the last position
            loss = criterion(logits, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_item = loss.item()
            epoch_train_loss += loss_item

            # Update pbar
            pbar.set_postfix({
                'epoch': f'{epoch + 1} / {epochs}',
                'loss': f'{loss_item:.4f}'
            })
            pbar.update(1)

        model.eval()
        epoch_test_loss = 0.0
        epoch_num_correct = 1.0
        for batch_x, batch_y in test_dataloader:
            # forward pass
            logits = model(batch_x)[:,-1,:] # only take the last position
            loss = criterion(logits, batch_y)
            preds = torch.argmax(logits, dim=-1)
            num_correct = (preds == batch_y).sum().item()
            epoch_num_correct += num_correct

            loss_item = loss.item()
            epoch_test_loss += loss_item

        epoch_test_acc = epoch_num_correct / len(test_dataset)

        # Update histories
        epoch_train_loss_history += [epoch_train_loss]
        epoch_test_loss_history += [epoch_test_loss]
        epoch_test_acc_history += [epoch_test_acc]
        W_E = model.embed.W_E.detach().cpu().numpy()
        W_E_ss = np.sum(W_E ** 2)
        epoch_W_E_ss_history.append(W_E_ss)

        if (epoch + 1) % model_save_freq == 0:
            epoch_checkpoint_dict = {
                'epoch': epoch,
                'epoch_train_loss_history': epoch_train_loss_history,
                'epoch_test_loss_history': epoch_test_loss_history,
                'epoch_test_acc_history': epoch_test_acc_history,
                'epoch_W_E_ss_history': epoch_W_E_ss_history,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }

            torch.save(epoch_checkpoint_dict, checkpoints_dir / f'epoch_{epoch}.pt')

        # Plot
        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(13, 6))
        cmap = plt.get_cmap('coolwarm', 7)
        enumerated_epochs = np.linspace(1, epoch + 1, epoch + 1)
        xmin = 0
        xmax = epochs
        ymin = 0

        # Losses
        axes[0].plot(enumerated_epochs, epoch_train_loss_history, color=cmap(2), label='Train Loss')
        axes[0].plot(enumerated_epochs, epoch_test_loss_history, color=cmap(5), label='Test Loss')
        axes[0].set_title('Losses', fontsize=10, fontweight='normal')
        beautify_ax(
            axes[0], xmin, xmax, ymin, max(epoch_train_loss_history + epoch_test_loss_history)
        )
        axes[0].legend()

        # Acc
        axes[1].plot(enumerated_epochs, epoch_test_acc_history, color='black')
        axes[1].set_title('Test Accuracy', fontsize=10, fontweight='normal')
        beautify_ax(axes[1], xmin, xmax, ymin, 1.0)

        # Sum of squares of W_E
        axes[2].plot(enumerated_epochs, epoch_W_E_ss_history, color='grey')
        axes[2].set_title('W_E Sum of Squares', fontsize=10, fontweight='normal')
        axes[2].set_yscale('log')  # Add log scale to y-axis
        W_E_ss_ymax = max(epoch_W_E_ss_history)
        W_E_ss_ymin = 10.
        beautify_ax(axes[2], xmin, xmax, W_E_ss_ymin, W_E_ss_ymax, ignore_aspect_ratio=True)
        y_span = np.log10(W_E_ss_ymax) - np.log10(W_E_ss_ymin)
        aspect_ratio = (xmax - xmin) / y_span
        axes[2].set_aspect(aspect_ratio)

        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close(fig)
        buf.seek(0)
        img = Image.open(buf)
        img_np = np.array(img)
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).float() / 255.0

        writer.add_image('Training Metrics', img_tensor, global_step=epoch)



if __name__ == '__main__':
    RUN_NAME = 'debug'

    CHECKPOINTS_DIR = Path(f'checkpoints/{RUN_NAME}')
    DATA_DIR = Path(f'datasets/{RUN_NAME}')
    TENSORBOARDS_DIR = Path(f'tensorboards/{RUN_NAME}')
    prepare_dirs([CHECKPOINTS_DIR, DATA_DIR, TENSORBOARDS_DIR], True)

    data_file = DATA_DIR / 'dataset.pkl'
    generate_data(0.3, P=PRIME, save_filename=data_file)


    model = Transformer()
    train(
        model,
        data_file,
        CHECKPOINTS_DIR,
        TENSORBOARDS_DIR,
        epochs=20000,
    )