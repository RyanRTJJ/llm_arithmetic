"""
Desc:   Having trained the model to grok the modulo arithmetic problem, we want to:
        1.  Confirm the existence of periodic structure
"""
from pathlib import Path
import pickle
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from grokking_model import MyDataset
from grokking_model import Transformer

# +-------+
# | utils |
# +-------+
BACKGROUND_COLOR = '#FCFBF8'

def beautify_ax(ax: plt.Axes, xmin: float, xmax: float, ymin: float, ymax: float, ignore_aspect_ratio=False):
    ax.set_facecolor(BACKGROUND_COLOR)
    ax.set_xlim((xmin, xmax))
    ax.set_ylim((ymin, ymax))
    if not ignore_aspect_ratio:
        aspect_ratio = (xmax - xmin) / (ymax - ymin)
        ax.set_aspect(aspect_ratio)
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)

def load_model(checkpoint_file: Path) -> Transformer:
    """
    Given a `.pt` / `.pth` file, loads the model checkpoint and returns it
    """
    model = Transformer()
    chkpt_dict = torch.load(checkpoint_file)
    model_state_dict = chkpt_dict['model_state_dict']
    model.load_state_dict(model_state_dict)

    print(f'✅ model loaded from {checkpoint_file}')
    return model

def load_data(data_file: Path) -> tuple[list, list]:
    """
    @return:    [0] train_pairs
                [1] test_pairs
    """
    # Read the data
    with open(data_file, 'rb') as f:
        all_data = pickle.load(f)

    train_pairs = all_data['train_pairs']
    test_pairs = all_data['test_pairs']

    print(f'✅ data loaded from {data_file}')
    return train_pairs, test_pairs

def get_fourier_coeffs_by_hand(W: torch.Tensor, P: int) -> torch.Tensor:
    """
    This function assumes the periodic structure is over dimension 1

    @param P:       How many samples (113)

    @return:        Returns a (d_model, P) shaped tensor (fourier coeffs) in
                    the sequence: [DC, cos(k), sin(k), cos(2k), sin(2k), etc.]
                    where each of these elements represent the coefficients for
                    that fourier component.
    """
    fourier_basis = []

    # start with the DC freq
    fourier_basis.append(torch.ones(P) / np.sqrt(P))

    # All the pairs of sin cos
    for i in range(1, P // 2 + 1):
        fourier_basis.append(torch.cos(2 * torch.pi * torch.arange(P) * i/P))
        fourier_basis.append(torch.sin(2 * torch.pi * torch.arange(P) * i/P))
        fourier_basis[-2] /= fourier_basis[-2].norm()
        fourier_basis[-1] /= fourier_basis[-1].norm()

    fourier_basis = torch.vstack(fourier_basis)   # Shape (P, P), waves going along dim 1

    fourier_coeffs = W @ fourier_basis.T
    return fourier_coeffs

def inspect_periodic_nature(
        model: Transformer,
        weight_matrix: str = 'W_E',
        do_DFT_by_hand: bool = False
    ):
    """
    We want to see high-magnitude fourier components for:
    1.  W_E
    2.  W_L = W_U @ W_out
    """
    if weight_matrix == 'W_E':
        W = model.embed.W_E.detach().cpu()                      # shape (d_model, d_vocab)
        W = W[:,:-1]                                            # shape (d_model, P)
        # We expect periodicity over the vocab dimension.
    elif weight_matrix == 'W_L':
        W_U = model.unembed.W_U.detach().cpu()                  # shape (d_model, d_vocab)
        W_down = model.blocks[0].mlp.W_down.detach().cpu()      # shape (d_model, d_mlp)
        W = W_U.T @ W_down                                      # shape (d_vocab, d_mlp)
        W = W[:-1,:]                                            # shape (P, d_mlp)
        W = W.T                                                 # shape (d_mlp, P)
    else:
        raise ValueError(f'Unrecognized weight_matrix type: {weight_matrix}')
    
    _, P = W.shape

    _, ax = plt.subplots(1, 1, figsize=(12, 4))
    cmap = plt.get_cmap('coolwarm', 7)

    if do_DFT_by_hand:
        fourier_coeffs = get_fourier_coeffs_by_hand(W, P)
        fourier_coeff_norms = fourier_coeffs.norm(dim=0)

        x_axis = np.linspace(1, P, P - 1)
        x_ticks = [i for i in range(0, P, 10)]
        x_tick_labels = [i // 2 for i in range(0, P, 10)]

        colors = [cmap(2) if i % 2 == 0 else cmap(5) for i in range(P)]
        ax.bar(x_axis, fourier_coeff_norms[1:], width=0.6, color=colors)
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_tick_labels)
    else:
        # coeffs[:,0] is the DC freq.
        # If num timesteps is even, coeffs[:,-1] is the Nyquist frequency (ignore)
        fft_coeffs = np.fft.rfft(W, axis=-1)                    # shape (d_model, d_vocab // 2)
        fft_coeffs_normed = np.linalg.norm(fft_coeffs, axis=0)  # shape (d_vocab // 2,)

        x_axis = np.linspace(1, len(fft_coeffs_normed), len(fft_coeffs_normed))

        ax.bar(x_axis, fft_coeffs_normed, width=0.5, color=cmap(2))

    ax.set_ylabel('Relative Norm of Coefficients')
    ax.set_xlabel('Frequency multiple (k)')
    ax.set_facecolor(BACKGROUND_COLOR)
    plt.show()

def inspect_PCA_W_E(
        model: Transformer,
        k_vals: list[int],
        weight_matrix: str = 'W_E',
    ):
    """
    @param k_vals:  List of frequency multiples
    """
    for k in k_vals:
        assert k > 0, 'frequency_multiple k must be > 0'

    
    if weight_matrix == 'W_E':
        W = model.embed.W_E.detach().cpu()                      # shape (d_model, d_vocab)
        W = W[:,:-1]                                            # shape (d_model, P)
        # We expect periodicity over the vocab dimension.
    elif weight_matrix == 'W_L':
        W_U = model.unembed.W_U.detach().cpu()                  # shape (d_model, d_vocab)
        W_down = model.blocks[0].mlp.W_down.detach().cpu()      # shape (d_model, d_mlp)
        W = W_U.T @ W_down                                      # shape (d_vocab, d_mlp)
        W = W[:-1,:]                                            # shape (P, d_mlp)
        W = W.T                                                 # shape (d_mlp, P)
    else:
        raise ValueError(f'Unrecognized weight_matrix type: {weight_matrix}')

    _, P = W.shape

    _, axs = plt.subplots(1, len(k_vals), figsize=(7 + 5 * (len(k_vals) - 1), 7))

    # cmap = plt.get_cmap('coolwarm', P)
    # colors = [cmap(i) for i in range(P)]

    fourier_coeffs = get_fourier_coeffs_by_hand(W, P)

    z = 3.
    for idx, k in enumerate(k_vals):
        print(f'\n🔍 Inspecting PCA for W for k = {k}...')

        # We do milli_periods because cmaps can't give us decimal periods
        milli_period = int(1000 * P / k)
        cmap = plt.get_cmap('coolwarm', milli_period)
        colors = [cmap(i * 1000 % milli_period) for i in range(P)]

        if len(k_vals) == 1:
            ax = axs
        else:
            ax = axs[idx]

        basis_vecs = fourier_coeffs[:, [1 + 2 * (k - 1), 2 + 2 * (k - 1)]]
        basis_vecs_norm = basis_vecs.norm(p=2, dim=0, keepdim=True)
        print(f'basis_vecs_norms: {basis_vecs_norm}')
        basis_vecs /= basis_vecs_norm    # shape (d_model, 2)

        feats = W.T         # shape (P, d_model)
        feats_projected_b1_mag = feats @ basis_vecs[:,0]
        feats_projected_b2_mag = feats @ basis_vecs[:,1]

        b1_mag = feats_projected_b1_mag.numpy()
        b2_mag = feats_projected_b2_mag.numpy()

        # Scatter plot
        ax.scatter(b1_mag, b2_mag, c=colors, s=50, alpha=0.8)

        # Annotate each point with its index
        for p in range(P):
            ax.annotate(
                str(p),
                (b1_mag[p], b2_mag[p]),
                fontsize=8,
                alpha=0.7,
                xytext=(3, 3),
                textcoords='offset points'
            )

        beautify_ax(
            ax,
            xmin=-z,
            xmax=z,
            ymin=-z,
            ymax=z
        )
        ax.set_xlabel('PC 1')
        ax.set_ylabel('PC 2')
        ax.set_title(
            'W_E columns in basis given by fourier_coeffs' 
            f'[:, [{1 + 2 * (k - 1)}, {2 + 2 * (k - 1)}]],\n'
            f'i.e. Freq = {k}, Period = {113. / k:.3f}',
            fontsize=10
        )

    plt.show()

def inspect_attention_maps(
        model: Transformer,
        test_pairs: list[tuple[int, int, int]],
        num_samples: int = 2,
):
    """
    Plots the attention maps of all heads on some of the test_pairs.
    """
    assert num_samples <= 8, 'Give a reasonable number of samples plz.'
    sample_pairs = random.sample(test_pairs, num_samples)
    num_samples = len(sample_pairs) # Just in case num_samples > len(test_pairs)
    print(f'\n🔍 Inspecting attn maps for pairs: {sample_pairs}...')

    # Make the dataloader (full batch)
    sample_dataset = MyDataset(sample_pairs)
    sample_dataloader = DataLoader(sample_dataset, batch_size = len(sample_dataset))

    # Install the attn activation cache
    activation_cache = {}
    model.remove_all_hooks()
    model.cache_all(activation_cache)

    # Forward pass and collect the activations (attn only)
    model.eval()
    activations_by_head = {}
    for batch_x, batch_y in sample_dataloader:
        _ = model(batch_x)

        attn_activations = activation_cache['blocks.0.attn.hook_attn']
        # Expect this to have shape (b, num_heads, pos=3, pos=3)

        _, num_heads, _, _ = attn_activations.shape
        for i in range(num_heads):
            activations_by_head[f'head_{i}'] = attn_activations[:,i,:,:]

    # Plot
    num_heads = len(activations_by_head)
    fig, axs = plt.subplots(nrows=num_heads, ncols=num_samples, figsize=(2 * num_samples, 8))

    for head_i in range(num_heads):
        head_activations = activations_by_head[f'head_{head_i}']
        for sample_i, attn_map in enumerate(head_activations):
            ax = axs[head_i, sample_i]
            attn_map_np = attn_map.numpy()
            im = ax.imshow(attn_map_np, cmap='coolwarm', vmin=-1, vmax=1)
            ax.axis('off')

            if head_i == 0:
                # label row
                s_pair = sample_pairs[sample_i]
                ax.set_title(f'{s_pair[0]} + {s_pair[1]} =', fontsize=8)
            if sample_i == 0:
                ax.text(
                    -0.1,
                    0.5,
                    f'Head {head_i}',
                    transform=ax.transAxes,
                    fontsize=10,
                    ha='right',
                    va='center',
                    rotation=0
                )
    
    # plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    CHECKPOINT_FILE = 'checkpoints/grokked_20k/epoch_19999.pt'
    # CHECKPOINT_FILE = 'sparse_checkpoints/grokked_10k/epoch_9999.pt'
    DATA_FILE = 'datasets/grokked_20k/dataset.pkl'

    model = load_model(CHECKPOINT_FILE)
    train_pairs, test_pairs = load_data(DATA_FILE)
    # inspect_periodic_nature(model, weight_matrix='W_L', do_DFT_by_hand=True)
    # inspect_PCA_W_E(model, weight_matrix='W_L', k_vals=[4, 32, 43])
    inspect_attention_maps(model, test_pairs, num_samples=8)