"""
Desc:   Having trained the model to grok the modulo arithmetic problem, we want to:
        1.  Confirm the existence of periodic structure
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from grokking_model import Transformer

# +-------+
# | utils |
# +-------+
BACKGROUND_COLOR = '#FCFBF8'

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

def embed_to_cos_sin(fourier_embed: np.ndarray) -> np.ndarray:
    """
    After projecting W_E to canonical fourier basis (normalized), and
    normed across d_model, we want to stack the norms of coeffs into 2 rows:
    1 for cos, 1 for sin
    """
    # We start at index 1 because index 0 is the DC current
    return torch.stack([
        fourier_embed[1::2],
        fourier_embed[2::2]
    ])

def inspect_periodic_nature(model: Transformer, do_DFT_by_hand: bool = False):
    """
    We want to see high-magnitude fourier components for:
    1.  W_E
    2.  W_L = W_U @ W_out
    """
    W_E = model.embed.W_E.detach().cpu()                    # shape (d_model, d_vocab)
    W_E = W_E[:,:-1]                                        # shape (d_model, P)
    # We expect periodicity over the vocab dimension.
    d_model, P = W_E.shape

    fig, ax = plt.subplots(1, 1, figsize=(12, 4))
    cmap = plt.get_cmap('coolwarm', 7)

    if do_DFT_by_hand:
        fourier_basis = []

        # start with the DC freq
        fourier_basis.append(torch.ones(P) / np.sqrt(P))

        # All the pairs of sin cos
        for i in range(1, P // 2 + 1):
            fourier_basis.append(torch.cos(2 * torch.pi * torch.arange(P) * i/P))
            fourier_basis.append(torch.sin(2 * torch.pi * torch.arange(P) * i/P))
            fourier_basis[-2]/=fourier_basis[-2].norm()
            fourier_basis[-1]/=fourier_basis[-1].norm()

        fourier_basis = torch.stack(fourier_basis, dim=0)   # Shape (P, P), waves going along dim 1

        fourier_coeffs = W_E @ fourier_basis.T
        fourier_coeff_norms = fourier_coeffs.norm(dim=0)
        cos_sim_embed = embed_to_cos_sin(fourier_coeff_norms)       # shape (2, P // 2)

        # Melt for the sake of plotting
        flattened = cos_sim_embed.T.flatten().numpy()
        flattened_including_DC = np.zeros(P)
        flattened_including_DC[1:] = flattened

        x_axis = np.linspace(1, P, P)
        x_ticks = [i for i in range(0, P, 10)]
        x_tick_labels = [i // 2 for i in range(0, P, 10)]

        colors = [cmap(2) if i % 2 == 0 else cmap(5) for i in range(P)]
        ax.bar(x_axis, flattened_including_DC, width=0.6, color=colors)
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_tick_labels)
    else:
        # coeffs[:,0] is the DC freq.
        # If num timesteps is even, coeffs[:,-1] is the Nyquist frequency (ignore)
        fft_coeffs = np.fft.rfft(W_E, axis=-1)                  # shape (d_model, d_vocab // 2)
        fft_coeffs_normed = np.linalg.norm(fft_coeffs, axis=0)  # shape (d_vocab // 2,)

        x_axis = np.linspace(1, len(fft_coeffs_normed), len(fft_coeffs_normed))

        ax.bar(x_axis, fft_coeffs_normed, width=0.5, color=cmap(2))

    ax.set_ylabel('Relative Norm of Coefficients')
    ax.set_xlabel('Frequency multiple (k)')
    ax.set_facecolor(BACKGROUND_COLOR)
    plt.show()
    
if __name__ == '__main__':
    CHECKPOINT_FILE = 'checkpoints/grokked/epoch_4999.pt'
    model = load_model(CHECKPOINT_FILE)
    inspect_periodic_nature(model, do_DFT_by_hand=True)