"""
Desc:   Having trained the model to grok the modulo arithmetic problem, we want to:
        1.  Confirm the existence of periodic structure
"""
from pathlib import Path
import pickle
import random

import einops
import matplotlib.pyplot as plt
import matplotlib.collections as mc
from matplotlib.colors import Normalize
import numpy as np
from scipy.interpolate import make_interp_spline
from scipy.linalg import subspace_angles
import torch
from torch.utils.data import DataLoader

from grokking_model import MyDataset
from grokking_model import Transformer

# +-------+
# | utils |
# +-------+
BACKGROUND_COLOR = '#FCFBF8'
O_PROJECTED_SAVE_LOC = 'o_values_projected_grokked_20k_k_{k}.npz'

def draw_line(ax, v1, v2, line_kwargs):
    """
    @param v1, v2:      np.ndarrays of shape (dim,)
    """
    assert v1.shape == v2.shape, f'v1 and v2 need to have same shape, got {v1.shape} vs. {v2.shape}'
    if v1.shape[0] == 2:
        vector_line = np.stack((v1, v2), axis=0)[None,:,:]
        ax.add_collection(
            mc.LineCollection(
                vector_line,
                **line_kwargs
            )
        )
    else:
        assert v1.shape[0] == 3, 'draw_line takes v1 and v2 that are either 2 or 3 dimensional'
        x_values = [v1[0], v2[0]]
        y_values = [v1[1], v2[1]]
        z_values = [v1[2], v2[2]]
        ax.plot(x_values, y_values, z_values, **line_kwargs)

def draw_vector(ax, v, scatter_kwargs, line_kwargs, origin: np.ndarray | None = None):
    """
    @param v:       np.ndarray of shape (dim,)
    """
    if origin is None:
        origin = np.zeros_like(v)

    # draw the point
    if v.shape[0] == 2:
        ax.scatter(v[0], v[1], **scatter_kwargs)
    else:
        assert v.shape[0] == 3, f'v must be 2 or 3-dimensional. got {v.shape[0]}'
        ax.scatter(v[0], v[1], v[2], **scatter_kwargs)

    # the stem of the vector
    draw_line(ax, origin, v, line_kwargs)

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
    chkpt_dict = torch.load(checkpoint_file, weights_only=False)
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
        show_only_last_attn_row: bool = True,
        full_p_by_p_plot: bool = False,
):
    """
    Plots the attention maps of all heads on some of the test_pairs.
    """
    # assert num_samples <= 8, 'Give a reasonable number of samples plz.'
    # sample_pairs = random.sample(test_pairs, num_samples)
    P = 113
    if full_p_by_p_plot:
        sample_pairs = []
        for x in range(P):
            for y in range(P):
                sample_pairs += [(x, y, P)]
    else:
        x = 42
        sample_pairs = [(x, y, P) for y in range(60)]

    num_samples = len(sample_pairs) # Just in case num_samples > len(test_pairs)

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

    if full_p_by_p_plot:
        fig, axs = plt.subplots(nrows=1, ncols=4, figsize=(16, 6))
    else:
        fig, axs = plt.subplots(nrows=num_samples, ncols=num_heads, figsize=(4, min(12, 2 * num_samples)))

    for head_i in range(num_heads):
        head_activations = activations_by_head[f'head_{head_i}']    # shape (b, pos, pos)
        if full_p_by_p_plot:
            # Record only attention on token `a` in last row
            attention_on_token_a = head_activations[:,-1,0]

            # Reshape
            attention_on_token_a = attention_on_token_a.reshape(P, P)

            ax = axs[head_i]
            ax.imshow(attention_on_token_a, cmap='coolwarm', vmin=-1, vmax=1)
            
            if head_i == 0:
                # label row
                ax.text(
                    -0.2,
                    0.5,
                    'token `a`',
                    transform=ax.transAxes,
                    fontsize=10,
                    ha='right',
                    va='center',
                    rotation=0
                )
            ax.set_xlabel('token `b`', fontsize=10)
            ax.set_title(f'Head {head_i}', fontsize=8)

        else:
            for sample_i, attn_map in enumerate(head_activations):
                ax = axs[sample_i, head_i]
                attn_map_np = attn_map.numpy()

                if show_only_last_attn_row:
                    attn_map_np = attn_map_np[-1,:][None,:]

                im = ax.imshow(attn_map_np, cmap='coolwarm', vmin=-1, vmax=1)
                ax.axis('off')

                if head_i == 0:
                    # label row
                    s_pair = sample_pairs[sample_i]
                    ax.text(
                        -0.2,
                        0.5,
                        f'{s_pair[0]} + {s_pair[1]} =',
                        transform=ax.transAxes,
                        fontsize=8,
                        ha='right',
                        va='center',
                        rotation=0
                    )
                if sample_i == 0:
                    ax.set_title(f'Head {head_i}', fontsize=8)
    if full_p_by_p_plot:
        plt.suptitle(f'Attn score from `=` to `a`', fontsize=11)
    # plt.tight_layout()
    plt.show()

def inspect_attention_maps_periodic_nature(
        model: Transformer,
):
    """
    Plot Fourier Coefficient Norms of last rows of attention maps
    """
    # Create sample pairs
    P = 113
    x = 70
    y = [i for i in range(0, P)]
    sample_pairs = [(x, _y, P) for _y in y]

    num_samples = len(sample_pairs) # Just in case num_samples > len(test_pairs)
    print(f'\n🔍 Inspecting attn maps for periodic nature for pairs: {sample_pairs}...')

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
    fig, axs = plt.subplots(nrows=num_heads, ncols=1, figsize=(12, 8))
    cmap = plt.get_cmap('coolwarm', 7)

    for head_i in range(num_heads):
        head_activations = activations_by_head[f'head_{head_i}']

        head_activations_last_rows = head_activations[:,-1,:]    # shape (B, 3)
        fourier_coeffs = get_fourier_coeffs_by_hand(head_activations_last_rows.T, P)
        fourier_coeff_norms = fourier_coeffs.norm(dim=0)

        x_axis = np.linspace(1, P, P - 1)
        x_ticks = [i for i in range(0, P, 10)]
        x_tick_labels = [i // 2 for i in range(0, P, 10)]

        colors = [cmap(2) if i % 2 == 0 else cmap(5) for i in range(P)]
        ax = axs[head_i]
        ax.bar(x_axis, fourier_coeff_norms[1:], width=0.6, color=colors)
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_tick_labels)

        ax.set_ylabel(f'Head {head_i}')
        ax.set_xlabel('Frequency (k)')
        ax.set_facecolor(BACKGROUND_COLOR)

    plt.suptitle(
        'Relative Norm of Attention Map (Last Row) Fourier Coefficients (by head)',
        fontsize=10
    )
    plt.show()

def inspect_attention_z_values(
        model: Transformer,
        head_i: int,
        k: int,
        plot_interpolation: bool = False,
        line_gradient: bool = False,
        annotate_scatter: bool = False,
):
    """
    Try to find the petal shaped outputs in Head 1
    """
    # Create sample pairs
    P = 113
    x = 70
    y = list(range(P))
    sample_pairs = [(x, _y, P) for _y in y]

    num_samples = len(sample_pairs) # Just in case num_samples > len(test_pairs)

    # Make the dataloader (full batch)
    sample_dataset = MyDataset(sample_pairs)
    sample_dataloader = DataLoader(sample_dataset, batch_size = len(sample_dataset))

    # Install the attn activation cache
    cache = {}
    model.remove_all_hooks()
    model.cache_all(cache)

    # Forward pass and collect the activations (attn only)
    model.eval()
    z_values_by_head = {}
    for batch_x, batch_y in sample_dataloader:
        _ = model(batch_x)

        z_values = cache['blocks.0.attn.hook_z']
        # Expect this to have shape (b, num_heads, num_tokens=3, d_head)

        _, num_heads, _, _ = z_values.shape
        for i in range(num_heads):
            z_values_by_head[f'head_{i}'] = z_values[:,i,:,:]

    # I still need the fourier coeff basis from W_E for k=4:
    W = model.embed.W_E.detach().cpu()                      # shape (d_model, d_vocab)
    W = W[:,:-1]                                            # shape (d_model, P)
    fourier_coeffs = get_fourier_coeffs_by_hand(W, P)

    # Because we want to extract those 2 dimensions of the embeddings, then
    # apply W_V to it
    ATTN_HEAD_TO_FREQ_TEXT = {
        0: '43 Hz (mostly), with some 4 Hz and small amounts of 27 Hz, 32 Hz, 47 Hz',
        1: '4 Hz',
        2: '43 Hz (mostly), with some 4 Hz',
        3: '32 Hz (mostly), with some 4 Hz'
    }
    basis_vecs = fourier_coeffs[:, [1 + 2 * (k - 1), 2 + 2 * (k - 1)]]      # shape (d_model, 2)
    basis_vecs_norm = basis_vecs.norm(p=2, dim=0, keepdim=True)
    basis_vecs /= basis_vecs_norm                                           # shape (d_model, 2)

    # We previously got magnitude w.r.t basis_vecs by doing feats @ basis_vecs,
    # which is (P, d_model) @ (d_model, 2) = (P, 2),
    # So now, because v = feats @ W_V,
    # which is (P, d_head) = (P, d_model) @ (d_model, d_head),
    # we have feats @ basis_vecs = feats @ W_V @ new_basis
    # => basis_vecs = W_V @ new_basis
    # => new_basis = np.linalg.pinv(W_V) @ basis_vecs, where pinv is just inv(W_V.T @ W_V) @ W_V.T
    W_V = model.blocks[0].attn.W_V.detach().cpu()           # shape (num_heads, d_head, d_model)
    W_V_this_head = W_V[head_i,:,:]                         # shape (d_head, d_model)
    W_V_this_head = W_V_this_head.T                         # shape (d_model, d_head)
    W_V_this_head_inv = torch.inverse(W_V_this_head.T @ W_V_this_head)
    W_V_pinv = W_V_this_head_inv @ W_V_this_head.T          # shape (d_head, d_model)
    new_basis = W_V_pinv @ basis_vecs                       # shape (d_head, 2)

    # Plot
    if plot_interpolation:
        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(14, 7))
    else:
        fix, ax = plt.subplots(1, 1, figsize=(6, 6))
    cmap = plt.get_cmap('coolwarm', 7)

    z_values_this_head = z_values_by_head[f'head_{head_i}'] # shape (b, num_tokens, d_head)
    # Only want to focus on the `=` token
    z_values_this_head = z_values_this_head[:,-1,:]         # shape (b, d_head)
    
    z_values_projected = (z_values_this_head @ new_basis).numpy()
    b1_mag = z_values_projected[:,0]
    b2_mag = z_values_projected[:,1]

    # Scatter plot
    # colors = [plt.get_cmap('coolwarm', 113)(i) for i in range(P)]
    if plot_interpolation:
        axes_with_scatters = [axes[0], axes[1]]
        axes_with_interpolation = [axes[1], axes[2]]
        for ax in axes_with_interpolation:
            # smooth interpolation
            # Create a parametric variable
            t = np.arange(len(b1_mag))

            # Create spline interpolations
            spl_b1 = make_interp_spline(t, b1_mag, k=3)  # k=3 for cubic spline
            spl_b2 = make_interp_spline(t, b2_mag, k=3)

            # Generate more points for smoother curve
            num_points = 1000
            t_smooth = np.linspace(t.min(), t.max(), num_points)
            b1_smooth = spl_b1(t_smooth)
            b2_smooth = spl_b2(t_smooth)

            if line_gradient:
                # make a color gradient line
                points = np.array([b1_smooth, b2_smooth]).T.reshape(-1, 1, 2)
                segments = np.concatenate([points[:-1], points[1:]], axis=1)
                # Create color array based on position along the curve
                num_points = len(b1_smooth)
                cmap = plt.get_cmap('coolwarm', num_points)
                colors = cmap(np.linspace(0, 1, len(segments)))

                # Create LineCollection
                lc = mc.LineCollection(segments, colors=colors, linewidth=1.0, alpha=0.6)
                ax.add_collection(lc)
            else:
                ax.plot(b1_smooth, b2_smooth, '-', linewidth=1.0, alpha=0.2)

        # Do scatter
        for ax, s in zip(axes_with_scatters, [30, 30]):
            ax.scatter(b1_mag, b2_mag, color=cmap(2), s=s, alpha=0.9)

        if annotate_scatter:
            # Annotate each point with its index
            for p in range(P):
                axes_with_scatters[0].annotate(
                    str(p),
                    (b1_mag[p], b2_mag[p]),
                    fontsize=6,
                    alpha=0.5,
                    xytext=(3, 3),
                    textcoords='offset points'
                )

        # Beautiful
        for i, ax in enumerate(axes):
            Z = 0.5
            beautify_ax(
                ax,
                xmin=-Z,
                xmax=Z,
                ymin=-Z,
                ymax=Z
            )
            if i == 0:
                ax_title = 'z values'
            elif i == 1:
                ax_title = 'z values with smoothing splines interpolation'
            else:
                ax_title = 'interpolation pattern only'
            ax.set_title(ax_title, fontsize=10)
    else:
        ax.scatter(b1_mag, b2_mag, color=cmap(2), s=50, alpha=0.9)
        Z = 0.5
        beautify_ax(
            ax,
            xmin=-Z,
            xmax=Z,
            ymin=-Z,
            ymax=Z
        )
        title_text = (
            f'$\\bf{{z\ values\ for:}}$\n'
            f"  - attn head {head_i}, with periodicity: {ATTN_HEAD_TO_FREQ_TEXT[head_i]}\n"
            f"  - in 2D subspace corresponding to the freq {k} Hz circle"
        )
        ax.set_title(title_text, loc='left', fontsize=10)


    plt.show()

def compute_pinv_by_hand(W: torch.Tensor):
    """
    Helper function
    """
    height, width = W.shape
    if height > width:
        # tall
        inv = torch.inverse(W.T @ W)
        pinv = inv @ W.T
    else:
        # wide
        inv = torch.inverse(W @ W.T)
        pinv = W.T @ inv
    return pinv

def get_interpolation(
        points: np.ndarray,
        num_points: int = 1000,
        degree: int = 3,
        return_line_collection: bool = False
    ):
    """
    Helper function

    @param points:      (n, 2) array of points to interpolate
    @param num_points:  smoothness. The higher the better but more expensive.
    @param degree:      Of polynomial of spline. Default 3 for cubic splines.
    """
    n, d = points.shape
    assert d == 2, 'get_interpolation not implemented for non-2d points'

    # Create a parametric variable
    t = np.arange(n)

    # Create spline interpolations
    spl_b1 = make_interp_spline(t, points[:,0], k=degree)  # k = 3 for cubic spline
    spl_b2 = make_interp_spline(t, points[:,1], k=degree)

    # Generate more points for smoother curve
    num_points = 1000
    t_smooth = np.linspace(t.min(), t.max(), num_points)
    b1_smooth = spl_b1(t_smooth)
    b2_smooth = spl_b2(t_smooth)

    if return_line_collection:
        # make a color gradient line
        vertices = points.T.reshape(-1, 1, 2)
        segments = np.concatenate([vertices[:-1], vertices[1:]], axis=1)
        # Create color array based on position along the curve
        cmap = plt.get_cmap('coolwarm', num_points)
        colors = cmap(np.linspace(0, 1, len(segments)))

        # Create LineCollection
        lc = mc.LineCollection(segments, colors=colors, linewidth=1.0, alpha=0.6)
        return lc
    else:
        result = np.hstack([b1_smooth[:,None], b2_smooth[:,None]])
        return result


def inspect_attention_outputs(
        model: Transformer,
        head_i: int,
        k: int,
        plot_interpolation: bool = False,
        line_gradient: bool = False,
        annotate_scatter: bool = False,
):
    """
    Try to find the petal shaped outputs we found in z-space in o-space as well
    """
    # Create sample pairs
    P = 113
    x = 70
    y = list(range(P))
    sample_pairs = [(x, _y, P) for _y in y]

    num_samples = len(sample_pairs) # Just in case num_samples > len(test_pairs)

    # Make the dataloader (full batch)
    sample_dataset = MyDataset(sample_pairs)
    sample_dataloader = DataLoader(sample_dataset, batch_size = len(sample_dataset))

    # Install the attn activation cache
    cache = {}
    model.remove_all_hooks()
    model.cache_all(cache)

    # Forward pass and collect the activations
    model.eval()
    z_values_by_head = {}
    for batch_x, batch_y in sample_dataloader:
        _ = model(batch_x)

        z_values = cache['blocks.0.attn.hook_z']
        # Expect this to have shape (b, num_heads, num_tokens=3, d_head)

        _, num_heads, _, _ = z_values.shape
        for i in range(num_heads):
            z_values_by_head[f'head_{i}'] = z_values[:,i,:,:]

    # I still need the fourier coeff basis from W_E for k=4:
    W = model.embed.W_E.detach().cpu()                      # shape (d_model, d_vocab)
    W = W[:,:-1]                                            # shape (d_model, P)
    fourier_coeffs = get_fourier_coeffs_by_hand(W, P)

    # Because we want to extract those 2 dimensions of the embeddings, then
    # apply W_V to it
    basis_vecs = fourier_coeffs[:, [1 + 2 * (k - 1), 2 + 2 * (k - 1)]]      # shape (d_model, 2)
    basis_vecs_norm = basis_vecs.norm(p=2, dim=0, keepdim=True)
    basis_vecs /= basis_vecs_norm                                           # shape (d_model, 2)

    # Calculate the basis transformation that would allow us to visualize in 2D
    W_V = model.blocks[0].attn.W_V.detach().cpu()           # shape (num_heads, d_head, d_model)
    W_V_this_head = W_V[head_i,:,:]                         # shape (d_head, d_model)
    W_V_this_head = W_V_this_head.T                         # shape (d_model, d_head)
    W_V_pinv = compute_pinv_by_hand(W_V_this_head)          # shape (d_head, d_model)

    num_heads, d_head, d_model = W_V.shape
    W_O = model.blocks[0].attn.W_O.detach().cpu()                   # shape (d_model, num_heads * d_head)
    W_O_this_head = W_O[:, head_i * d_head: (head_i + 1) * d_head]  # shape (d_model, d_head)
    W_O_this_head = W_O_this_head.T                                 # shape (d_head, d_model)
    W_O_pinv = compute_pinv_by_hand(W_O_this_head)                  # shape (d_model, d_head)

    WV_WO_pinv = W_O_pinv @ W_V_pinv                                # shape (d_model, d_model)
    new_basis = WV_WO_pinv @ basis_vecs                             # shape (d_model, 2)

    z_values_this_head = z_values_by_head[f'head_{head_i}']         # shape (b, num_tokens, d_head)
    o_values_this_head = z_values_this_head @ W_O_this_head         # shape (b, num_tokens, d_model)
    # Only want to focus on the `=` token
    o_values_this_head = o_values_this_head[:,-1,:]                 # shape (b, d_model)
    print(f'o_values_this_head.shape: {o_values_this_head.shape}')
    
    o_values_projected = (o_values_this_head @ new_basis).numpy()
    b1_mag = o_values_projected[:,0]
    b2_mag = o_values_projected[:,1]

    # Plot
    if plot_interpolation:
        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(14, 7))
    else:
        fix, ax = plt.subplots(1, 1, figsize=(6, 6))
    cmap = plt.get_cmap('coolwarm', 7)

    # Scatter plot
    Z = 0.5
    if plot_interpolation:
        axes_with_scatters = [axes[0], axes[1]]
        axes_with_interpolation = [axes[1], axes[2]]
        for ax in axes_with_interpolation:
            if line_gradient:
                lc = get_interpolation(
                    o_values_projected,
                    return_line_collection=True
                )
                ax.add_collection(lc)
            else:
                projected_smooth = get_interpolation(o_values_projected)
                ax.plot(projected_smooth[:,0], projected_smooth[:,1], '-', linewidth=1.0, alpha=0.2)

        # Do scatter
        for ax, s in zip(axes_with_scatters, [30, 30]):
            ax.scatter(b1_mag, b2_mag, color=cmap(2), s=s, alpha=0.9)
        
        if annotate_scatter:
            # Annotate each point with its index
            for p in range(P):
                axes_with_scatters[0].annotate(
                    str(p),
                    (b1_mag[p], b2_mag[p]),
                    fontsize=8,
                    alpha=0.6,
                    xytext=(3, 3),
                    textcoords='offset points'
                )

        # Beautiful
        for i, ax in enumerate(axes):
            beautify_ax(
                ax,
                xmin=-Z,
                xmax=Z,
                ymin=-Z,
                ymax=Z
            )
            if i == 0:
                ax_title = 'z values'
            elif i == 1:
                ax_title = 'z values with smoothing splines interpolation'
            else:
                ax_title = 'interpolation pattern only'
            ax.set_title(ax_title, fontsize=10)
    else:
        ax.scatter(b1_mag, b2_mag, color=cmap(2), s=50, alpha=0.9)
        # print(len(b1_mag))
        beautify_ax(
            ax,
            xmin=-Z,
            xmax=Z,
            ymin=-Z,
            ymax=Z
        )

    plt.show()

def inspect_attention_outputs_in_agg(
        model: Transformer,
        k: int,
        plot_interpolation: bool = False,
        annotate_scatter: bool = False,
        o_projected_save_loc: str = '',
):
    """
    Plots each head's contribution to the 2d space where the original
    frequency-`k` embedding circles were found
    """
    # Create sample pairs
    P = 113
    x = 70
    y = list(range(P))
    sample_pairs = [(x, _y, P) for _y in y]

    # Make the dataloader (full batch)
    sample_dataset = MyDataset(sample_pairs)
    sample_dataloader = DataLoader(sample_dataset, batch_size = len(sample_dataset))

    # Install the attn activation cache
    cache = {}
    model.remove_all_hooks()
    model.cache_all(cache)

    # Forward pass and collect the activations
    model.eval()
    z_values_by_head = {}
    o_values_cached = None
    o_values_by_head = {}
    num_heads = None

    for batch_x, batch_y in sample_dataloader:
        _ = model(batch_x)

        z_values = cache['blocks.0.attn.hook_z']        # (b, num_heads, num_tokens, d_head)
        o_values_cached = cache['blocks.0.attn.hook_o']        # (b, num_tokens, d_model)

        _, num_heads, _, _ = z_values.shape
        for i in range(num_heads):
            z_values_by_head[f'head_{i}'] = z_values[:,i,:,:]

    # I still need the fourier coeff basis from W_E for k=4:
    W = model.embed.W_E.detach().cpu()                      # shape (d_model, d_vocab)
    W = W[:,:-1]                                            # shape (d_model, P)
    fourier_coeffs = get_fourier_coeffs_by_hand(W, P)

    # Because we want to extract those 2 dimensions of the embeddings, then
    # apply W_V to it
    basis_vecs = fourier_coeffs[:, [1 + 2 * (k - 1), 2 + 2 * (k - 1)]]      # shape (d_model, 2)
    basis_vecs_norm = basis_vecs.norm(p=2, dim=0, keepdim=True)
    basis_vecs /= basis_vecs_norm                                           # shape (d_model, 2)

    fig, axes = plt.subplots(nrows=2, ncols=num_heads + 1, figsize=(14, 6))
    SCATTER_SIZE = 20
    SCATTER_ALPHA = 0.7
    cmap = plt.get_cmap('coolwarm', 7)
    Z = 0.55
    # Calculate the basis transformation that would allow us to visualize in 2D
    o_projected_by_head = {}
    for head_i in range(num_heads + 1):
        scatter_ax = axes[0,head_i]
        interpolation_ax = axes[1,head_i]

        if head_i < num_heads:
            W_V = model.blocks[0].attn.W_V.detach().cpu()           # shape (num_heads, d_head, d_model)
            W_V_this_head = W_V[head_i,:,:]                         # shape (d_head, d_model)
            W_V_this_head = W_V_this_head.T                         # shape (d_model, d_head)
            W_V_pinv = compute_pinv_by_hand(W_V_this_head)          # shape (d_head, d_model)

            num_heads, d_head, d_model = W_V.shape
            W_O = model.blocks[0].attn.W_O.detach().cpu()                   # shape (d_model, num_heads * d_head)
            W_O_this_head = W_O[:, head_i * d_head: (head_i + 1) * d_head]  # shape (d_model, d_head)
            W_O_this_head = W_O_this_head.T                                 # shape (d_head, d_model)
            W_O_pinv = compute_pinv_by_hand(W_O_this_head)                  # shape (d_model, d_head)

            WV_WO_pinv = W_O_pinv @ W_V_pinv                                # shape (d_model, d_model)
            new_basis = WV_WO_pinv @ basis_vecs                             # shape (d_model, 2)

            z_values_this_head = z_values_by_head[f'head_{head_i}']         # shape (b, num_tokens, d_head)
            o_values_this_head = z_values_this_head @ W_O_this_head         # shape (b, num_tokens, d_model)
            # Only want to focus on the `=` token
            o_values_this_head = o_values_this_head[:,-1,:]                 # shape (b, d_model)
            print(f'o_values_this_head.shape: {o_values_this_head.shape}')
        
            o_values_projected = (o_values_this_head @ new_basis).numpy()
            o_projected_by_head[head_i] = o_values_projected
            b1_mag = o_values_projected[:,0]
            b2_mag = o_values_projected[:,1]
            color = cmap(2)
            ax_title = f'attn head {head_i} only'
        else:
            # Overall
            use_correct_method = True
            if not use_correct_method:
                # This logic doesn't have to be here. I just kept it because I have yet
                # to understand why this is different from 'use_correct_method' logic.
                # Why doesn't this work?
                W_V = model.blocks[0].attn.W_V.detach().cpu()           # shape (num_heads, d_head, d_model)
                W_V = einops.rearrange(W_V, 'i h d -> (i h) d')         # shape (num_heads * d_head, d_model)
                W_V = W_V.T                                             # shape (d_model, num_heads * d_head)

                W_O = model.blocks[0].attn.W_O.detach().cpu()           # shape (d_model, num_heads * d_head)
                W_O = W_O.T                                             # shape (num_heads * d_head, d_model)

                # Alt calculation
                WV_WO = W_V @ W_O                                       # shape (d_model, d_model)
                WV_WO_pinv = torch.inverse(WV_WO)
                # WV_WO_pinv = W_O_pinv @ W_V_pinv                        # shape (d_model, d_model)
                new_basis = WV_WO_pinv @ basis_vecs

                # Only want to focus on the `=` token
                o_values = o_values_cached[:,-1,:]                      # shape (P, d_model)
                o_values_projected = (o_values @ new_basis).numpy()     # shape (P, 2)
                print(o_values_projected)
            
                Z = 10.0
            else:
                # This should work
                o_values_projected = np.vstack([projected[None,:,:] for projected in o_projected_by_head.values()])
                o_values_projected = np.sum(o_values_projected, axis=0)

                # Save this
                if o_projected_save_loc:
                    np.savez(
                        o_projected_save_loc,
                        o_projected=o_values_projected,
                        o=o_values_cached[:,-1,:]
                    )
                    print(f'✅ o_values_projected saved in {O_PROJECTED_SAVE_LOC}')

            b1_mag = o_values_projected[:,0]
            b2_mag = o_values_projected[:,1]
            Z = 2.0
            color = cmap(5)
            ax_title = 'overall / summed'

        # Do scatter
        scatter_ax.scatter(b1_mag, b2_mag, color=color, s=SCATTER_SIZE, alpha=SCATTER_ALPHA)
        scatter_ax.set_title(ax_title, fontsize=10)

        # Do interpolation
        projected_smooth = get_interpolation(o_values_projected, degree=3)
        interpolation_ax.plot(
            projected_smooth[:,0], projected_smooth[:,1], '-', color=color, linewidth=1.0, alpha=0.4
        )

        for ax in [scatter_ax, interpolation_ax]:
            beautify_ax(
                ax,
                xmin=-Z,
                xmax=Z,
                ymin=-Z,
                ymax=Z
            )

    plt.suptitle(
        f'$\\bf{{o\ vectors}}$\nin 2D subspace corresponding to freq {k} Hz embedding circle',
        fontsize=11
    )
    plt.show()

def inspect_attention_outputs_periodic_nature(
        model: Transformer,
        o_dict: dict[str, np.ndarray],
        do_projected: bool = True,
):
    """
    Do fourier analysis of o_projected to see if it's really a nice frequency k circle

    @param do_projected:    Whether to fourier analyze the projected o or overall o embeddings
    """
    o_projected = o_dict['o_projected']
    o = o_dict['o']

    if do_projected:
        o_values = o_projected
    else:
        o_values = o

    P, d_model = o_values.shape
    o_values = torch.Tensor(o_values.T)

    fourier_coeffs = get_fourier_coeffs_by_hand(o_values, P)
    fourier_coeff_norms = fourier_coeffs.norm(dim=0)

    _, ax = plt.subplots(1, 1, figsize=(12, 4))
    cmap = plt.get_cmap('coolwarm', 7)

    x_axis = np.linspace(1, P, P - 1)
    x_ticks = [i for i in range(0, P, 10)]
    x_tick_labels = [i // 2 for i in range(0, P, 10)]

    colors = [cmap(2) if i % 2 == 0 else cmap(5) for i in range(P)]
    ax.bar(x_axis, fourier_coeff_norms[1:], width=0.6, color=colors)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_tick_labels)

    ax.set_ylabel('Relative Norm of Coefficients')
    ax.set_xlabel('Frequency multiple (k)')
    ax.set_facecolor(BACKGROUND_COLOR)
    plt.show()

def get_embeddings_circle_bases(
        embeddings: np.ndarray,
        k_values: list[int] = [4, 32, 43],
):

    P, d = embeddings.shape
    o_values = torch.Tensor(embeddings.T)

    fourier_coeffs = get_fourier_coeffs_by_hand(o_values, P)

    k_to_basis_vecs = {}

    for k in k_values:
        basis_vecs = fourier_coeffs[:, [1 + 2 * (k - 1), 2 + 2 * (k - 1)]]
        basis_vecs_norm = basis_vecs.norm(p=2, dim=0, keepdim=True)
        basis_vecs /= basis_vecs_norm    # shape (d_model, 2)

        k_to_basis_vecs[k] = basis_vecs.numpy()

    return k_to_basis_vecs

def get_WE_circle_bases(
        W_E: np.ndarray | torch.Tensor,
        k_values: list[int] = [4, 32, 43],
):
    b, P = W_E.shape
    fourier_coeffs = get_fourier_coeffs_by_hand(W_E, P)

    k_to_embedding_basis_vecs = {}
    for idx, k in enumerate(k_values):
        basis_vecs = fourier_coeffs[:, [1 + 2 * (k - 1), 2 + 2 * (k - 1)]]
        basis_vecs_norm = basis_vecs.norm(p=2, dim=0, keepdim=True)
        basis_vecs /= basis_vecs_norm    # shape (d_model, 2)
        k_to_embedding_basis_vecs[k] = basis_vecs
    return k_to_embedding_basis_vecs

def visualize_o_circles(
        o_dict: dict[str, np.ndarray],
):
    """
    More of just a sanity check
    """
    K_VALUES = [4, 32, 43]
    k_to_basis_vecs = get_embeddings_circle_bases(
        embeddings=o_dict['o'],
        k_values=K_VALUES,
    )

    o_values = o_dict['o']      # shape (P, d_model)
    P, d_model = o_values.shape

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(12, 6))
    # cmap = plt.get_cmap('coolwarm', 7)
    Z = 7.0
    for i, k in enumerate(K_VALUES):
        ax = axes[i]

        basis_vecs = k_to_basis_vecs[k]
        o_values_projected = o_values @ basis_vecs

        # Scatter plot
        # We do milli_periods because cmaps can't give us decimal periods
        milli_period = int(1000 * P / k)
        cmap = plt.get_cmap('coolwarm', milli_period)
        colors = [cmap(i * 1000 % milli_period) for i in range(P)]
        ax.scatter(o_values_projected[:,0], o_values_projected[:,1], color=colors, s=50, alpha=0.7)

        beautify_ax(
            ax,
            xmin=-Z,
            xmax=Z,
            ymin=-Z,
            ymax=Z
        )
        ax.set_xlabel('PC 1')
        ax.set_ylabel('PC 2')
        ax.set_title(
            f'Fourier Coeffs for {k} Hz ',
            fontsize=10
        )
    plt.suptitle(f'$\\bf{{o\ vectors\ in\ 2D\ Subspace\ Corresponding\ To:}}$', fontsize=11)
    plt.show()

def do_WE_and_o_coexist(
        model: Transformer,
        o_dict: dict[str, np.ndarray],
        k_values: list[int] = [4, 32, 43],
):
    """
    Finds the basis vectors for the o vectors as well as the W_E vectors
    """
    k_to_o_basis_vecs = get_embeddings_circle_bases(
        embeddings=o_dict['o'],
        k_values=k_values,
    )

    o_values = o_dict['o']      # shape (P, d_model)
    P, d_model = o_values.shape

    W_E = model.embed.W_E.detach().cpu()                    # shape (d_model, d_vocab)
    W_E = W_E[:,:-1]                                        # shape (d_model, P)

    k_to_embedding_basis_vecs = get_WE_circle_bases(
        W_E=W_E,
        k_values=k_values,
    )

    num_k_values = len(k_values)
    subspace_angles_grid = np.zeros((num_k_values, num_k_values))

    for i in range(num_k_values):
        ki = k_values[i]
        for j in range(num_k_values):
            kj = k_values[j]
            o_basis_vecs = k_to_o_basis_vecs[ki]
            e_basis_vecs = k_to_embedding_basis_vecs[kj]

            angles = subspace_angles(o_basis_vecs, e_basis_vecs)
            angles_deg = np.degrees(angles)
            dihedral_angle_deg = angles_deg[1]
            subspace_angles_grid[i,j] = dihedral_angle_deg

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    im = ax.imshow(subspace_angles_grid, cmap='coolwarm', vmin=-90., vmax=90.)

    # Add text annotations to each cell
    for i in range(num_k_values):
        for j in range(num_k_values):
            angle = subspace_angles_grid[i, j]
            text = ax.text(
                j, i,
                f'{angle:.1f}°',
                ha="center",
                va="center", 
                color="black" if abs(angle) < 45 else "white",
                fontsize=10
            )

    # Add axis labels
    ax.set_xlabel('basis vectors derived from W_E Fourier Coefficients', fontsize=10)
    ax.set_ylabel(
        'basis vectors derived from\no vectors Fourier Coefficients',
        fontsize=10,
        rotation=0,
        ha='right',
        labelpad=20
    )

    # Optional: add colorbar
    plt.colorbar(im, ax=ax, label='Dihedral Angle (°)')

    # Optional: set tick labels if you want to show k_values
    ax.set_xticks(range(num_k_values))
    ax.set_yticks(range(num_k_values))
    ax.set_xticklabels(k_values)
    ax.set_yticklabels(k_values)
    plt.show()

def visualize_W_up_PCA(
        model: Transformer,
        o_dict: dict[str, np.ndarray],
        k_values: list[int] = [4, 32, 43],
):
    """
    Finds the basis vectors for the o vectors as well as the W_E vectors
    """
    k_to_o_basis_vecs = get_embeddings_circle_bases(
        embeddings=o_dict['o'],
        k_values=k_values,
    )

    o_values = o_dict['o']      # shape (P, d_model)
    P, d_model = o_values.shape

    W_E = model.embed.W_E.detach().cpu()                    # shape (d_model, d_vocab)
    W_E = W_E[:,:-1]                                        # shape (d_model, P)

    k_to_embedding_basis_vecs = get_WE_circle_bases(
        W_E=W_E,
        k_values=k_values,
    )
    # Each basis in each of the above dicts are shape (d_model, 2)

    W_up = model.blocks[0].mlp.W_up.detach().cpu()          # shape (d_mlp, d_model)

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(14, 5))
    cmap = plt.get_cmap('coolwarm', 7)
    Z = 5.
    SCATTER_WIDTH = 30
    LINEWIDTH = 1
    COLOR = 'darkslategrey'
    scatter_kwargs = { 's': SCATTER_WIDTH, 'color': COLOR, 'alpha': 0.7 }
    line_kwargs = { 'linewidth': LINEWIDTH, 'color': COLOR, 'alpha': 0.7 }

    for i, k in enumerate(k_values):
        ax = axes[i]
        W_up_projected = W_up @ k_to_o_basis_vecs[k]
        
        # Scatter plot: this is really inefficient lol
        for v in W_up_projected:
            draw_vector(ax, v, scatter_kwargs, line_kwargs)

        beautify_ax(
            ax,
            xmin=-Z,
            xmax=Z,
            ymin=-Z,
            ymax=Z
        )
        ax.set_title(
            f'Fourier Coeffs for {k} Hz ',
            fontsize=10
        )
    
    plt.suptitle(f'$\\bf{{W\_up\ vectors\ in\ 2D\ Subspace\ Corresponding\ To:}}$', fontsize=11)
    plt.show()

def profile_b_up(model: Transformer, do_b_down_instead: bool = False):
    """
    What the fuck's up with this MLP? MLP is severely over provisioned.
    No superposition necessary
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 4))
    if do_b_down_instead:
        b = model.blocks[0].mlp.b_down.detach().cpu().numpy()
    else:
        b = model.blocks[0].mlp.b_up.detach().cpu().numpy()
    b_top_8_idxs = np.argsort(b)[::-1][:8]
    print(f'b_top_8_idxs: {b_top_8_idxs}')
    b.sort()

    x_axis = np.arange(len(b))

    # color
    max_abs_value = max(np.abs(b))
    norm = Normalize(vmin=-max_abs_value, vmax=max_abs_value)
    cmap = plt.cm.coolwarm
    colors = cmap(norm(b))

    ax.bar(x_axis, b, width=1.0, color=colors)
    ax.set_facecolor(BACKGROUND_COLOR)
    ax.set_title(f'b_up values sorted', fontsize=10)
    plt.show()

def profile_W_up_singular_values(model: Transformer):
    """
    Well... replacing W_up with Linear layer doesn't super work either,
    so what's going on?
    """
    W_up = model.blocks[0].mlp.W_up.detach().cpu().numpy() # shape (d_mlp, d_model)

    singular_values = np.linalg.svd(W_up, compute_uv=False)

    x_axis = np.arange(len(singular_values))

    # Color based on magnitude
    max_abs_value = max(np.abs(singular_values))
    norm = Normalize(vmin=-max_abs_value, vmax=max_abs_value)
    cmap = plt.cm.coolwarm
    colors = cmap(norm(singular_values))

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.bar(x_axis, singular_values, width=1.0, color=colors)
    ax.set_facecolor(BACKGROUND_COLOR)
    ax.set_title('Singular values of W_up', fontsize=10)
    ax.set_xlabel('Index')
    ax.set_ylabel('Singular value')
    plt.show()

def profile_W_up_singular_vector_spaces(model: Transformer):
    """
    Identifies the top 8 singular vectors and visualizes all 2D subspaces
    that could be spanned by pairs of them (64 total)
    """
    TOP_K = 8
    W_up = model.blocks[0].mlp.W_up.detach().cpu().numpy() # shape (d_mlp, d_model)

    # SVD
    U, S, Vt = np.linalg.svd(W_up)

    # Get top 8 singular vectors (right singular vectors)
    # These are the first 8 rows of Vt, or equivalently, first 8 columns of V
    V = Vt.T
    top_k_vectors = V[:, :TOP_K]  # shape (128, 8)

    # Project all 512 rows of W_up onto these 8 directions
    projections = W_up @ top_k_vectors  # shape (512, 8)

    # Also compute each dimension (standard basis vector e_i)'s similarity
    # to the 8D subspace spanned by the top 8 eigen vecs. This will show
    # the top 8 dimensions that are most spanned by the eigen vecs.
    basis_vecs = np.identity(128)
    subspace_angles_deg = []
    for basis_vec in basis_vecs:
        angles = subspace_angles(top_k_vectors, basis_vec[:,None])
        principle_angle = angles[0]
        angle_deg = np.degrees(principle_angle)
        subspace_angles_deg.append(angle_deg)
    subspace_angles_deg = np.array(subspace_angles_deg)
    
    # Smallest subspace angle idxs
    smallest_subspace_angle_idxs = np.argsort(subspace_angles_deg)[:8]
    print(f'Most similar dimensions to top 8 singular vectors: {smallest_subspace_angle_idxs}')
    print(f'With degrees: {subspace_angles_deg[smallest_subspace_angle_idxs]}')

    # Now plot all 64 pairs (8x8 grid)
    fig, axes = plt.subplots(8, 8, figsize=(12, 12))
    cmap = plt.get_cmap('coolwarm', 7)
    Z = 2.5

    for i in range(8):
        for j in range(8):
            ax = axes[i, j]
            
            # Get coordinates for the two directions
            x_coords = projections[:, j]  # column j for x-axis
            y_coords = projections[:, i]  # column i for y-axis
            
            # Plot the 512 points
            ax.scatter(x_coords, y_coords, color=cmap(2), alpha=0.5, s=3)
            ax.set_title(f'SV{j + 1} vs SV{i + 1}', fontsize=8)
            ax.set_xlabel(f'SV{j + 1}', fontsize=6)
            ax.set_ylabel(f'SV{i + 1}', fontsize=6)
            ax.tick_params(labelsize=6)
            beautify_ax(ax, -Z, Z, -Z, Z)

    plt.tight_layout()
    plt.show()

def inspect_mlp_acts_periodic_nature(
        model: Transformer,
        show_norms: bool = True,
    ):
    """
    @param show_norms:      If True, show the bar chart of fourier norms.
                            If False, show the activations in fourier coeff space
    """
    # Create sample pairs
    P = 113
    x = 70
    y = list(range(P))
    sample_pairs = [(x, _y, P) for _y in y]

    # Make the dataloader (full batch)
    sample_dataset = MyDataset(sample_pairs)
    sample_dataloader = DataLoader(sample_dataset, batch_size = len(sample_dataset))

    # Install the attn activation cache
    cache = {}
    model.remove_all_hooks()
    model.cache_all(cache)

    # Forward pass and collect the activations
    model.eval()
    mlp_acts_cached = None
    num_heads = None

    for batch_x, batch_y in sample_dataloader:
        _ = model(batch_x)
        mlp_acts_cached = cache['blocks.0.mlp.hook_post']   # (b, num_tokens, d_mlp)

    # Only interested in the mlp activations of the `=` token
    mlp_acts_cached = mlp_acts_cached[:, -1, :]             # (b = P, d_mlp)
    mlp_acts_cached = mlp_acts_cached                       # (P, d_mlp)
    P, _ = mlp_acts_cached.shape

    negative_count = (mlp_acts_cached < 0).sum().item()
    assert negative_count == 0, f'{negative_count} negative entries found in MLP outputs?'

    fourier_coeffs = get_fourier_coeffs_by_hand(mlp_acts_cached.T, P)
    fourier_coeff_norms = fourier_coeffs.norm(dim=0)

    if show_norms:
        _, ax = plt.subplots(1, 1, figsize=(12, 4))
        cmap = plt.get_cmap('coolwarm', 7)

        x_axis = np.linspace(1, P, P - 1)
        x_ticks = [i for i in range(0, P, 10)]
        x_tick_labels = [i // 2 for i in range(0, P, 10)]

        colors = [cmap(2) if i % 2 == 0 else cmap(5) for i in range(P)]
        ax.bar(x_axis, fourier_coeff_norms[1:], width=0.6, color=colors)
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_tick_labels)                               # shape (d_model, 2)
        ax.set_facecolor(BACKGROUND_COLOR)
    else:
        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(14, 7))
        K_VALUES = [4, 32, 43]

        for i, k in enumerate(K_VALUES):
            ax = axes[i]
            basis_vecs = fourier_coeffs[:, [1 + 2 * (k - 1), 2 + 2 * (k - 1)]]
            basis_vecs_norm = basis_vecs.norm(p=2, dim=0, keepdim=True)
            basis_vecs /= basis_vecs_norm    # shape (d_model, 2)

            projected = mlp_acts_cached @ basis_vecs

            b1_mag = projected[:,0].numpy()
            b2_mag = projected[:,1].numpy()

            # We do milli_periods because cmaps can't give us decimal periods
            milli_period = int(1000 * P / k)
            cmap = plt.get_cmap('coolwarm', milli_period)
            colors = [cmap(i * 1000 % milli_period) for i in range(P)]

            # Scatter plot
            ax.scatter(b1_mag, b2_mag, c=colors, s=50, alpha=0.8)

            Z = 40.0
            beautify_ax(
                ax,
                xmin=-Z,
                xmax=Z,
                ymin=-Z,
                ymax=Z
            )
            ax.set_xlabel('PC 1')
            ax.set_ylabel('PC 2')
            ax.set_title(
                f'{k} Hz Circle',
                fontsize=10
            )
        plt.suptitle('MLP activations in 2D subspace corresponding to:')

    plt.show()

def show_imperfect_circle(
        model: Transformer,
        k: int
    ):
    """
    Show the empeddings in MLP activation space

    @param show_norms:      If True, show the bar chart of fourier norms.
                            If False, show the activations in fourier coeff space
    """
    # Create sample pairs
    P = 113
    x = 100
    y = list(range(P))
    sample_pairs = [(x, _y, P) for _y in y]

    # Make the dataloader (full batch)
    sample_dataset = MyDataset(sample_pairs)
    sample_dataloader = DataLoader(sample_dataset, batch_size = len(sample_dataset))

    # Install the attn activation cache
    cache = {}
    model.remove_all_hooks()
    model.cache_all(cache)

    # Forward pass and collect the activations
    model.eval()
    mlp_acts_cached = None

    for batch_x, batch_y in sample_dataloader:
        _ = model(batch_x)
        mlp_acts_cached = cache['blocks.0.mlp.hook_post']   # (b, num_tokens, d_mlp)

    # Only interested in the mlp activations of the `=` token
    mlp_acts_cached = mlp_acts_cached[:, -1, :]             # (b = P, d_mlp)
    P, _ = mlp_acts_cached.shape

    fourier_coeffs = get_fourier_coeffs_by_hand(mlp_acts_cached.T, P)

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 8))

    basis_vecs = fourier_coeffs[:, [1 + 2 * (k - 1), 2 + 2 * (k - 1)]]
    basis_vecs_norm = basis_vecs.norm(p=2, dim=0, keepdim=True)
    basis_vecs /= basis_vecs_norm    # shape (d_model, 2)

    projected = mlp_acts_cached @ basis_vecs

    # # Do additional step of centering
    # projected -= projected.mean(dim=0, keepdim=True)

    b1_mag = projected[:,0].numpy()
    b2_mag = projected[:,1].numpy()

    # We do milli_periods because cmaps can't give us decimal periods
    milli_period = int(1000 * P / k)
    cmap = plt.get_cmap('coolwarm', milli_period)
    colors = [cmap(i * 1000 % milli_period) for i in range(P)]

    # Scatter plot
    SCATTER_WIDTH = 50
    ax.scatter(b1_mag, b2_mag, c=colors, s=SCATTER_WIDTH, alpha=0.8)

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

    LINEWIDTH = 1
    COLOR = 'darkslategrey'
    FEATURE_ALPHA = 1.0
    scatter_kwargs = { 's': SCATTER_WIDTH, 'color': COLOR, 'alpha': FEATURE_ALPHA }
    line_kwargs = { 'linewidth': LINEWIDTH, 'color': COLOR, 'alpha': FEATURE_ALPHA, 'zorder': 10 }
    # Plot a vector to represent probe for feature 65
    draw_vector(
        ax,
        v = projected[65] * 0.7,
        scatter_kwargs=scatter_kwargs,
        line_kwargs=line_kwargs
    )
    ax.annotate(
        '$\\bf{{W\_L[65]\ (candidate)}}$',
        projected[65] * 0.7,
        fontsize=10,
        alpha=0.7,
        xytext=(3, 3),
        textcoords='offset points'
    )

    # Extended dashed line along the feature's direction
    dotted_line_x = np.linspace(0, projected[65, 0] * 10, 5)
    dotted_line_y = np.linspace(0, projected[65, 1] * 10, 5)
    ax.plot(dotted_line_x, dotted_line_y, '--', color='grey', linewidth=1.5, alpha=0.6, zorder=-10)

    Z = 30.0
    beautify_ax(
        ax,
        xmin=-Z,
        xmax=Z,
        ymin=-Z,
        ymax=Z
    )
    ax.set_xlabel('PC 1')
    ax.set_ylabel('PC 2')
    # ax.set_xticks([])
    # ax.set_yticks([])
    ax.set_title('Up Close: 4 Hz Circle (MLP Activations)', fontsize=11)

    plt.show()

def show_WL_wrt_MLP_outs(
        model: Transformer,
        expected_ans: int = 65
    ):
    """
    @param feat_i:      the expected answer
    """
    # Create sample pairs
    P = 113
    x = 70
    y = list(range(P))
    sample_pairs = [(x, _y, P) for _y in y]

    # Make the dataloader (full batch)
    sample_dataset = MyDataset(sample_pairs)
    sample_dataloader = DataLoader(sample_dataset, batch_size = len(sample_dataset))

    # Install the attn activation cache
    cache = {}
    model.remove_all_hooks()
    model.cache_all(cache)

    # Forward pass and collect the activations
    model.eval()
    mlp_outs_cached = None

    for batch_x, batch_y in sample_dataloader:
        _ = model(batch_x)
        mlp_outs_cached = cache['blocks.0.hook_mlp_out']    # (b, num_tokens, d_model)

    # Only interested in the mlp activations of the `=` token
    mlp_outs_cached = mlp_outs_cached[:, -1, :]                 # (b = P, d_model)
    P, _ = mlp_outs_cached.shape
    fourier_coeffs = get_fourier_coeffs_by_hand(mlp_outs_cached.T, P)

    # Get W_U as well
    W_U = model.unembed.W_U.detach().cpu()                  # shape (d_model, d_vocab)
    W_U = W_U.T                                             # shape (d_vocab, d_model)
    W_U = W_U[:-1,:]                                        # shape (P, d_model)
    ans_feat = W_U[expected_ans]                            # shape (d_model,)

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(16, 8))
    K_VALUES = [4, 32, 43]
    Z = 200.0

    for i, k in enumerate(K_VALUES):
        ax = axes[i]
        basis_vecs = fourier_coeffs[:, [1 + 2 * (k - 1), 2 + 2 * (k - 1)]]
        basis_vecs_norm = basis_vecs.norm(p=2, dim=0, keepdim=True)
        basis_vecs /= basis_vecs_norm                       # shape (d_model, 2)

        mlp_acts_projected = mlp_outs_cached @ basis_vecs
        ans_feat_projected = ans_feat @ basis_vecs                  # shape (2,)
        ans_feat_projected /= ans_feat_projected.norm(keepdim=True)
        ans_feat_projected *= 0.5 * Z

        mlp_b1 = mlp_acts_projected[:,0].numpy()
        mlp_b2 = mlp_acts_projected[:,1].numpy()

        # We do milli_periods because cmaps can't give us decimal periods
        milli_period = int(1000 * P / k)
        cmap = plt.get_cmap('coolwarm', milli_period)
        colors = [cmap(i * 1000 % milli_period) for i in range(P)]

        # Scatter plot
        SCATTER_WIDTH = 50
        ax.scatter(mlp_b1, mlp_b2, c=colors, s=SCATTER_WIDTH, alpha=0.6)

        # Annotate each point with its index
        for p in range(P):
            ax.annotate(
                str(p),
                (mlp_b1[p], mlp_b2[p]),
                fontsize=15 if p ==108 else 8,
                color='white' if p == 108 else 'black',
                alpha=0.7,
                xytext=(3, 3),
                textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.8) if p == 108 else None
            )

        LINEWIDTH = 1
        COLOR = 'darkslategrey'
        FEATURE_ALPHA = 1.0
        scatter_kwargs = { 's': SCATTER_WIDTH, 'color': COLOR, 'alpha': FEATURE_ALPHA }
        line_kwargs = {
            'linewidth': LINEWIDTH,
            'color': COLOR,
            'alpha': FEATURE_ALPHA,
            'zorder': 10 
        }
        # Plot a vector to represent probe for feature 65
        draw_vector(
            ax,
            v = ans_feat_projected * 0.7,
            scatter_kwargs=scatter_kwargs,
            line_kwargs=line_kwargs
        )
        ax.annotate(
            f'$\\bf{{W\_L[{expected_ans}]}}$',
            ans_feat_projected * 0.7,
            fontsize=10,
            alpha=0.7,
            xytext=(3, 3),
            textcoords='offset points'
        )

        # Extended dashed line along the feature's direction
        dotted_line_x = np.linspace(0, ans_feat_projected[0] * 10, 5)
        dotted_line_y = np.linspace(0, ans_feat_projected[1] * 10, 5)
        ax.plot(dotted_line_x, dotted_line_y, '--', color='grey', linewidth=1.5, alpha=0.6, zorder=-10)

        beautify_ax(
            ax,
            xmin=-Z,
            xmax=Z,
            ymin=-Z,
            ymax=Z
        )
        ax.set_xlabel('PC 1')
        ax.set_ylabel('PC 2')
        ax.set_title('Up Close: 4 Hz Circle (MLP outputs)', fontsize=11)

    plt.show()

def o_circles_clockwork(
        model: Transformer,
        a_values: list[int],
        k: int,
        expected_ans: int | None,
        do_headwise_computation: bool = False,
        hook_point: str = 'blocks.0.attn.hook_o',
        use_basis_cached: bool = True,
):
    """
    @param k:       frequency of circle space interested
    """
    # Create sample pairs
    P = 113
    b_values = list(range(P))

    fig, axes = plt.subplots(
        nrows=(3 + len(a_values)) // 4,
        ncols=min(len(a_values), 4),
        figsize=(12, 12)
    )
    axes = axes.flatten()
    SCATTER_SIZE = 20
    SCATTER_ALPHA = 0.7

    # Get W_U to plot ans vectors
    W_U = model.unembed.W_U.detach().cpu()                  # shape (d_model, d_vocab)
    W_U = W_U.T                                             # shape (d_vocab, d_model)
    W_U = W_U[:-1,:]                                        # shape (P, d_model)
    ans_feat = W_U[expected_ans]                            # shape (d_model,)

    basis_cached = None
    bases = []
    for a_i, a in enumerate(a_values):
        if isinstance(expected_ans, int):
            b = (expected_ans - a) % P
        ax = axes[a_i]
        sample_pairs = [(a, b, P) for b in b_values]

        # Make the dataloader (full batch)
        sample_dataset = MyDataset(sample_pairs)
        sample_dataloader = DataLoader(sample_dataset, batch_size = len(sample_dataset))

        # Install the attn activation cache
        cache = {}
        model.remove_all_hooks()
        model.cache_all(cache)

        # Forward pass and collect the activations
        model.eval()
        z_values_by_head = {}
        embeddings_cached = None
        o_values_by_head = {}
        num_heads = None

        for batch_x, _ in sample_dataloader:
            _ = model(batch_x)

            z_values = cache['blocks.0.attn.hook_z']                # (b, num_heads, num_tokens, d_head)
            embeddings_cached = cache[hook_point]         # (b, num_tokens, d_model)

            _, num_heads, _, _ = z_values.shape
            for i in range(num_heads):
                z_values_by_head[f'head_{i}'] = z_values[:,i,:,:]

        if do_headwise_computation:
            # I still need the fourier coeff basis from W_E for k
            W_E = model.embed.W_E.detach().cpu()                        # shape (d_model, d_vocab)
            W_E = W_E[:,:-1]                                            # shape (d_model, P)
            fourier_coeffs = get_fourier_coeffs_by_hand(W_E, P)

            # Because we want to extract those 2 dimensions of the embeddings, then
            # apply W_V to it
            basis_vecs = fourier_coeffs[:, [1 + 2 * (k - 1), 2 + 2 * (k - 1)]] # shape (d_model, 2)
            basis_vecs_norm = basis_vecs.norm(p=2, dim=0, keepdim=True)
            basis_vecs /= basis_vecs_norm                            

            # I need W_V and W_O because I need to transform embeddings from W_E to W_V to W_O space
            W_V = model.blocks[0].attn.W_V.detach().cpu()       # shape (num_heads, d_head, d_model)
            W_O = model.blocks[0].attn.W_O.detach().cpu()       # shape (d_model, num_heads * d_head)

            # Calculate the basis transformation that would allow us to visualize in 2D
            o_projected_by_head = {}
            for head_i in range(num_heads):
                W_V_this_head = W_V[head_i,:,:]                         # shape (d_head, d_model)
                W_V_this_head = W_V_this_head.T                         # shape (d_model, d_head)
                W_V_pinv = compute_pinv_by_hand(W_V_this_head)          # shape (d_head, d_model)

                num_heads, d_head, d_model = W_V.shape
                W_O_this_head = W_O[:, head_i * d_head: (head_i + 1) * d_head]  # shape (d_model, d_head)
                W_O_this_head = W_O_this_head.T                                 # shape (d_head, d_model)
                W_O_pinv = compute_pinv_by_hand(W_O_this_head)                  # shape (d_model, d_head)

                WV_WO_pinv = W_O_pinv @ W_V_pinv                                # shape (d_model, d_model)
                new_basis = WV_WO_pinv @ basis_vecs                             # shape (d_model, 2)

                z_values_this_head = z_values_by_head[f'head_{head_i}']         # shape (b, num_tokens, d_head)
                o_values_this_head = z_values_this_head @ W_O_this_head         # shape (b, num_tokens, d_model)
                # Only want to focus on the `=` token
                o_values_this_head = o_values_this_head[:,-1,:]                 # shape (b, d_model)

                # Project onto the 2 directions we computed from W_E
                o_values_projected = (o_values_this_head @ new_basis).numpy()
                o_projected_by_head[head_i] = o_values_projected

            # Overall
            # This should work
            o_values_projected = np.vstack([projected[None,:,:] for projected in o_projected_by_head.values()])
            o_values_projected = np.sum(o_values_projected, axis=0)
            embeddings_projected = o_values_projected

            Z = 2.0
        else:
            embeddings = embeddings_cached[:,-1,:]
            if use_basis_cached and isinstance(basis_cached, (np.ndarray, torch.Tensor)):
                print('using cached basis')
                basis_vecs = basis_cached
            else:
                k_to_basis_vecs = get_embeddings_circle_bases(
                    embeddings,
                    k_values=[k]
                )
                basis_vecs = k_to_basis_vecs[k]
                basis_cached = basis_vecs
                bases.append(basis_vecs)

            embeddings_projected = embeddings @ basis_vecs

            if hook_point == 'blocks.0.attn.hook_o':
                Z = 10.
            elif hook_point == 'blocks.0.hook_mlp_out':
                Z = 150.
            else:
                Z = 30.

        b1_mag = embeddings_projected[:,0]
        b2_mag = embeddings_projected[:,1]

        # We do milli_periods because cmaps can't give us decimal periods
        milli_period = int(1000 * P / k)
        cmap = plt.get_cmap('coolwarm', milli_period)
        colors = [cmap(i * 1000 % milli_period) for i in range(P)]

        # Do scatter
        ax.scatter(b1_mag, b2_mag, color=colors, s=SCATTER_SIZE, alpha=SCATTER_ALPHA)
        # BIG scatter
        for anchor_p in [3, 33, 63]:
            ax.scatter(b1_mag[anchor_p], b2_mag[anchor_p], color='black', s=SCATTER_SIZE * 2, alpha=1.0)
            ax.annotate(
                str(anchor_p),
                (b1_mag[anchor_p], b2_mag[anchor_p]),
                fontsize=8,
                alpha=0.7,
                xytext=(3, 3),
                textcoords='offset points'
            )
        ax.set_title(f'(a = {a}) + b = {expected_ans or ""}', fontsize=9)

        # Want to also just plot where on earth the embeddings are
        if '.mlp.' not in hook_point:
            W_E = model.embed.W_E.detach().cpu()                        # shape (d_model, d_vocab)
            W_E = W_E[:,:-1]                                            # shape (d_model, P)
            W_E = W_E.T                                                 # shape (P, d_model)
            W_E = W_E[:,None,:]                                         # shape (P, 1, d_model)

            # Transform these into O space
            W_V = model.blocks[0].attn.W_V.detach().cpu()               # shape (num_heads, d_head, d_model)
            W_O = model.blocks[0].attn.W_O.detach().cpu()               # shape (d_model, num_heads * d_head)
            number_feats = torch.einsum(
                'ihd,bpd -> biph',
                W_V,
                W_E
            )
            number_feats = einops.rearrange(number_feats, 'b i p h -> b p (i h)')
            number_feats = torch.einsum('df,bqf->bqd', W_O, number_feats)   # shape (P, 1, d_model)
            number_feats = number_feats[:,0,:]                              # shape (P, d_model)
            number_feats = (number_feats @ basis_vecs).numpy()

            ax.scatter(
                number_feats[:,0],
                number_feats[:,1],
                color='grey', s=SCATTER_ALPHA, alpha=SCATTER_ALPHA
            )

        # Plot W_U line for expected answer
        if isinstance(expected_ans, int):
            LINEWIDTH = 1
            COLOR = 'darkslategrey'
            FEATURE_ALPHA = 1.0
            scatter_kwargs = { 's': SCATTER_SIZE, 'color': COLOR, 'alpha': FEATURE_ALPHA }
            line_kwargs = {
                'linewidth': LINEWIDTH,
                'color': COLOR,
                'alpha': FEATURE_ALPHA,
                'zorder': 10 
            }
            # Plot a vector to represent linear direction of answer
            ans_feat_projected = ans_feat @ basis_vecs                  # shape (2,)
            ans_feat_projected /= ans_feat_projected.norm(keepdim=True)
            ans_feat_projected *= 0.5 * Z

            draw_vector(
                ax,
                v = ans_feat_projected * 0.7,
                scatter_kwargs=scatter_kwargs,
                line_kwargs=line_kwargs
            )
            ax.annotate(
                f'$\\bf{{W\_L[{expected_ans}]}}$',
                ans_feat_projected * 0.7,
                fontsize=10,
                alpha=0.7,
                xytext=(3, 3),
                textcoords='offset points'
            )

            # Extended dashed line along the feature's direction
            dotted_line_x = np.linspace(0, ans_feat_projected[0] * 10, 5)
            dotted_line_y = np.linspace(0, ans_feat_projected[1] * 10, 5)
            ax.plot(dotted_line_x, dotted_line_y, '--', color='grey', linewidth=1.5, alpha=0.6, zorder=-10)

    for ax in axes:
        beautify_ax(ax, -Z, Z, -Z, Z)
        # ax.set_xticks([])
        # ax.set_yticks([])

    # plt.suptitle(
    #     f'$\\bf{{o\ vectors}}$\nin 2D subspace corresponding to freq {k} Hz embedding circle',
    #     fontsize=11
    # )
    plt.show()

    # subspace angles
    plt.close()
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    subspace_angle_matrix = np.zeros((len(bases), len(bases)))
    for i, basis_i in enumerate(bases):
        for j, basis_j in enumerate(bases):
            principle_angle = subspace_angles(basis_i, basis_j)[0]
            principle_angle_deg = np.degrees(principle_angle)
            subspace_angle_matrix[i, j] = principle_angle_deg
            
    ax.imshow(subspace_angle_matrix, cmap='coolwarm', vmin=-90., vmax=90.)
    # Add text annotations to each cell
    for i in range(len(bases)):
        for j in range(len(bases)):
            angle = subspace_angle_matrix[i, j]
            text = ax.text(
                j, i,
                f'{angle:.0f}°',
                ha="center",
                va="center",
                color="black" if abs(angle) < 45 else "white",
                fontsize=7
            )
    
    plt.show()


if __name__ == '__main__':
    CHECKPOINT_FILE = 'checkpoints/grokked_20k/epoch_19999.pt'
    # CHECKPOINT_FILE = 'sparse_checkpoints/grokked_10k/epoch_9999.pt'
    DATA_FILE = 'datasets/grokked_20k/dataset.pkl'

    model = load_model(CHECKPOINT_FILE)
    train_pairs, test_pairs = load_data(DATA_FILE)
    # inspect_periodic_nature(model, weight_matrix='W_L', do_DFT_by_hand=True)
    # inspect_PCA_W_E(model, weight_matrix='W_L', k_vals=[4, 32, 43])
    # inspect_attention_maps(model, test_pairs, num_samples=6, show_only_last_attn_row=True, full_p_by_p_plot=True)
    # inspect_attention_maps_periodic_nature(model)

    # inspect_attention_z_values(
    #     model,
    #     head_i=1,
    #     k=32,
    #     plot_interpolation=True,
    #     annotate_scatter=True
    # )

    # inspect_attention_outputs(
    #     model,
    #     head_i=3,
    #     k=4,
    #     plot_interpolation=True,
    #     annotate_scatter=False,
    # )

    # This saves the o / o_projected vectors used in the next function
    K = 43
    # inspect_attention_outputs_in_agg(
    #     model,
    #     k = K,
    #     o_projected_save_loc=O_PROJECTED_SAVE_LOC.format(k=K)
    # )

    # o_dict = np.load(O_PROJECTED_SAVE_LOC.format(k=K))
    # inspect_attention_outputs_periodic_nature(
    #     model,
    #     o_dict,
    #     do_projected=False
    # )

    # visualize_o_circles(
    #     o_dict
    # )

    # SECTION 8
    # do_WE_and_o_coexist(model, o_dict)

    # visualize_W_up_PCA(model, o_dict)

    # profile_b_up(model, do_b_down_instead=True)

    # profile_W_up_singular_values(model)

    # profile_W_up_singular_vector_spaces(model)

    # inspect_mlp_acts_periodic_nature(model, show_norms=False)

    # show_imperfect_circle(model, k = 4)

    # show_WL_wrt_MLP_outs(model, expected_ans=65)

    # WIP (discovery made)
    o_circles_clockwork(
        model,
        a_values = [i for i in range(30, 60)],
        expected_ans=None,
        k = 4,
        # hook_point='blocks.0.hook_mlp_out'
        hook_point='blocks.0.mlp.hook_post',
        use_basis_cached=False,
    )