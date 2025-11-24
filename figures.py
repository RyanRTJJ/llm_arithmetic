import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle
from matplotlib import collections as mc

BACKGROUND_COLOR = '#FCFBF8'

def beautify_ax(ax: plt.Axes, xmin: float, xmax: float, ymin: float, ymax: float, ignore_aspect_ratio=False):
    ax.set_facecolor(BACKGROUND_COLOR)
    ax.set_xlim((xmin, xmax))
    ax.set_ylim((ymin, ymax))
    if not ignore_aspect_ratio:
        aspect_ratio = (xmax - xmin) / (ymax - ymin)
        ax.set_aspect(aspect_ratio)

def do_curve_thingy_animation():
    circle_freq = 0.5
    alpha_freq = 0.5
    alpha_period_offsets = [0.0, 0.5]  # Different offsets for each subplot

    # Animation settings
    FPS = 100
    DURATION = 6
    CMAP = 'coolwarm'
    cmap = plt.get_cmap(CMAP, 7)
    num_frames = FPS * DURATION
    LEGEND_FONTSIZE = 9
    ALPHA_LEGEND_FONTSIZE = 10

    # Fixed point x at (-1, 0)
    a = np.array([-1.0, 0.0])

    # Time array
    T = np.linspace(0, DURATION, num_frames + 1)[:num_frames]

    # Pre-calculate all positions for the trajectory
    all_data = []
    for alpha_period_offset in alpha_period_offsets:
        b_positions = np.zeros((num_frames, 2))
        alpha_values = np.zeros(num_frames)
        o_positions = np.zeros((num_frames, 2))

        for i, t in enumerate(T):
            # b moves around the circle
            angle = 2 * np.pi * circle_freq * t
            b_positions[i] = [np.cos(angle), np.sin(angle)]

            # alpha oscillates between 0 and 1
            # % of the way to 1 full period is t / wavelength * 2 pi = t * alpha_freq * 2 pi
            # If I want to add 0.5 full periods, I get:
            # (t + 0.5 * wavelength) / wavelength * 2 pi = ((t * alpha) + 0.5) * 2 pi
            alpha_values[i] = 0.5 - 0.5 * np.cos((t * alpha_freq + alpha_period_offset) * 2 * np.pi)

            # p is convex combination: p = alpha * x + (1 - alpha) * b
            o_positions[i] = alpha_values[i] * a + (1 - alpha_values[i]) * b_positions[i]

        all_data.append({
            'b_positions': b_positions,
            'alpha_values': alpha_values,
            'o_positions': o_positions,
            'offset': alpha_period_offset
        })


    # Set up the figure and axis
    fig, axes = plt.subplots(nrows=1, ncols=len(alpha_period_offsets), figsize=(20, 8))
    plt.suptitle(f'Embedding freq == attention freq == {circle_freq:.2f} Hz', fontweight='bold')
    Z = 1.5

    plot_elements = []

    for _, (ax, data) in enumerate(zip(axes, all_data)):
        beautify_ax(ax, -Z, Z, -Z, Z)
        ax.set_title(f'Attention (alpha) offset (wavelength): {data["offset"]:.2f}', fontsize=12)


        # Draw the unit circle
        circle = Circle((0, 0), 1, fill=False, color=cmap(3), linewidth=2, linestyle='--')
        ax.add_patch(circle)

        # Initialize plot elements
        point_a, = ax.plot([], [], 'o', color=cmap(2), markersize=12, label='a', zorder=5)
        point_b, = ax.plot([], [], 'o', color=cmap(5), markersize=12, label='b', zorder=5)
        point_o, = ax.plot([], [], 'ko', markersize=10, label='z', zorder=5)
        line_ab, = ax.plot([], [], '-', color='grey', linewidth=1, alpha=0.5)
        trajectory, = ax.plot([], [], '-', color='grey', linewidth=1.5, alpha=1.0, label='trajectory of z')

        # Text for displaying alpha value
        alpha_text = ax.text(
            0.02,
            0.98,
            '',
            transform=ax.transAxes,
            verticalalignment='top',
            fontsize=ALPHA_LEGEND_FONTSIZE,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        )

        ax.legend(loc='upper right', fontsize=LEGEND_FONTSIZE)
        plot_elements.append({
            'point_a': point_a,
            'point_b': point_b,
            'point_o': point_o,
            'line_ab': line_ab,
            'trajectory': trajectory,
            'alpha_text': alpha_text,
            'data': data
        })

    def init():
        """Initialize animation"""
        artists = []
        for elem in plot_elements:
            elem['point_a'].set_data([a[0]], [a[1]])
            elem['point_b'].set_data([], [])
            elem['point_o'].set_data([], [])
            elem['line_ab'].set_data([], [])
            elem['trajectory'].set_data([], [])
            elem['alpha_text'].set_text('')
            artists.extend([
                elem['point_a'], 
                elem['point_b'], 
                elem['point_o'],
                elem['line_ab'],
                elem['trajectory'],
                elem['alpha_text']
            ])
        return artists

    def animate(frame):
        """Update animation frame"""
        artists = []
        for elem in plot_elements:
            data = elem['data']

            # Update b position
            b = data['b_positions'][frame]
            elem['point_b'].set_data([b[0]], [b[1]])

            # Update o position
            o = data['o_positions'][frame]
            elem['point_o'].set_data([o[0]], [o[1]])

            # Update line from a to b
            elem['line_ab'].set_data([a[0], b[0]], [a[1], b[1]])

            # Update trajectory
            elem['trajectory'].set_data(
                data['o_positions'][:frame + 1, 0],
                data['o_positions'][:frame + 1, 1]
            )

            # Update alpha text
            elem['alpha_text'].set_text(f'α = {data["alpha_values"][frame]:.2f}')

            artists.extend([
                elem['point_a'],
                elem['point_b'],
                elem['point_o'],
                elem['line_ab'],
                elem['trajectory'],
                elem['alpha_text']
            ])
        return artists

    # Create animation
    anim = FuncAnimation(
        fig,
        animate,
        init_func=init,
        frames=num_frames,
        interval=1000/FPS,
        blit=True,
        repeat=True
    )

    # Save animation
    # print("Saving animation... This may take a minute.")
    # anim.save('convex_trajectory.mp4', writer='ffmpeg', fps=FPS, dpi=100)
    # print("Animation saved!")

    plt.show()

def do_family_of_curves(
        circle_freq: float,
        alpha_freqs: list[float],
        num_periodss: list[float],
        do_animation: bool = False,
        alpha_period_offsets: list[float] | np.ndarray | None = None,
        plot_scatter: bool = False,
        num_trajectory_samples: int = 1000,
    ):
    """
    Plots animations of trajectories over various phase offsets for attention freq.

    @param num_periodss:            For each value in alpha_freqs, provide a num_periods
                                    of periods of the attention wave to draw. Usually, the
                                    higher the alpha_freq, the more waves for a complete
                                    trajectory.
    @param do_animation:            If True, animations of phase offset from 0. --> 1.0
                                    will be made. if False, we expect `alpha_period_offset`
                                    to be given.
    @param alpha_period_offsets:    When not `do_animation`, this needs to be provided cuz
                                    this function will draw a static image of the trajectory
                                    at some given phase offset.
    @param plot_scatter:            Line plot or scatter plot?
    @param num_trajectory_samples:  Self-explanatory.
    """
    # circle_freq = 0.5
    # alpha_freqs = [0.5]

    # Animation Settings
    FPS = 100
    DURATION = 2.5
    CMAP = 'coolwarm'
    cmap = plt.get_cmap(CMAP, 7)
    num_frames = int(FPS * DURATION)
    LEGEND_FONTSIZE = 9
    NUM_CURVES_TO_NROWS_NCOLS_FIGSIZE = {
        1: ((1, 1), (6, 6)),
        2: ((1, 2), (12, 6)),
        3: ((1, 3), (12, 6)),
        4: ((2, 2), (12, 12)),
    }

    # Fixed point a at (-1, 0)
    a = np.array([-1.0, 0.0])

    # Step 2: Alpha period offset values for animation (from 0 to 1)
    if do_animation:
        alpha_period_offsets = np.linspace(0, 1.0, num_frames)
    else:
        assert isinstance(alpha_period_offsets, (list, np.ndarray)), \
            'Provide alpha_period_offsets when do_animation == False'

    if plot_scatter:
        alpha_period_offsets = alpha_period_offsets[:-1]

    # Set up the figure and axes
    ax_dims, figsize = NUM_CURVES_TO_NROWS_NCOLS_FIGSIZE[len(alpha_freqs)]
    fig, axes = plt.subplots(*ax_dims, figsize=figsize)
    # fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
    if isinstance(axes, plt.Axes):
        axes = np.array([axes])
    axes = axes.flatten()

    plt.suptitle(f'Embedding freq == {circle_freq:.2f} Hz, for different alpha freqs', fontweight='bold')
    # plt.suptitle(f'Embedding freq == attention freq == {circle_freq:.2f} Hz', fontweight='bold')
    Z = 1.5

    plot_elements = []

    for _, (ax, alpha_freq, num_periods) in enumerate(zip(axes, alpha_freqs, num_periodss)):
        beautify_ax(ax, -Z, Z, -Z, Z)
        ax.set_title(f'Alpha freq: {alpha_freq:.2f} Hz', fontsize=12)

        # Draw the unit circle
        circle = Circle((0, 0), 1, fill=False, color=cmap(3), linewidth=2, linestyle='--')
        ax.add_patch(circle)

        # Initialize trajectory line
        if plot_scatter:
            trajectory = ax.scatter([], [], s=50, color=cmap(2), alpha=0.6, label='trajectory of z')
        else:
            trajectory, = ax.plot([], [], '-', color='grey', linewidth=1.5, label='trajectory of z')

        # Text for displaying alpha offset value
        offset_text = ax.text(
            0.02,
            0.98,
            '',
            transform=ax.transAxes,
            verticalalignment='top',
            fontsize=LEGEND_FONTSIZE,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        )

        ax.legend(loc='upper right', fontsize=LEGEND_FONTSIZE)

        plot_elements.append({
            'trajectory': trajectory,
            'offset_text': offset_text,
            'alpha_freq': alpha_freq,
            'num_periods': num_periods,
        })

    def compute_trajectory(alpha_freq, alpha_period_offset, num_periods, num_samples):
        """Compute the complete trajectory for given parameters"""
        # Time array over k full periods of alpha
        period_alpha = 1.0 / alpha_freq
        total_time = num_periods * period_alpha
        t_array = np.linspace(0, total_time, num_samples)

        o_positions = np.zeros((num_samples, 2))

        for i, t in enumerate(t_array):
            # b moves around the circle
            angle = 2 * np.pi * circle_freq * t
            b = np.array([np.cos(angle), np.sin(angle)])

            # alpha oscillates between 0 and 1
            alpha = 0.5 - 0.5 * np.cos((t * alpha_freq + alpha_period_offset) * 2 * np.pi)

            # o is convex combination: o = alpha * a + (1 - alpha) * b
            o_positions[i] = alpha * a + (1 - alpha) * b

        return o_positions

    def init():
        """Initialize animation"""
        artists = []
        for elem in plot_elements:
            if isinstance(trajectory, mc.PathCollection):
                elem['trajectory'].set_offsets(np.empty((0, 2)))
            else:
                elem['trajectory'].set_data([], [])
            elem['offset_text'].set_text('')
            artists.extend([elem['trajectory'], elem['offset_text']])
        return artists

    def compute_o_positions(
            plot_elem: dict,
            alpha_period_offset: float,
            num_trajectory_samples: int
    ):
        """
        Small helper function to be called by animate() and also the non-
        animate code pathway.
        """
        o_positions = compute_trajectory(
            plot_elem['alpha_freq'],
            alpha_period_offset,
            plot_elem['num_periods'],
            num_trajectory_samples
        )
        return o_positions
        

    def animate(frame):
        """Update animation frame"""
        alpha_period_offset = alpha_period_offsets[frame]

        artists = []
        for elem in plot_elements:
            # Compute trajectory for this alpha_freq and offset
            o_positions = compute_o_positions(
                elem,
                alpha_period_offset,
                num_trajectory_samples
            )

            # Update trajectory
            if isinstance(trajectory, mc.PathCollection):
                elem['trajectory'].set_offsets(o_positions)
            else:
                elem['trajectory'].set_data(o_positions[:, 0], o_positions[:, 1])

            # Update offset text
            elem['offset_text'].set_text(f'offset = {alpha_period_offset:.3f}')

            artists.extend([elem['trajectory'], elem['offset_text']])

        return artists

    if do_animation:
        # Create animation
        _ = FuncAnimation(
            fig,
            animate,
            init_func=init,
            frames=num_frames,
            interval=1000/FPS,
            blit=True,
            repeat=True
        )
    else:
        for i, elem in enumerate(plot_elements):
            o_positions = compute_o_positions(
                elem,
                alpha_period_offsets[i],
                num_trajectory_samples
            )

            # Update trajectory
            if isinstance(trajectory, mc.PathCollection):
                elem['trajectory'].set_offsets(o_positions)
            else:
                elem['trajectory'].set_data(o_positions[:, 0], o_positions[:, 1])


    plt.show()

if __name__ == '__main__':
    # do_curve_thingy_animation()

    # ALPHA_FREQS = [1./8, 4. / 43, 2.0, 8.0]
    # NUM_PERIODSS = [4.0, 43. / 4., 3.0, 8.0]

    # ALPHA_FREQS = [1.0, 8.0]
    # NUM_PERIODSS = [1.0, 8.0]

    ALPHA_FREQS = [0.25, 1./3, 1.0, 1.50,]
    NUM_PERIODSS = [2., 3., 2., 3.,]
    do_family_of_curves(
        circle_freq = 0.5,
        alpha_freqs = ALPHA_FREQS,
        num_periodss = NUM_PERIODSS,
        do_animation = True,
        alpha_period_offsets = [0.0, 0.0, 0.0, 0.0],
        plot_scatter = False,
        num_trajectory_samples = 1000
    )
