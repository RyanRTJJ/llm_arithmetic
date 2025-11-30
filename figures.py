from typing import Any

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
from matplotlib import collections as mc

BACKGROUND_COLOR = '#FCFBF8'

class Arrow3D(FancyArrowPatch):
    """
    A helper class to draw 3D arrows
    """
    def __init__(self, xs, ys, zs, *args, **kwargs):
        """
        @param xs:      (List) of length 2, from [src, dst]
        """
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, _ = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        FancyArrowPatch.draw(self, renderer)

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))

        return np.min(zs)

def beautify_ax_3d(ax, Z):
    """
    @param Z:               limits of plot
    """
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlim(-Z, Z)
    ax.set_ylim(-Z, Z)
    ax.set_zlim(-Z, Z)
    ax.set_facecolor(BACKGROUND_COLOR)
    # Blow up the grid
    ax.grid(False)
    ax.set_axis_off()

def draw_arrow_3d(ax, src, dst, arrow_kwargs):
    arrow = Arrow3D(
        [src[0], dst[0]],
        [src[1], dst[1]],
        [src[2], dst[2]],
        **arrow_kwargs
    )
    ax.add_artist(arrow)

def add_axis_lines_3d(ax, Z, arrow_kwargs: dict[str, Any]=None, partial=False, **kwargs):
    """
    @param Z:               arrow spans from -Z to Z
    @param arrow_kwargs:    Optional non-default arrow_kwargs
    """
    # Add new axis lines
    if not arrow_kwargs:
        arrow_kwargs = {'arrowstyle': '-|>', 'mutation_scale': 10, 'lw': 1, 'color': 'black'}

    lb = 0 if partial else -Z
    if not 'omit_x' in kwargs:
        x_axis = Arrow3D([lb, Z], [0, 0], [0, 0], **arrow_kwargs)
        ax.add_artist(x_axis)
    if not 'omit_y' in kwargs:
        y_axis = Arrow3D([0, 0], [lb, Z], [0, 0], **arrow_kwargs)
        ax.add_artist(y_axis)
    if not 'omit_z' in kwargs:
        z_axis = Arrow3D([0, 0], [0, 0], [lb, Z], **arrow_kwargs)
        ax.add_artist(z_axis)

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

def show_shifting_circle():
    fig = plt.figure(figsize=(14, 6))
    cmap = plt.get_cmap('coolwarm', 7)
    edge_cmap = plt.get_cmap('coolwarm')

    N_POINTS = 100
    RADIUS = 1.0
    Z = 3.0
    FILL_COLOR = cmap(3)
    FILL_ALPHA = 0.9
    SHADE = False
    OFFSET_DIST = 1.8
    OFFSET_ARROW_KWARGS = {
        'arrowstyle': '-|>',
        'mutation_scale': 10,
        'lw': 1.5,
        'color': 'black',
        'alpha': 1
    }

    theta = np.linspace(0, 2 * np.pi, N_POINTS)

    ax1 = fig.add_subplot(131, projection='3d')

    x1 = RADIUS * np.cos(theta)
    y1 = RADIUS * np.sin(theta)
    z1 = np.zeros_like(theta)

    # Plot the filled circle
    ax1.plot_trisurf(x1, y1, z1, color=FILL_COLOR, alpha=FILL_ALPHA, shade=SHADE)
    # Plot the border with gradient colors
    for i in range(len(theta) - 1):
        color = edge_cmap(i / (len(theta) - 1))
        ax1.plot(
            [x1[i], x1[i + 1]],
            [y1[i], y1[i + 1]],
            [z1[i], z1[i + 1]],
            color=color,
            linewidth=3
        )

    ax1.set_title('Pre-shift', fontsize=11)

    ax2 = fig.add_subplot(132, projection='3d')

    # Plot the up-shifted circle
    upshifted_z = np.ones_like(theta) * OFFSET_DIST
    ax2.plot_trisurf(
        x1,
        y1,
        upshifted_z,
        color=FILL_COLOR,
        alpha=FILL_ALPHA,
        shade=SHADE
    )
    # Plot the border with gradient colors
    for i in range(len(theta) - 1):
        color = edge_cmap(i / (len(theta) - 1))
        ax2.plot(
            [x1[i], x1[i + 1]],
            [y1[i], y1[i + 1]],
            [upshifted_z[i], upshifted_z[i + 1]],
            color=color,
            linewidth=3
        )

    # Plot the normal vector for reference
    offset = np.array([0., 0., OFFSET_DIST])
    draw_arrow_3d(
        ax2,
        np.zeros((3,)),
        offset,
        OFFSET_ARROW_KWARGS
    )
    ax2.text(
        OFFSET_DIST / 3, 0.1, 0.8,
        'offset',
        fontsize=10,
        ha='left',
        color='black'
    )
    ax2.set_title('Shift by Offset', fontsize=11)

    # Plot 3
    ax3 = fig.add_subplot(133, projection='3d')

    # Center of the circle
    offset_coord = np.sqrt(OFFSET_DIST ** 2 / 3.)
    offset = np.array([offset_coord, offset_coord, offset_coord])

    # Normal vector (perpendicular direction)
    normal = offset
    normal = normal / np.linalg.norm(normal)  # normalize

    # Create two orthogonal vectors in the plane perpendicular to normal
    # First, find any vector perpendicular to normal
    if abs(normal[0]) < 0.9:
        v1 = np.cross(normal, [1, 0, 0])
    else:
        v1 = np.cross(normal, [0, 1, 0])
    v1 = v1 / np.linalg.norm(v1)

    # Second perpendicular vector
    v2 = np.cross(normal, v1)
    v2 = v2 / np.linalg.norm(v2)


    # Parametrize circle in the perpendicular plane
    x3 = offset[0] + RADIUS * (v1[0] * np.cos(theta) + v2[0] * np.sin(theta))
    y3 = offset[1] + RADIUS * (v1[1] * np.cos(theta) + v2[1] * np.sin(theta))
    z3 = offset[2] + RADIUS * (v1[2] * np.cos(theta) + v2[2] * np.sin(theta))

    # Plot the filled circle
    ax3.plot_trisurf(x3, y3, z3, color=FILL_COLOR, alpha=FILL_ALPHA, shade=SHADE)

    # Plot the border with gradient colors
    for i in range(len(theta) - 1):
        color = edge_cmap(i / (len(theta)-1))
        ax3.plot(
            [x3[i], x3[i + 1]],
            [y3[i], y3[i + 1]],
            [z3[i], z3[i + 1]],
            color=color,
            linewidth=3
        )

    draw_arrow_3d(
        ax3,
        np.zeros((3,)),
        offset,
        OFFSET_ARROW_KWARGS
    )
    ax3.text(
        0.7, 0., 0.5,
        'offset (rotated)',
        fontsize=10,
        ha='left',
        color='black'
    )

    ax3.set_title('Rotate: Now In Positive Orthant', fontsize=11)

    for ax in [ax1, ax2, ax3]:
        beautify_ax_3d(ax, Z=2)
        add_axis_lines_3d(ax, Z=Z)
        ax.view_init(elev=8, azim=-115, roll=0)

    plt.show()

def show_shifting_circle_CORRECT():
    fig = plt.figure(figsize=(14, 6))
    cmap = plt.get_cmap('coolwarm', 7)
    edge_cmap = plt.get_cmap('coolwarm')

    N_POINTS = 100
    RADIUS = 1.0
    Z = 3.0
    FILL_COLOR = cmap(3)
    FILL_ALPHA = 0.9
    SHADE = False
    OFFSET_DIST = 1.8
    OFFSET_ARROW_KWARGS = {
        'arrowstyle': '-|>',
        'mutation_scale': 10,
        'lw': 1.5,
        'color': 'black',
        'alpha': 1
    }

    theta = np.linspace(0, 2 * np.pi, N_POINTS)

    ax1 = fig.add_subplot(121, projection='3d')

    x1 = RADIUS * np.cos(theta)
    y1 = RADIUS * np.sin(theta)
    z1 = np.zeros_like(theta)

    # Plot the filled circle
    ax1.plot_trisurf(x1, y1, z1, color=FILL_COLOR, alpha=FILL_ALPHA, shade=SHADE)
    # Plot the border with gradient colors
    for i in range(len(theta) - 1):
        color = edge_cmap(i / (len(theta) - 1))
        ax1.plot(
            [x1[i], x1[i + 1]],
            [y1[i], y1[i + 1]],
            [z1[i], z1[i + 1]],
            color=color,
            linewidth=3
        )

    ax1.set_title('Pre-shift', fontsize=11)

    ax2 = fig.add_subplot(122, projection='3d')

    # Plot the up-shifted circle
    # Center of the circle
    offset_coord = np.sqrt(OFFSET_DIST ** 2 / 3.)
    offset = np.array([offset_coord, offset_coord, offset_coord])
    x2 = x1 + np.ones_like(x1) * offset_coord
    y2 = y1 + np.ones_like(y1) * offset_coord
    z2 = z1 + np.ones_like(z1) * offset_coord

    ax2.plot_trisurf(
        x2,
        y2,
        z2,
        color=FILL_COLOR,
        alpha=FILL_ALPHA,
        shade=SHADE
    )
    # Plot the border with gradient colors
    for i in range(len(theta) - 1):
        color = edge_cmap(i / (len(theta) - 1))
        ax2.plot(
            [x2[i], x2[i + 1]],
            [y2[i], y2[i + 1]],
            [z2[i], z2[i + 1]],
            color=color,
            linewidth=3
        )

    # Plot the normal vector for reference
    draw_arrow_3d(
        ax2,
        np.zeros((3,)),
        offset,
        OFFSET_ARROW_KWARGS
    )
    ax2.text(
        0.7, 0., 0.5,
        'offset',
        fontsize=10,
        ha='left',
        color='black'
    )
    ax2.set_title('Shifted; Now In Positive Orthant', fontsize=11)

    for ax in [ax1, ax2]:
        beautify_ax_3d(ax, Z=2)
        add_axis_lines_3d(ax, Z=Z)
        ax.view_init(elev=20, azim=-115, roll=0)

    plt.show()

def show_sine_waves():
    NUM_SAMPLES = 501
    K_VALUES = [4, 32, 43]

    x = np.linspace(0, 2 * np.pi, NUM_SAMPLES)
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    cmap = plt.get_cmap('coolwarm', 7)

    y_sum = np.zeros_like(x)
    for i, k in enumerate(K_VALUES):
        y = np.sin(k * x)
        y_sum += y
        ax.plot(x, y, color=cmap(1 + i * 2))
    ax.plot(x, y_sum, '--', color='darkslategrey')
    ax.set_facecolor(BACKGROUND_COLOR)
    plt.show()

if __name__ == '__main__':
    # do_curve_thingy_animation()

    # ALPHA_FREQS = [1./8, 4. / 43, 2.0, 8.0]
    # NUM_PERIODSS = [4.0, 43. / 4., 3.0, 8.0]

    # ALPHA_FREQS = [1.0, 8.0]
    # NUM_PERIODSS = [1.0, 8.0]

    ALPHA_FREQS = [0.25, 1./3, 1.0, 1.50,]
    NUM_PERIODSS = [2., 3., 2., 3.,]
    # do_family_of_curves(
    #     circle_freq = 0.5,
    #     alpha_freqs = ALPHA_FREQS,
    #     num_periodss = NUM_PERIODSS,
    #     do_animation = True,
    #     alpha_period_offsets = [0.0, 0.0, 0.0, 0.0],
    #     plot_scatter = False,
    #     num_trajectory_samples = 1000
    # )

    # show_shifting_circle()
    # show_shifting_circle_CORRECT()

    # show_sine_waves()