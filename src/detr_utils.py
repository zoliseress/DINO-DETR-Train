
"""
Utility functions for working with DETR components, such as visualizing position
embeddings. These functions are not part of the core training loop but can be helpful
for understanding and debugging the model.
"""

import math
import torch
import numpy as np

from detr import PositionEmbeddingSine2D


def visualize_position_embedding(
        pos_module: PositionEmbeddingSine2D,
        H: int,
        W: int,
        device: str = 'cpu',
        channels: list = None,
        ncols: int = 8,
        figsize: tuple = (12, 6),
        cmap: str = 'viridis',
        show_pairs: bool = True,
        save_path: str = None
    ):
    """
    Visualize outputs of a `PositionEmbeddingSine2D` module.

    Parameters
    ----------
    pos_module: PositionEmbeddingSine2D
        Instance of `PositionEmbeddingSine2D` (or any module that accepts x
        and returns shape [B, Cpos, H, W]).
    H: int
        Spatial height for the position embedding.
    W: int
        Spatial width for the position embedding.
    device: str
        'cpu' or 'cuda'.
    channels: list/iterable
        List/iterable of channel indices to plot. If None, select up to first
        16 channels.
    ncols: int
        Number of columns in the plot grid.
    figsize: tuple
        Figure size.
    cmap: str
        Matplotlib colormap.
    show_pairs: bool
        If True and channels are full-range, will try to visualize sin/cos
        pairs side-by-side (assumes channel ordering [pos_y..., pos_x...]).
    save_path: str
        Optional path to save the figure (PNG).

    Notes:
      1. Channels alternate across frequency bands (sin/cos) and the first half
      represent the Y embedding, second half the X embedding (like the DETR-style
      implementation).
      2. Visualizing full `cpos` in pair mode is handy to inspect corresponding
      sin/cos pairs.

    Returns:
      (fig, axes) matplotlib objects.
    """

    import matplotlib.pyplot as plt

    pos_module = pos_module.to(device)
    pos_module.eval()

    # create a dummy input: only shape matters (B, C, H, W)
    dummy = torch.zeros(1, 1, H, W, device=device)
    with torch.no_grad():
        pos = pos_module(dummy)  # [1, cpos, H, W]
    pos = pos[0].cpu().numpy()  # [cpos, H, W]

    cpos = pos.shape[0]

    # Default channels: first min(16, cpos).
    if channels is None:
        max_show = min(16, cpos)
        channels = list(range(max_show))

    # Build pairs, if requested and full channel set is available.
    if show_pairs and (len(channels) == cpos):
        # Assume ordering pos = [pos_y_channels..., pos_x_channels...].
        half = cpos // 2
        pairs = [(i, i + half) for i in range(half)]
        # Flatten pair indices but keep pair information.
        channels_to_plot = pairs
        pair_mode = True
    else:
        channels_to_plot = channels
        pair_mode = False

    # Setup grid.
    if pair_mode:
        n_plots = len(channels_to_plot) * 2
    else:
        n_plots = len(channels_to_plot)
    ncols = min(ncols, n_plots) if n_plots > 0 else 1
    nrows = math.ceil(n_plots / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = np.array(axes).reshape(-1)

    # Plotting.
    for i, ch in enumerate(channels_to_plot):
        if pair_mode:
            a_idx = 2 * i
            idx_y, idx_x = ch
            arr_y = pos[idx_y]
            arr_x = pos[idx_x]
            # Normalize each channel for visualization.
            arr_y = (arr_y - arr_y.min()) / (arr_y.max() - arr_y.min() + 1e-9)
            arr_x = (arr_x - arr_x.min()) / (arr_x.max() - arr_x.min() + 1e-9)
            axes[a_idx].imshow(arr_y, cmap=cmap)
            axes[a_idx].set_title(f"chan {idx_y} (y)")
            axes[a_idx].axis('off')
            axes[a_idx + 1].imshow(arr_x, cmap=cmap)
            axes[a_idx + 1].set_title(f"chan {idx_x} (x)")
            axes[a_idx + 1].axis('off')
        else:
            arr = pos[ch]
            arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-9)
            axes[i].imshow(arr, cmap=cmap)
            axes[i].set_title(f"chan {ch}")
            axes[i].axis('off')

    # Hide remaining axes.
    for ax in axes[n_plots:]:
        ax.axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.show()
    return fig, axes


if __name__ == "__main__":

    # Create position embedding module.
    # hidden_dim = 256 -> num_pos_feats = hidden_dim//2 = 128 -> pos = 256
    pos = PositionEmbeddingSine2D(num_pos_feats=128)

    # Visualize first 16 channels on a 32x32 grid.
    visualize_position_embedding(
        pos_module=pos,
        H=32,
        W=32,
        device="cpu",
        channels=list(range(16)),
        ncols=8,
    )
    