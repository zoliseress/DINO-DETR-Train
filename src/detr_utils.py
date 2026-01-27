
import math
import torch
import numpy as np
import matplotlib.pyplot as plt

from detr import PositionEmbeddingSine2D


def visualize_position_embedding(
        pos_module,
        H,
        W,
        device='cpu',
        channels=None,
        ncols=8,
        figsize=(12, 6),
        cmap='viridis',
        show_pairs=True,
        save_path=None
    ):
    """
    Visualize outputs of a `PositionEmbeddingSine2D` module.

    Args:
      pos_module: instance of `PositionEmbeddingSine2D` (or any module that accepts x
                  and returns shape [B, Cpos, H, W]).
      H, W: spatial height and width to create a dummy input (int).
      device: 'cpu' or 'cuda'.
      channels: list/iterable of channel indices to plot. If None, select up to first
                16 channels.
      ncols: number of columns in the plot grid.
      figsize: figure size.
      cmap: matplotlib colormap.
      show_pairs: if True and channels are full-range, will try to visualize sin/cos
                  pairs side-by-side (assumes channel ordering [pos_y..., pos_x...]).
      save_path: optional path to save the figure (PNG).

    Notes:
      1. Channels alternate across frequency bands (sin/cos) and the first half represent
      the Y embedding, second half the X embedding (per the DETR-style implementation).
      2. Visualizing full Cpos in pair mode is handy to inspect corresponding sin/cos pairs.

    Returns:
      (fig, axes) matplotlib objects.
    """
    pos_module = pos_module.to(device)
    pos_module.eval()

    # create a dummy input: only shape matters (B, C, H, W)
    dummy = torch.zeros(1, 1, H, W, device=device)
    with torch.no_grad():
        pos = pos_module(dummy)  # [1, Cpos, H, W]
    pos = pos[0].cpu().numpy()  # [Cpos, H, W]

    Cpos = pos.shape[0]

    # default channels: first min(16, Cpos)
    if channels is None:
        max_show = min(16, Cpos)
        channels = list(range(max_show))

    # If user requested to show pairs and full channel set is available, build pairs
    if show_pairs and (len(channels) == Cpos):
        # assume ordering pos = [pos_y_channels..., pos_x_channels...]
        half = Cpos // 2
        pairs = [(i, i + half) for i in range(half)]
        # flatten pair indices but keep pair information
        channels_to_plot = pairs
        pair_mode = True
    else:
        channels_to_plot = channels
        pair_mode = False

    # Setup grid
    if pair_mode:
        n_plots = len(channels_to_plot) * 2
    else:
        n_plots = len(channels_to_plot)
    ncols = min(ncols, n_plots) if n_plots > 0 else 1
    nrows = math.ceil(n_plots / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = np.array(axes).reshape(-1)

    # plot
    for i, ch in enumerate(channels_to_plot):
        if pair_mode:
            a_idx = 2 * i
            idx_y, idx_x = ch
            arr_y = pos[idx_y]
            arr_x = pos[idx_x]
            # normalize each channel for visualization
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

    # hide remaining axes
    for ax in axes[n_plots:]:
        ax.axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.show()
    return fig, axes


if __name__ == "__main__":

    # create module (hidden_dim = 256 -> num_pos_feats = hidden_dim//2 = 128 -> Cpos = 256)
    pos = PositionEmbeddingSine2D(num_pos_feats=128)

    # print(torch.cuda.device_count())
    # print(torch.cuda.get_device_name(0))
    # print(torch.cuda.get_device_name(1))
    # print(torch.cuda.current_device())

    # visualize first 16 channels on a 32x32 grid
    visualize_position_embedding(
        pos_module=pos,
        H=32,
        W=32,
        device="cpu",
        channels=list(range(16)),
        ncols=8,
    )
    