from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon as MplPolygon

from .grasp import Grasp


def draw_grasp(
    ax: plt.Axes,
    grasp: Grasp,
    colour: str = "lime",
    linewidth: float = 2.0,
    label: Optional[str] = None,
) -> None:
    corners = grasp.corners()
    poly = MplPolygon(corners, fill=False, edgecolor=colour, linewidth=linewidth, label=label)
    ax.add_patch(poly)
    for i, j in [(0, 1), (2, 3)]:
        ax.plot(
            [corners[i, 0], corners[j, 0]],
            [corners[i, 1], corners[j, 1]],
            color="red",
            linewidth=linewidth + 0.5,
        )


def plot_prediction(
    image: np.ndarray,
    prompt: str,
    pred_grasps: Sequence[Grasp],
    gt_grasps: Optional[Sequence[Grasp]] = None,
    save_path: Optional[Union[str, Path]] = None,
    show_maps: Optional[dict] = None,
) -> plt.Figure:
    has_maps = show_maps is not None
    ncols = 1 + (3 if has_maps else 0)
    fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 5))
    if ncols == 1:
        axes = [axes]

    ax = axes[0]
    ax.imshow(image)
    if gt_grasps:
        for g in gt_grasps:
            draw_grasp(ax, g, colour="cyan", linewidth=1.5)
    for i, g in enumerate(pred_grasps):
        draw_grasp(ax, g, colour="lime" if i == 0 else "yellow", linewidth=2.0)
    ax.set_title(f'"{prompt}"', fontsize=11)
    ax.axis("off")

    if has_maps:
        titles = ["Quality Q", "Angle (atan2)", "Width W"]
        keys = ["quality", "angle", "width"]
        for k, (key, title) in enumerate(zip(keys, titles)):
            arr = show_maps[key]
            im = axes[k + 1].imshow(arr, cmap="viridis" if key != "angle" else "twilight")
            axes[k + 1].set_title(title)
            axes[k + 1].axis("off")
            fig.colorbar(im, ax=axes[k + 1], fraction=0.046)

    fig.tight_layout()
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def make_grid_figure(
    images: List[np.ndarray],
    prompts: List[str],
    pred_grasps_list: List[List[Grasp]],
    gt_grasps_list: List[List[Grasp]],
    ncols: int = 4,
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    n = len(images)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.5 * ncols, 3.5 * nrows))
    axes = np.atleast_2d(axes)

    for k in range(nrows * ncols):
        ax = axes[k // ncols, k % ncols]
        if k >= n:
            ax.axis("off")
            continue
        ax.imshow(images[k])
        for g in gt_grasps_list[k]:
            draw_grasp(ax, g, colour="cyan", linewidth=1.0)
        if pred_grasps_list[k]:
            draw_grasp(ax, pred_grasps_list[k][0], colour="lime", linewidth=2.0)
        ax.set_title(prompts[k], fontsize=9)
        ax.axis("off")

    fig.tight_layout()
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig
