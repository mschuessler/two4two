"""Module for the baseline."""

from __future__ import annotations

import dataclasses
import pickle
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np


@dataclasses.dataclass
class BaselineInputData:
    """Data used to construct the baseline.

    Attributes:
        images: A numpy array of shape (N, C, H, W) containing the images.
        labels: A numpy array of shape (N,) containing the labels.
        logits: A numpy array of shape (N,) containing the logits.
        histogram_bins: A numpy array of shape (N,) containing the histogram bins.

    """

    images: np.ndarray
    logits: np.ndarray
    histogram_bins: np.ndarray
    labels: Optional[np.ndarray] = None

    @staticmethod
    def load(state: dict[str, Any]) -> BaselineInputData:
        """Loads a baseline input data from a state dict."""
        return BaselineInputData(
            state["images"],
            state["labels"],
            state["logits"],
            state["histogram_bins"],
        )

    def state_dict(self) -> dict[str, Any]:
        """Returns a state dict."""
        return {
            "images": self.images,
            "labels": self.labels,
            "logits": self.logits,
            "histogram_bins": self.histogram_bins,
        }

    def save(self, path: str):
        """Dumps the data to a pickle file."""
        with open(path, "wb") as f:
            pickle.dump(self.state_dict(), f)


def _sample_from_bins(
    logits: np.ndarray,
    bins: np.ndarray,
    logit_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    n_bins = len(bins) - 1
    indicies = []
    mask = np.zeros_like(logits, dtype=bool)
    for i in range(n_bins):
        mask = np.logical_or(
            mask, np.logical_and(bins[i] <= logits, logits <= bins[i + 1])
        )
        if logit_mask is not None:
            mask = np.logical_and(mask, logit_mask)

        indicies.append(
            np.random.choice(len(mask), 1, p=mask.astype(np.float64) / mask.sum())
        )

    return np.concatenate(indicies)


@dataclasses.dataclass
class BaselineGridItem:
    """A single grid item for the baseline.

    This corresponds to a single image, logit, and label.

    Attributes:
        image: A numpy array of shape (C, H, W) containing the image.
        image_idx: The index of the image in the original dataset (BaselineInput).
        logit: The logit value of the image.
        row_pos: The row position of the image in the grid.
        col_pos: The column position of the image in the grid.
        label: The label of the image.

    """

    image: np.ndarray
    image_idx: int
    logit: float
    row_pos: int
    col_pos: int
    label: Optional[float] = None


def _get_counterfactual_bins(
    logits: np.ndarray,
    logit_percentiles: list[float],
) -> tuple[np.ndarray, np.ndarray]:
    """Computes the bins and values for the counterfactual interpolations.

    Args:
        logits: The logits of the model.
        logit_percentiles: The percentiles to use for the bins.
            These must be symmetric

    Returns:
        A tuple of counterfacutal values and bins borders between them.
    """

    abs_logits = np.abs(logits)

    bins = np.array(logit_percentiles) - 50
    neg_bins = bins[bins < 0]
    pos_bins = bins[bins > 0]

    if not np.allclose(-neg_bins[::-1], pos_bins):
        raise ValueError(
            "Negative and positive logit bins are not equal. Must be symmetric. "
            f"Negative bins: {neg_bins}, positive bins: {pos_bins}"
        )

    abs_scales = np.percentile(abs_logits, pos_bins)

    logit_scales = np.concatenate([-abs_scales[::-1], np.array([0]), abs_scales])

    bins = (logit_scales[:-1] + logit_scales[1:]) / 2
    cf_values = logit_scales.astype(np.float32)

    cf_bins_w_inf = np.array([-np.inf] + bins.tolist() + [np.inf])
    return cf_values, cf_bins_w_inf


def sample_grid(
    data: BaselineInputData,
    n_rows: int = 30,
    seed: int = 0,
) -> list[BaselineGridItem]:
    """Samples a grid of images from the logits.

    Args:
        data: The baseline input data.
        n_rows: The number of rows in the grid.
        seed: The seed to use for sampling.

    Returns:
        A list of BaselineGridItem objects.
    """
    np.random.seed(seed)

    imgs_original = data.images
    labels_original = data.labels

    logits = data.logits

    logit_mask = np.ones(logits.shape)
    grid = []
    for row in range(0, n_rows):
        image_idx = _sample_from_bins(logits, data.histogram_bins, logit_mask)

        logit_mask[image_idx] = 0
        imgs_row = imgs_original[image_idx]
        if labels_original is not None:
            labels_row = labels_original[image_idx]
        else:
            labels_row = None

        sel_logits = logits[image_idx]

        for col in range(0, len(imgs_row)):
            grid.append(
                BaselineGridItem(
                    image=imgs_row[col],
                    image_idx=image_idx[col],
                    logit=sel_logits[col],
                    row_pos=row,
                    col_pos=col,
                    label=labels_row[col] if labels_row is not None else None,
                )
            )
    return grid


@dataclasses.dataclass
class BaselineResult:
    """The result of the baseline.

    Attributes:
        data: The baseline input data.
        fig_path: The path to the figure.
        grid_items: The items in the grid.
    """

    data: BaselineInputData
    fig_path: Optional[str]
    grid_items: list[BaselineGridItem]


def baseline(
    images: np.ndarray,
    logits: np.ndarray,
    labels: Optional[np.ndarray] = None,
    n_rows: int = 10,
    show: bool = False,
    bins: Optional[np.ndarray] = None,
    percentiles: list[float] = [20, 40, 60, 80],
    figure_path: Optional[str] = None,
    show_logits: bool = False,
) -> BaselineResult:
    """Runs the baseline.

    Args:
        images: The images to use for the baseline. Numpy array of shape (N, C, H, W).
        logits: The logits of the model. Numpy array of shape (N, ).
        labels: The labels of the images. Numpy array of shape (N, ).
        n_rows: The number of rows in the grid.
        show: Whether to display the plot.
        bins: The bins to use for the histogram. If None, the bins will be computed
            from the percentiles. The bins must include -inf and inf.
        percentiles: The percentiles to use for the bins. They must be symmetric
            around 50, e.g., `[20, 40, 60, 80]` is fine but `[10, 40, 80]` would
            be not okay.
        figure_path: The path to save the figure to. If None, the figure will not
            be saved.
        show_logits: Whether to display the logits in the plot.

    Returns:
        A BaselineResult object.
    """

    if len(logits.shape) != 1:
        raise ValueError("logits must be a 1D array")

    if bins is None:
        _, bins = _get_counterfactual_bins(logits, percentiles)

    if bins[0] != -np.inf or bins[-1] != np.inf:
        raise ValueError("bins must include -inf and inf")

    n_cols = len(bins) - 1
    data = BaselineInputData(images, logits, bins)
    grid_items = sample_grid(
        data,
        n_rows=n_rows,
        seed=0,
    )

    fig, axes = plt.subplots(
        nrows=n_rows,
        ncols=n_cols,
        figsize=(n_cols, n_rows * 1.1),
    )

    for item in grid_items:
        ax = axes[item.row_pos, item.col_pos]
        ax.imshow(item.image.transpose(1, 2, 0))
        ax.set_xticks([])
        ax.set_yticks([])
        if show_logits:
            ax.set_title(f"logit: {item.logit:.2f}")

    for ax in axes.flatten():
        [i.set_linewidth(0.3) for i in ax.spines.values()]

    fig.subplots_adjust(
        left=0.0,
        right=1.0,
        top=1.0,
        bottom=0.00,
        wspace=0.00,
        hspace=0.10,
    )

    fig.set_dpi(600)

    if figure_path is not None:
        fig.savefig(figure_path, bbox_inches="tight", dpi=600)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return BaselineResult(
        data=data,
        fig_path=figure_path,
        grid_items=grid_items,
    )
