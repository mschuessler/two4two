"""visulization function for rendered scenes."""

import math
from typing import Sequence, Tuple

import matplotlib as mlp
import matplotlib.pyplot as plt


from two4two import blender
from two4two import scene_parameters


def render_grid(
    params: Sequence[scene_parameters.SceneParameters],
    num_cols_per_class: int = 3,
    equal_class_distribution: bool = True
) -> Tuple[mlp.figure.Figure, Sequence[Sequence[mlp.axes.Axes]]]:
    """Renders scene from param file and displays the in an image grid.

    Attrs:
        params: List of parameters files describing SceneParameterSample
        num_cols_per_class: Number of coloums of the grid reserved for a single class
        equal_class_distribution: Enforce equal class distribution in visulization.

    """
    sticky_params, stretchy_params = scene_parameters.split_sticky_stretchy(params)

    if equal_class_distribution:
        number_equal_samples = min(len(sticky_params), len(stretchy_params))
        num_rows = int(math.floor(number_equal_samples / num_cols_per_class))
        num_samples_per_class = int(num_rows * num_cols_per_class)
        sticky_params, stretchy_params = scene_parameters.split_sticky_stretchy(
            params, num_samples_per_class)
    else:
        max_number_samples = max(len(sticky_params), len(stretchy_params))
        num_rows = int(math.ceil(max_number_samples / num_cols_per_class))

    fig, ax = plt.subplots(nrows=num_rows,
                           ncols=num_cols_per_class * 2,
                           figsize=(20., 20. * num_rows / (num_cols_per_class * 2)))

    sticky_ax = ax[:, :num_cols_per_class].flatten().tolist()[::-1]
    stretchy_ax = ax[:, num_cols_per_class:].flatten().tolist()[::-1]

    for (img, mask, param) in blender.render(
            params=sticky_params + stretchy_params,
            chunk_size=num_cols_per_class,
            download_blender=True):
        ax1 = sticky_ax.pop() if param.obj_name == 'sticky' else stretchy_ax.pop()
        ax1.axis('off')
        ax1.set_aspect('equal')
        ax1.imshow(img)

    [ax.axis('off') for ax in stretchy_ax + sticky_ax]
    fig.subplots_adjust(wspace=0, hspace=0)
    return fig, ax
