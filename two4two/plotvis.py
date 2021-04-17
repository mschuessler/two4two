"""visulization function for rendered scenes."""

import math
from typing import List, Sequence, Tuple

import matplotlib as mlp
import matplotlib.pyplot as plt


from two4two import blender
from two4two import scene_parameters


def render_grid(
    params: List[scene_parameters.SceneParameters],
    num_cols_per_class: int = 3,
    equal_class_distribution: bool = True,
    # see #75 download_blender: bool = False
) -> Tuple[mlp.figure.Figure, Sequence[Sequence[mlp.axes.Axes]]]:
    """Renders scene from a list of SceneParameters and displays the in an image grid.

    Attrs:
        params: List of SceneParameters
        num_cols_per_class: Number of colums of the grid reserved for a single class
        equal_class_distribution: Enforce equal class distribution in visulization.

    """
    peaky_params, stretchy_params = scene_parameters.split_peaky_stretchy(params)

    if equal_class_distribution:
        number_equal_samples = min(len(peaky_params), len(stretchy_params))
        num_rows = int(math.floor(number_equal_samples / num_cols_per_class))
        num_samples_per_class = int(num_rows * num_cols_per_class)
        peaky_params, stretchy_params = scene_parameters.split_peaky_stretchy(
            params, num_samples_per_class)
    else:
        max_number_samples = max(len(peaky_params), len(stretchy_params))
        num_rows = int(math.ceil(max_number_samples / num_cols_per_class))

    fig, ax = plt.subplots(nrows=num_rows,
                           ncols=num_cols_per_class * 2,
                           figsize=(20., 20. * num_rows / (num_cols_per_class * 2)))

    peaky_ax = ax[:, :num_cols_per_class].flatten().tolist()[::-1]
    stretchy_ax = ax[:, num_cols_per_class:].flatten().tolist()[::-1]

    for (img, mask, param) in blender.render(
            params=peaky_params + stretchy_params,
            chunk_size=num_cols_per_class,
            download_blender=True):  # download_blender is true until #75 is fixed
        ax1 = peaky_ax.pop() if param.obj_name == 'peaky' else stretchy_ax.pop()
        ax1.axis('off')
        ax1.set_aspect('equal')
        ax1.imshow(img)

    [ax.axis('off') for ax in stretchy_ax + peaky_ax]
    fig.subplots_adjust(wspace=0, hspace=0)
    return fig, ax


def render_single_param(param: scene_parameters.SceneParameters):
    """Renders the image from the given ``SceneParameters`` and plots it.

    Attrs:
        param: SceneParameters of the image
    """
    (img, mask) = blender.render_single(param)
    plt.imshow(img)
    plt.axis('off')
