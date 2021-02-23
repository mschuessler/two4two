from two4two import blender
from two4two import scene_parameters
from matplotlib import pyplot as plt
import os
import numpy as np


sampler = scene_parameters.ColorBiasedSceneParameterSampler()

# #####
# sampled_params = [sampler.sample() for _ in range(5000)]
#
# np.array([p.obj_scalar for p in sampled_params if p.obj_name == 'sticky']).mean()
# np.array([p.obj_scalar for p in sampled_params if p.obj_name == 'stretchy']).mean()
#
# sampler.bg_color_map = 'viridis'
#
# num_rows = np.ceil(num_samples_per_class / num_cols_per_class)
# ######


def showExamplesForSampler(sampler=scene_parameters.SampleSceneParameters, num_rows=10, num_cols_per_class=3):
    num_samples_per_class = num_rows * num_cols_per_class

    sampled_params = [sampler.sample() for _ in range(4 * num_samples_per_class)]
    selected_params = [p for p in sampled_params if p.obj_name == 'sticky'][:num_samples_per_class] + \
        [p for p in sampled_params if p.obj_name == 'stretchy'][:num_samples_per_class]

    # Creating a matrix of indicies for the subplot
    plot_index = np.arange(num_samples_per_class * 2).reshape(-1, num_cols_per_class * 2) + 1
    # Left indicies are used for sticky
    sticky_index = plot_index[:, :num_cols_per_class]
    # Right indicies are used for stretchy
    stretchy_index = plot_index[:, num_cols_per_class:]
    # Convert to reverse order list to be usable as stack
    sticky_index = sticky_index.flatten().tolist()[::-1]
    stretchy_index = stretchy_index.flatten().tolist()[::-1]

    plt.figure(figsize=(num_rows, num_cols_per_class * 2), dpi=200)
    # plt.subplots(nrows=...)
    for (img, param) in blender.render(
            params=selected_params,
            chunk_size=num_cols_per_class,
            download_blender=True):

        index = sticky_index.pop() if param.obj_name == 'sticky' else stretchy_index.pop()
        plt.subplot(num_rows, num_cols_per_class * 2, index)
        plt.subplots_adjust(wspace=0, hspace=0)
        plt.axis('off')
        plt.imshow(img)

showExamplesForSampler(sampler)
