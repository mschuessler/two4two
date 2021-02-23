from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

from two4two import blender
from two4two.utils import splitStickyStretchy

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

params = [sampler.sample() for _ in range(30)]
len(params)
len(sticky_params)
len(stretchy_params)
enforce_equal_class_distribution = False
num_cols_per_class = 3
max(len(sticky_params), len(stretchy_params))
np.ceil(41 / 3)
num_rows
np.ceil(5).astype('int')

sticky_ax



def renderGrid(params, num_cols_per_class=3, enforce_equal_class_distribution=True):
    sticky_params, stretchy_params = splitStickyStretchy(params)

    if enforce_equal_class_distribution:
        num_rows = np.floor(min(len(sticky_params), len(stretchy_params)) / num_cols_per_class)
        num_samples_per_class = int(num_rows * num_cols_per_class)
        sticky_params, stretchy_params = splitStickyStretchy(params, num_samples_per_class)
    else:
        num_rows = np.ceil(max(len(sticky_params), len(stretchy_params)) / num_cols_per_class)

    num_rows = num_rows.astype('int')

    fig, ax = plt.subplots( nrows=num_rows, ncols=num_cols_per_class * 2)
    fig.subplots_adjust(wspace=0, hspace=0)
    sticky_ax = ax[:, :num_cols_per_class].flatten().tolist()[::-1]
    stretchy_ax = ax[:, num_cols_per_class:].flatten().tolist()[::-1]

    for (img, param) in blender.render(
            params=sticky_params + stretchy_params,
            chunk_size=num_cols_per_class,
            download_blender=True):
        ax1 = sticky_ax.pop() if param.obj_name == 'sticky' else stretchy_ax.pop()
        ax1.axis('off')
        ax1.set_aspect('equal')
        ax1.imshow(img)


renderGrid(params)



def showExamplesForSampler(sampler=scene_parameters.SampleSceneParameters, num_rows=10, num_cols_per_class=3):
    num_samples_per_class = num_rows * num_cols_per_class

    sampled_params = [sampler.sample() for _ in range(4 * num_samples_per_class)]
    selected_params = [p for p in sampled_params if p.obj_name == 'sticky'][:num_samples_per_class] + \
        [p for p in sampled_params if p.obj_name == 'stretchy'][:num_samples_per_class]

    # Creating a matrix of indicies for the subplot
    plot_index = np.arange(num_samples_per_class * 2).reshape(-1, num_cols_per_class * 2)
    # Left indicies are used for sticky
    sticky_index = plot_index[:, :num_cols_per_class]
    # Right indicies are used for stretchy
    stretchy_index = plot_index[:, num_cols_per_class:]
    # Convert to reverse order list to be usable as stack
    sticky_index = sticky_index.flatten().tolist()[::-1]
    stretchy_index = stretchy_index.flatten().tolist()[::-1]

    plt.figure(figsize=(num_cols_per_class * 2, num_rows), dpi=200)
    gs1 = gridspec.GridSpec(num_rows, num_cols_per_class * 2, wspace=0.005, hspace=0.005)
    # gs1.update(wspace=0.005, hspace=0.005)
    # plt.subplots(nrows=...)
    for (img, param) in blender.render(
            params=selected_params,
            chunk_size=num_cols_per_class,
            download_blender=True):

        index = sticky_index.pop() if param.obj_name == 'sticky' else stretchy_index.pop()
        ax1 = plt.subplot(gs1[index])
        ax1.axis('off')
        ax1.set_aspect('equal')
        ax1.imshow(img)


showExamplesForSampler(sampler)
