# Two4Two: Evaluating Interpretable Machine Learning -- A Synthetic Dataset For Controlled Experiments
![example_images_with_color_bias](examples/images/colorbiased_examples.png)

## Introduction
Two4Two is a library to create synthetic image data crafted for human evaluations of interpretable ML approaches (esp. image classification).
The synthetic images show two abstract animals: **Peaky** (arms inwards) and **Stretchy** (arms outwards). They are similar-looking, abstract animals, made of eight blocks.

![peaky_and_strecthy](examples/images/peaky_stretchy.png)

These animals are simple enough to be used in instructions for human-subject evaluations on crowd-sourcing platforms. We also provide segmentation masks so they can be used for algorithmic evaluations as well. The core functionality of this library is that one can correlate different parameters with an animal type to create bias in the data. The changeable parameters are:
- Arm position – the main feature differentiating Peaky (inward) and Stretchy (outwards)
- Color of the animal and color of the background
- Shape of the blocks – continuously varying from
spherical to cubic
- Deformation/Bending of the animal
- Rotation (3 DoF) and Position (2 DoF) of the animal

![variation](examples/images/variations.png)


This repository contains the source code that can be used to create custom biases and links to ready-to-download datasets that may already be sufficient for some experiments.

We created this library because we see the choice of dataset and user study scenario as a major obstacle to human subject evaluations.
If you find this dataset helpful, please cite our paper:
```
@inproceedings{
    sixt2022do,
    title={Do Users Benefit From Interpretable Vision? A User Study, Baseline, And Dataset},
    author={Leon Sixt and Martin Schuessler and Oana-Iuliana Popescu and Philipp Wei{\ss} and Tim Landgraf},
    booktitle={International Conference on Learning Representations},
    year={2022},
    url={https://openreview.net/forum?id=v6s3HVjPerv}
}
```

## Pregenerated datasets and models
Before you generate your own data, consider using our pregenerated datasets. You do not need to install this software to use this data.

The following package contains the biased dataset which we used in our own study. It has one easy to spot bias and one that is difficult to spot (for details please refer to the paper)
- **[https://f002.backblazeb2.com/file/iclr2022/two4two_obj_color_and_spherical_finer_search_spherical_uniform_0.33_uniform_0.15.tar](https://f002.backblazeb2.com/file/iclr2022/two4two_obj_color_and_spherical_finer_search_spherical_uniform_0.33_uniform_0.15.tar)** (3.3 GB)

If you want the same dataset without any biases use this download:
- **[https://f002.backblazeb2.com/file/iclr2022/two4two_obj_color_and_spherical_finer_search_spherical_uniform_0.33_uniform_0.15_unbiased.tar](https://f002.backblazeb2.com/file/iclr2022/two4two_obj_color_and_spherical_finer_search_spherical_uniform_0.33_uniform_0.15_unbiased.tar)** (1.85 GB)

If you would like to reproduce our results consider using our invertible neural network model:
- **[https://f002.backblazeb2.com/file/iclr2022/do_users_benefit_from_interpretable_vision_model.tar.gz](https://f002.backblazeb2.com/file/iclr2022/do_users_benefit_from_interpretable_vision_model.tar.gz)** (2.66 GB)

Advanced: To obtain ground truth feature importance please refer to the following repository:
- **[https://github.com/berleon/do_users_benefit_from_interpretable_vision](https://github.com/berleon/do_users_benefit_from_interpretable_vision)** (Link)

The following package contains training data for 4 datasets, one with no bias, one with a bias on the object color, one with a bias on spherical, and one with the two baises combined.
- **[https://f001.backblazeb2.com/file/two4two/datasets_models/golden240k.tar.gz](https://f001.backblazeb2.com/file/two4two/datasets_models/golden240k.tar.gz)** (7.6 GB)

If you only want the dataset with the two biases the following smaller package is sufficient:
- **[https://f001.backblazeb2.com/file/two4two/datasets_models/golden80k.tar.gz](https://f001.backblazeb2.com/file/two4two/datasets_models/golden80k.tar.gz)** (2.2 GB)


We provide a [Colab Notebook](https://colab.research.google.com/drive/1-_sp1_eCc1ToeTQRxrXxGzaW-FLbGHxN?usp=sharing) that illustrates how you can use this dataset to **train a simple model that has two biases**.

If you would like to understand how this dataset was generated have a look at [the config that was used to generate it](config/color_spher_bias.toml) and *[bias.py](two4two/bias.py)*.

## Installation
If you want to generate your own data, follow these instructions.
Currently, this project is not available through pip but has to be installed manually.

Download this repository:

```git
git clone https://github.com/mschuessler/two4two.git

```

We suggest creating a python3 or conda environment instead of using your system python.

```
python3 -m venv ~/242_enviroment
source ~/242_enviroment/bin/activate
```

To install the **minimal installation** two4two package change into the cloned directory and run setuptools.

```
cd two4two
pip install .
```

To install the **installation including all requirements for generating your own training data** run:
```
pip install .[example_notebooks_data_generation]
```

## Training Models on two4two
For training your own models, you have two choices:
1) *No GPU required and installation free*: Run our example notebook inside of [Colab](https://colab.research.google.com/drive/1-_sp1_eCc1ToeTQRxrXxGzaW-FLbGHxN?usp=sharing) (this will download pregenerated datasets) - you can also run this notebook on your own machine you can find the notebook in [examples/train_lenet_colab.ipynb](examples/train_lenet_colab.ipynb)
2) Install the installation, including all requirements for generating your own training data and training your own models**. This will install TensorFlow, and we recommend having a GPU available. After that you can user our [training script](examples/train_lenet.py) (we also have an [untested ResNet implementation](examples/train_resnet.py))
```
pip install .[example_notebooks_model_training]
wget https://f001.backblazeb2.com/file/two4two/datasets_models/golden240k.tar.gz
tar -xf golden240k.tar.gz
python two4two/examples/train_lenet.py spherical_color_bias
python two4two/examples/train_lenet.py color_bias
python two4two/examples/train_lenet.py spherical_bias
python two4two/examples/train_lenet.py no_bias
```

## Generate your own dataset
The jupyter-notebook **[examples/GenerateData.ipynb](examples/GenerateData.ipynb)** provides extensive details on how you can create your own *custom* data.
There are countless options to add biases to your custom dataset. The notebook is a great place to get started.
More code-savy folks may want got get started with *[bias.py](two4two/bias.py)* directly.

There is also a comand line script:
```
    two4two_render_dataset config/color_spher_bias.toml
```
The comand line script has different options:
```
  -h, --help            show this help message and exit
  --download_blender    Download blender if not found.
  --blender_dir BLENDER_DIR
                        Download blender to this directory.
  --split_by SPLIT_BY   Divide number of samples (usefull for distributed sampling).
  --keep-only-tar       Remove output dir and only keep `.tar` file.
  --skip-xgb            Run xgb model.
  --debug               Print debug information and enter pdb on error.
```

An example for a config file can be found at [config/color_spher_bias.toml](config/color_spher_bias.toml).
It allows to specify your sampler class, how many samples to generate and 
also to run simple interventions.


## PyTorch data loader

We provide a ready to use PyTorch dataloader.

```python

from two4two.pytorch import Two4Two


train_set = Two4Two(
    "path/to/dataset/dir/",
    split="train",
    return_attributes=['obj_name', 'obj_color']
)
train_loader = Dataloader(train_set, batch_size=30)

img, mask, label = train_set[0]

# returns name to the label dimensions
train_set.get_label_names()

# get all labels as pandas dataframe
df = train_set.get_dataframe()
```

### Funding
Funded by the GermanFederal Ministry of Education and Research(BMBF) - NR 16DII113.
