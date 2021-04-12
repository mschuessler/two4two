"""Dataloader for PyTorch."""

import dataclasses
import json
import os
from typing import Any, Callable, Dict, Sequence, Tuple

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

import two4two
from two4two import scene_parameters
from two4two import utils
import two4two.blender


def filter_attributes(param: two4two.SceneParameters) -> Dict[str, Any]:
    """Transforms a SceneParameters object to a dict.

    Filter the ``*_rbga``, ``resolution`` and ``_attributes_status`` attributes.
    """
    state = dataclasses.asdict(param)
    del state['obj_color_rgba']
    del state['bg_color_rgba']
    del state['resolution']
    del state['_attributes_status']
    return state


def all_attributes(param: two4two.SceneParameters) -> Dict[str, Any]:
    """Transforms a SceneParameters object to a dict.

    The attribute status is encoded in the members
    ``"attributes_status_{attr_name}"``, for example
    ``attribute_status_obj_name`` for the object name.
    """
    state = dataclasses.asdict(param)
    for attr, attr_state in state['_attributes_status'].items():
        state[f'attribute_status_{attr}'] = attr_state
    del state['_attributes_status']
    return state


class Two4Two(Dataset):
    """Pytorch Dataset for Two4Two.

    The dataset can return all attributes of SceneParameters. You can configure
    which attributes to return by the ``return_attributes`` constructor
    argument. Once created, you can also change it using the
    ``set_return_attributes`` method.

    You can inspect the name of each dimension of the returned label array by
    using the ``get_label_names`` method.

    For example:

    ```
        dset = Two4Two(
            my_root_dir, 'train',
            return_attributes=['obj_name', 'bending'])
        img, mask, label = dset[0]
        len(label)
        # -> 8

        dset.get_label_names()
        # -> ['obj_name', 'bending']

        dset.set_return_attributes(['obj_name', 'bg_color'])

        dset.get_label_names()
        # -> ['obj_name', 'bg_color']
    ```

    Args:
        root_dir: Dataset root directory
        split: Name of the split. The images are expected to exists in ``{root_dir}/{split}.``.
        transform: Any transformations to apply to the image.
        return_attributes: List of attributes of SceneParameters to return as
            labels. See ``get_label_names`` for the name of each dimension.
        return_segmentation_mask: should the segmentation mask be returned
            (default: ``True``).
    """

    def __init__(self,
                 root_dir: str,
                 split: str,
                 transform: Any = transforms.ToTensor(),
                 return_attributes: Sequence[str] = ['obj_name'],
                 return_segmentation_mask: bool = True,
                 ):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.params = []
        self._return_attributes = return_attributes
        self._return_segmentation_mask = return_segmentation_mask
        param_fname = os.path.join(self.root_dir, self.split, "parameters.jsonl")
        with open(param_fname) as f:
            for line in f.readlines():
                state = json.loads(line)
                self.params.append(two4two.SceneParameters.load(state))

    def set_return_attributes(self, labels: Sequence[str]):
        """Set the labels to return."""
        self._return_attributes = labels

    def get_dataframe(
        self,
        to_dict: Callable[[two4two.SceneParameters],
                          Dict[str, Any]] = filter_attributes
    ) -> pd.DataFrame:
        """Returns a pandas dataframe of all labels.

        Args:
            to_dict: Transforms a SceneParameters object to a dict. Default is
                to filter the ``*_rbga``, ``resolution`` and ``_attributes_status``
                attributes. Use the function ``all_attributes`` to return all
                attributes.
        """
        return pd.DataFrame(map(to_dict, self.params))

    def get_label_names(self) -> Sequence[str]:
        """Returns name of each label returned by ``self[idx]``."""
        label_names, _ = self._scene_parameters_to_flat_array(self.params[0])
        return label_names

    def _scene_parameters_to_flat_array(
        self,
        params: two4two.SceneParameters,
    ) -> Tuple[Sequence[str], torch.Tensor]:
        """Returns a list of label names and numpy array with the labels.

        The class exposes all attributes of SceneParameters.  This function
        concatenates all attributes selected by ``set_return_attributes`` to a 1d array.
        Any attributes that are tuples such as will be also be appended.

        Additonally, the function also keeps track of the name of each dimension.

        """
        arr = []
        label_names = []
        for attr_name in self._return_attributes:
            attr_value = getattr(params, attr_name)
            if attr_name == 'obj_name':
                label_names.append(attr_name)
                arr.append(params.obj_name_as_int)
            elif utils.supports_iteration(attr_value):
                for i, item in enumerate(attr_value):
                    label_names.append(f'{attr_name}_{i}')
                    arr.append(item)
            else:
                label_names.append(attr_name)
                arr.append(attr_value)
        return label_names, torch.FloatTensor(arr)

    def int_to_obj_name(self, label_int: int) -> str:
        """Returns the ``obj_name`` encoded by the integer."""
        return {
            idx: name
            for name, idx in scene_parameters.OBJ_NAME_TO_INT.items()
        }[label_int]

    def segmentation_int_to_label(self, index: int) -> str:
        """Return the label of the segmentation mask."""
        return two4two.blender.SEGMENTATION_INT_TO_NAME[index]

    def segmentation_label_to_int(self, label: str) -> int:
        """Return the index of the segmentation label."""
        return two4two.blender.SEGMENTATION_NAME_TO_INT[label]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        param = self.params[idx]
        fname = param.filename
        img = Image.open(os.path.join(self.root_dir, self.split, fname))
        img = img.convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        _, label_arr = self._scene_parameters_to_flat_array(param)

        if self._return_segmentation_mask:
            mask_img = Image.open(os.path.join(self.root_dir, self.split, param.mask_filename))
            mask = torch.from_numpy(np.array(mask_img)[np.newaxis])
            return img, mask, label_arr
        else:
            return img, label_arr

    def __len__(self) -> int:
        return len(self.params)
