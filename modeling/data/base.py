import os
import glob
import torch
import random

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import transforms


def build_transforms(cfg):
    return transforms.Compose(transforms.ToTensor())


class BaseDataset(Dataset):
    """
    This class is the base of all image dataset
    Every subclass implement from this class should
    rewrite the following functions.
    __init__()         initialize the class
    __Len__()          return the size of dataset
    __getitem__()      get the image by index
    """
    IMG_EXTENSIONS = [".jpg", ".jpeg", ".png"]

    def __init__(self, cfg):
        super(BaseDataset, self).__init__()
        self.root_path = cfg.root_path

    def __len__(self):
        raise NotImplementedError("Must rewrite the function __len__")

    def __getitem__(self, item):
        raise NotImplementedError("Must rewrite the function __getitem__")

    def _check_image_format(self, image_path):
        return any(image_path.endswith(ext for ext in self.IMG_EXTENSIONS))


class ImageDataset(BaseDataset):
    def __init__(self, root_path):
        super(ImageDataset, self).__init__()
        self.image_dataset = self.build_dataset(root_path)
        if not len(self.image_dataset):
            raise RuntimeError("Found no image in the file '{}'".format(self.root_path))
        self.root_path = root_path
        self.transform = build_transforms()

    def __getitem__(self, index: int):
        image_path = self.image_dataset[index]
        img = Image.open(image_path)
        if self.transform is not None:
            img = self.transform(img)
        else:
            img = transforms.ToTensor(img)
        return img

    def __len__(self):
        return len(self.image_dataset)

    def build_dataset(self):
        images_dataset = []
        assert os.path.isdir(self.root_path), "'{}' is not a valid directory"
        for image_file in os.listdir(self.root_path):
            if self._check_image_format(image_file):
                image_abs_path = os.path.join(self.root_path, image_file)
                images_dataset.append(image_abs_path)
        return images_dataset
