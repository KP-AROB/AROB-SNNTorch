from torchvision import transforms
from torchvision.datasets import ImageFolder
import os
import logging
from monai.transforms import *
from collections import Counter
from torchvision.transforms.v2 import GaussianNoise
from torchvision import transforms
from torchvision.datasets import ImageFolder
import os
import logging
from monai.transforms import *
from collections import Counter
import torch


class AddGaussianNoise(object):
    def __init__(self, mean=0.0, std=1.0):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        noise = torch.randn(tensor.size()) * self.std + self.mean
        return tensor + noise

    def __repr__(self):
        return f"{self.__class__.__name__}(mean={self.mean}, std={self.std})"


class CustomImageFolder(ImageFolder):
    def __init__(self, root, image_size: int = 224, train: bool = True, augment_type: str = None, noise: bool = False):
        root = os.path.join(root, 'train' if train else 'test')
        super().__init__(root)

        if augment_type not in ['geometric', 'photometric', 'all', None]:
            raise ValueError('augment_type must be of {}'.format(augment_type))

        geometric_augmentation = [
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomApply(
                transforms=[transforms.RandomRotation(45)], p=0.3),
        ]

        photometric_augmentation = [
            RandAdjustContrast(prob=0.3, gamma=(1, 2)),
            RandGaussianSmooth(prob=0.3, sigma_x=(0.25, 0.5)),
        ]

        self.transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Resize((image_size, image_size)),
            *(geometric_augmentation +
              photometric_augmentation if train and augment_type == 'all' else []),
            *(geometric_augmentation if train and augment_type == 'geometric' else []),
            *(photometric_augmentation if train and augment_type ==
              'photometric' else []),
        ])

        class_counter = Counter(self.targets)
        logging.info('Class distribution : {}'.format(class_counter))
