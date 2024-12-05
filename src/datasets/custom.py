from torchvision import transforms
from torchvision.datasets import ImageFolder
import os
import logging
from monai.transforms import *
from collections import Counter

from torchvision import transforms
from torchvision.datasets import ImageFolder
import os
import logging
from monai.transforms import *
from collections import Counter


class CustomImageFolder(ImageFolder):
    def __init__(self, root, image_size: int = 256, train: bool = True, augment_type: str = 'geometric'):
        root = os.path.join(root, 'train' if train else 'test')
        super().__init__(root)

        if augment_type not in ['geometric', 'photometric', 'all']:
            raise ValueError('augment_type must be of {}'.format(augment_type))

        geometric_augmentation = [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomApply(
                transforms=[transforms.RandomRotation(90)], p=0.3),
        ]

        photometric_augmentation = [
            RandAdjustContrast(prob=0.3, gamma=(1, 3)),
            RandGaussianSmooth(prob=0.3, sigma_x=(0.25, 0.8)),
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
            transforms.Normalize((0.28), (0.23))
        ])

        class_counter = Counter(self.targets)
        logging.info('Class distribution : {}'.format(class_counter))
