from torchvision import transforms
from torchvision.datasets import ImageFolder
import os
import logging
from monai.transforms import *
from collections import Counter


class CustomImageFolder(ImageFolder):
    def __init__(self, root, image_size: int = 256, train: bool = True):
        root = os.path.join(root, 'train' if train else 'test')
        super().__init__(root)

        augmentations = [
            transforms.RandomApply(
                transforms=[transforms.ColorJitter(brightness=0.2, contrast=0.2)], p=0.5),
            transforms.RandomApply(transforms=[transforms.RandomAffine(
                degrees=10, scale=(0.9, 1.1))], p=0.5),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
        ]

        self.transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Resize((image_size, image_size)),
            *(augmentations if train else []),
            transforms.Normalize((0.28), (0.23))
        ])

        class_counter = Counter(self.targets)
        logging.info('Class distribution : {}'.format(class_counter))
