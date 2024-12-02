from torchvision import transforms
from torchvision.datasets import ImageFolder
import os
from monai.transforms import *


class CustomImageFolder(ImageFolder):
    def __init__(self, root, image_size: int = 256, train: bool = True):
        root = os.path.join(root, 'train' if train else 'test')
        super().__init__(root)
        self.transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Resize((image_size, image_size)),
        ])
