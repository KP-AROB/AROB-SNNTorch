import torch
import os
import logging
from torchvision import transforms
from src.utils.parameters import instanciate_cls
from torch.utils.data import DataLoader, random_split
from monai.transforms import *
from torchvision.datasets import ImageFolder
from collections import Counter
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
import numpy as np


def load_dataloader(dataset_name: str, dataset_params: dict, useGPU: bool = True):
    image_size = dataset_params['image_size']
    batch_size = dataset_params['batch_size']
    channels = dataset_params['channels']
    mean, std = dataset_params['mean'], dataset_params['std']
    use_augmentations = dataset_params.get('use_augmentations', True)

    n_workers = 4 * torch.cuda.device_count() if useGPU else 2

    train_augmentations = [
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomHorizontalFlip(p=0.3),
        RandRotate90(prob=0.3, max_k=3),
    ] if use_augmentations else []

    base_transforms = [
        *([transforms.Grayscale()] if channels == 1 else []),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor()
    ]

    normalize_transform = [transforms.Normalize((mean,) if channels == 1 else (mean, mean, mean),
                                                (std,) if channels == 1 else (std, std, std))]

    # Compose transforms for each phase
    train_transforms = transforms.Compose(
        base_transforms + train_augmentations + normalize_transform)
    val_test_transforms = transforms.Compose(
        base_transforms + normalize_transform)

    # Helper function for dataset instantiation
    def get_dataset(name, root, is_train):
        if name == "ImageFolder":
            return ImageFolder(root=root, transform=train_transforms if is_train else val_test_transforms)
        return instanciate_cls("torchvision.datasets", name, {
            'root': root, "train": is_train, "download": True, "transform": train_transforms if is_train else val_test_transforms
        })

    # Load datasets
    if dataset_name == "ImageFolder":
        train_dataset = get_dataset(dataset_name, os.path.join(
            dataset_params['data_dir'], 'train'), True)
        test_dataset = get_dataset(dataset_name, os.path.join(
            dataset_params['data_dir'], 'test'), False)
    else:
        train_dataset = get_dataset(
            dataset_name, dataset_params['data_dir'], True)
        test_dataset = get_dataset(
            dataset_name, dataset_params['data_dir'], False)

    if hasattr(train_dataset, 'class_to_idx'):
        logging.info(
            f"Available labels in dataset: {train_dataset.class_to_idx}")

    # training_label_counts = Counter(train_dataset.targets)
    # logging.info('Training class balance : {}'.format(training_label_counts))
    # total_samples = sum(training_label_counts.values())
    # class_weights = torch.tensor([total_samples / training_label_counts[cls]
    #                              for cls in range(len(training_label_counts))], dtype=torch.float).to('cuda' if useGPU else 'cpu')
    # logging.info('Using class weights : {}'.format(
    #     class_weights.cpu().numpy()))

    indices = np.arange(len(train_dataset))
    train_indices, val_indices = train_test_split(
        indices, test_size=0.2, stratify=train_dataset.targets
    )
    tr_dataset = Subset(train_dataset, train_indices)
    val_dataset = Subset(train_dataset, val_indices)

    train_loader = DataLoader(tr_dataset, batch_size=batch_size,
                              shuffle=True, pin_memory=useGPU, num_workers=n_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            shuffle=False, pin_memory=useGPU, num_workers=n_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                             shuffle=False, pin_memory=useGPU, num_workers=n_workers)

    return train_loader, val_loader, test_loader
