import torch
import logging
from torchvision import transforms
from src.utils.parameters import instanciate_cls
from torch.utils.data import Dataset, DataLoader


def load_dataloader(dataset_name: str, dataset_params: dict, useGPU: bool = True):
    image_size = dataset_params['image_size']
    batch_size = dataset_params['batch_size']
    n_workers = useGPU * 4 * torch.cuda.device_count()

    if dataset_params['channels'] == 3:
        dataset_transforms = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    (dataset_params['mean'], dataset_params['mean'],
                     dataset_params['mean']),
                    (dataset_params['std'], dataset_params['std'], dataset_params['std'])),
            ]
        )
    else:
        dataset_transforms = transforms.Compose(
            [
                transforms.Grayscale(),
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    (dataset_params['mean'],), (dataset_params['std'],)),
            ]
        )

    if dataset_name == "ImageFolder":
        train_dataset: Dataset = instanciate_cls("torchvision.datasets", dataset_name, {
            'root': dataset_params['data_dir'] + '/train'})
        test_dataset: Dataset = instanciate_cls("torchvision.datasets", dataset_name, {
            'root': dataset_params['data_dir'] + '/test'})
    else:
        train_dataset: Dataset = instanciate_cls("torchvision.datasets", dataset_name, {
            'root': dataset_params['data_dir'], "train": True})
        test_dataset: Dataset = instanciate_cls("torchvision.datasets", dataset_name, {
            'root': dataset_params['data_dir'], "train": False})

    train_dataset.transform = dataset_transforms
    test_dataset.transform = dataset_transforms
    if train_dataset.class_to_idx:
        logging.info(
            f"Available labels in dataset : {train_dataset.class_to_idx}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=useGPU,
        num_workers=n_workers
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=useGPU,
        num_workers=n_workers
    )

    return train_loader, test_loader
