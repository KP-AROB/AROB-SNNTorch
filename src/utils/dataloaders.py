import torch
from torch.utils.data import DataLoader
from monai.transforms import *
from torch.utils.data import Dataset


def create_dataloaders(train_dataset: Dataset, test_dataset: Dataset, dataloader_params: dict):
    """Takes train and testing dataset to return associated Dataloaders

    Args:
        train_dataset (Dataset): Train Torch Dataset
        test_dataset (Dataset): Test Torch Dataset
        dataloader_params (dict): Parameters of the dataloader (batch_size)
    Returns:
        Dataloader: Training and Testing Dataloaders
    """
    useGPU = True if torch.cuda.is_available() else False
    n_workers = 4 * torch.cuda.device_count() if useGPU else 2

    train_loader = DataLoader(train_dataset, batch_size=dataloader_params['batch_size'],
                              shuffle=True, pin_memory=useGPU, num_workers=n_workers)

    test_loader = DataLoader(test_dataset, batch_size=dataloader_params['batch_size'],
                             shuffle=False, pin_memory=useGPU, num_workers=n_workers)

    return train_loader, test_loader
