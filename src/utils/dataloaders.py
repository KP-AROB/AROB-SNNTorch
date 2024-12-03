import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from monai.transforms import *
from torch.utils.data import Dataset
from collections import Counter


def create_sampler(dataset: Dataset):
    label_counts = Counter(dataset.targets)
    class_weights = {label: 1.0 / count for label,
                     count in label_counts.items()}
    sample_weights = [class_weights[label]
                      for _, label in dataset.samples]
    sample_weights_tensor = torch.FloatTensor(sample_weights)
    sampler = WeightedRandomSampler(weights=sample_weights_tensor, num_samples=len(
        sample_weights_tensor), replacement=True)
    return sampler


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
    train_sampler = create_sampler(train_dataset)

    train_loader = DataLoader(train_dataset, batch_size=dataloader_params['batch_size'],
                              pin_memory=useGPU, num_workers=n_workers, sampler=train_sampler)

    test_loader = DataLoader(test_dataset, batch_size=dataloader_params['batch_size'],
                             pin_memory=useGPU, num_workers=n_workers, shuffle=False)

    return train_loader, test_loader
