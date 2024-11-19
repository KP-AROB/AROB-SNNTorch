import torch
import logging
import torchvision
from torch.utils.data import random_split
from torchvision import transforms


def load_mnist_dataloader(
        data_dir: str,
        image_size: int,
        batch_size: int = 16,
        gpu: bool = True):
    """
    Retrieves the MNIST Dataset and 
    returns torch dataloader for training and testing

    Parameters :
    ----------------

    :image_size: The input image size
    :batch_size: The batch size
    :gpu: Whether or not the gpu is used

    Returns :
    ----------------
    :train_dataloader:
    :test_dataloader:

    """
    n_workers = gpu * 4 * torch.cuda.device_count()

    train_dataset = torchvision.datasets.MNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.Grayscale(),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        ),
    )

    test_dataset = torchvision.datasets.MNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(image_size),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        ),
    )

    logging.info("Available labels in dataset : ", train_dataset.class_to_idx)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=gpu,
        num_workers=n_workers
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=gpu,
        num_workers=n_workers
    )

    n_classes = len(train_dataset.classes)

    return train_loader, test_loader, n_classes
