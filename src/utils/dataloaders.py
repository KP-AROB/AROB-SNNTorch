import torch
import logging
import torchvision
from torch.utils.data import random_split
from torchvision import transforms


def load_mnist_dataloader(
        params: dict,
        gpu: bool = True):
    """
    Retrieves the MNIST Dataset and 
    returns torch dataloader for training and testing

    Parameters :
    ----------------

    :params: The parameter dictionary
    :gpu: Whether or not the gpu is used

    Returns :
    ----------------
    :train_dataloader:
    :test_dataloader:

    """
    image_size = params['image_size']
    batch_size = params['batch_size']

    n_workers = gpu * 4 * torch.cuda.device_count()

    train_dataset = torchvision.datasets.MNIST(
        root=params['data_dir'],
        train=True,
        download=True,
        transform=transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.Grayscale(),
                transforms.ToTensor(),
                transforms.Normalize(
                    (params['mean'],), (params['std'],)),
            ]
        ),
    )

    test_dataset = torchvision.datasets.MNIST(
        root=params['data_dir'],
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


def load_cbis_ddsm_dataloader(
        params: dict,
        gpu: bool = True):

    n_workers = gpu * 4 * torch.cuda.device_count()

    train_dataset = torchvision.datasets.ImageFolder(
        params['data_dir'] + '/train',
        transform=transforms.Compose(
            [
                transforms.Grayscale(),
                transforms.ToTensor(),
                transforms.Normalize((params['mean'],), (params['std'],)),
                transforms.Resize(
                    (params['image_size'], params['image_size'])),
            ]
        )
    )

    test_dataset = torchvision.datasets.ImageFolder(
        params['data_dir'] + '/test',
        transform=transforms.Compose(
            [
                transforms.Grayscale(),
                transforms.ToTensor(),
                transforms.Normalize((params['mean'],), (params['std'],)),
                transforms.Resize(
                    (params['image_size'], params['image_size'])),
            ]
        )
    )

    n_classes = len(train_dataset.classes)
    logging.info("Available labels in dataset : {}".format(
        train_dataset.class_to_idx))

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=params['batch_size'],
        shuffle=True,
        num_workers=n_workers,
        pin_memory=gpu,
    )

    val_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=params['batch_size'],
        shuffle=True,
        num_workers=n_workers,
        pin_memory=gpu,
    )

    return train_dataloader, val_dataloader, n_classes


def load_cifar_dataloader(
        params: dict,
        gpu: bool = True):
    """
    Retrieves the MNIST Dataset and 
    returns torch dataloader for training and testing

    Parameters :
    ----------------

    :params: The parameter dictionary
    :gpu: Whether or not the gpu is used

    Returns :
    ----------------
    :train_dataloader:
    :test_dataloader:

    """
    image_size = params['image_size']
    batch_size = params['batch_size']

    n_workers = gpu * 4 * torch.cuda.device_count()

    train_dataset = torchvision.datasets.CIFAR10(
        root=params['data_dir'],
        train=True,
        download=True,
        transform=transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.Grayscale(),
                transforms.ToTensor(),
                transforms.Normalize(
                    (params['mean'],), (params['std'],)),
            ]
        ),
    )

    test_dataset = torchvision.datasets.CIFAR10(
        root=params['data_dir'],
        train=False,
        download=True,
        transform=transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.Grayscale(),
                transforms.ToTensor(),
                transforms.Normalize(
                    (params['mean'],), (params['std'],)),
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
