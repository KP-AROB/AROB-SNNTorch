import torch
from .base import AbstractExperiment
from tqdm import tqdm
import torch.nn.functional as F
from spikingjelly.activation_based import functional
import numpy as np
from torch.utils.tensorboard import SummaryWriter


class SJellyExperiment(AbstractExperiment):
    def __init__(self, model: torch.nn.Module, writer: SummaryWriter, log_interval: int, lr: float, early_stopping_patience: int, weight_decay: float, param_obj: dict):
        super().__init__(model, writer, log_interval,
                         lr, early_stopping_patience, weight_decay, param_obj)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay)

        self.criterion = torch.nn.MSELoss()

        np.int = np.int64

    def train(self, train_loader):
        self.model.train()

        train_loss = 0.0
        train_preds = []
        train_labels = []

        with tqdm(train_loader, leave=False, desc="Running training phase") as pbar:
            for data, targets in train_loader:
                inputs, labels = data.to(self.device), targets.to(self.device)
                label_onehot = F.one_hot(labels, self.model.n_output).float()
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, label_onehot)
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()
                train_preds.append(outputs.cpu())
                train_labels.append(labels.cpu())
                functional.reset_net(self.model)
                pbar.update(1)

        train_loss /= len(train_loader)
        train_preds = torch.concat(train_preds)
        train_labels = torch.concat(train_labels)
        train_metrics = self.compute_metrics(
            train_preds, train_labels, 'train')
        return train_loss, train_metrics

    def test(self, test_loader):
        self.model.eval()
        test_loss = 0.0
        test_preds = []
        test_labels = []

        with torch.no_grad():
            with tqdm(test_loader, leave=False, desc="Running testing phase") as pbar:
                for data, targets in test_loader:
                    inputs, labels = data.to(
                        self.device), targets.to(self.device)
                    label_onehot = F.one_hot(
                        labels, self.model.n_output).float()
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, label_onehot)
                    test_loss += loss.item()

                    test_preds.append(outputs.cpu())
                    test_labels.append(labels.cpu())
                    functional.reset_net(self.model)
                    pbar.update(1)

        test_loss /= len(test_loader)
        test_preds = torch.concat(test_preds)
        test_labels = torch.concat(test_labels)

        test_metrics = self.compute_metrics(
            test_preds, test_labels, 'val')
        return test_loss, test_metrics
