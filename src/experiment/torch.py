from torch.utils.tensorboard import SummaryWriter
from .base import AbstractExperiment
from tqdm import tqdm
from torch import nn
import torch


class CNNExperiment(AbstractExperiment):
    def __init__(self, model: nn.Module, writer: SummaryWriter, log_interval: int, lr: float, early_stopping_patience: int, weight_decay: float):
        super().__init__(model, writer, log_interval,
                         lr, early_stopping_patience, weight_decay)

        if self.model.n_output == 2:
            self.criterion = nn.BCELoss()
        else:
            self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    def train(self, train_loader):
        self.model.train()
        train_loss = 0.0
        train_preds = []
        train_labels = []

        with tqdm(train_loader, leave=False, desc="Running training phase") as pbar:
            for data, targets in train_loader:
                inputs, labels = data.to(self.device), targets.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()
                train_preds.append(outputs.cpu())
                train_labels.append(labels.cpu())
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
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)
                    test_loss += loss.item()

                    test_preds.append(outputs.cpu())
                    test_labels.append(labels.cpu())
                    pbar.update(1)

        test_loss /= len(test_loader)
        test_preds = torch.concat(test_preds)
        test_labels = torch.concat(test_labels)

        test_metrics = self.compute_metrics(
            test_preds, test_labels, 'val')
        return test_loss, test_metrics
