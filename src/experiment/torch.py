from torch.utils.tensorboard import SummaryWriter
from .base import AbstractExperiment
from tqdm import tqdm
from torch import nn
import torch.optim as optim
import torch
from torchmetrics import Accuracy


class CNNExperiment(AbstractExperiment):
    def __init__(self, model: nn.Module, writer: SummaryWriter, log_interval: int, lr: float, class_weights: torch.Tensor, weight_decay: float):
        super().__init__(model, writer, log_interval, lr, class_weights, weight_decay)
        self.criterion = nn.CrossEntropyLoss(
            weight=class_weights) if self.class_weights else nn.CrossEntropyLoss()
        self.metric = Accuracy(
            task="multiclass", num_classes=self.model.n_output).to(self.device)
        self.optimizer = optim.NAdam(
            model.parameters(), lr=lr, weight_decay=weight_decay)

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
                train_preds.append(torch.argmax(outputs, dim=1).cpu())
                train_labels.append(labels.cpu())
                pbar.update(1)

        train_loss /= len(train_loader)
        train_acc = self.metric(torch.cat(train_preds),
                                torch.cat(train_labels))
        return train_loss, train_acc.item()

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

                    test_preds.append(torch.argmax(outputs, dim=1).cpu())
                    test_labels.append(labels.cpu())
                    pbar.update(1)

        test_loss /= len(test_loader)
        test_acc = self.metric(torch.cat(test_preds), torch.cat(test_labels))
        return test_loss, test_acc.item()
