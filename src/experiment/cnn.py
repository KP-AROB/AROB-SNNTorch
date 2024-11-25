from torch.nn.modules import Module
from torch.utils.tensorboard import SummaryWriter
from .base import AbstractExperiment
from tqdm import tqdm, trange
from torch import nn
import torch.optim as optim
import torch
import os
import logging
from torchmetrics import Accuracy, F1Score
from torch.optim.lr_scheduler import ReduceLROnPlateau


class CNNExperiment(AbstractExperiment):
    def __init__(self, model: nn.Module, writer: SummaryWriter, log_interval: int, lr: float):
        super().__init__(model, writer, log_interval, lr)
        self.criterion = nn.CrossEntropyLoss()
        self.metric = Accuracy(
            task="multiclass", num_classes=self.model.n_output).to(self.device)
        self.optimizer = optim.Adam(
            model.parameters(), lr=lr)

    def train(self, train_loader):
        self.model.train()
        train_loss = 0.0
        train_preds = []
        train_labels = []

        with tqdm(train_loader, leave=False, desc="Running training phase") as pbar:
            for data, targets in train_loader:
                inputs, labels = data.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                self.optimizer.zero_grad()
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

    def fit(self, train_loader, val_loader, num_epochs):
        best_val_loss = float('inf')

        for epoch in range(num_epochs):
            train_loss, train_acc = self.train(train_loader)
            val_loss, val_acc = self.test(val_loader)

            self.writer.add_scalar('train/acc', train_acc, epoch)
            self.writer.add_scalar('train/loss', train_loss, epoch)
            self.writer.add_scalar('val/acc', val_acc, epoch)
            self.writer.add_scalar('val/loss', val_loss, epoch)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), os.path.join(
                    self.writer.log_dir, 'best_model.pth'))

            logging.info(
                f"Epoch {epoch+1}: Train/Val Acc: {train_acc:.4f} | {val_acc:4f}, Train/Val Loss: {train_loss:.4f} | {val_loss:4f}")

        torch.save(self.model.state_dict(), os.path.join(
            self.writer.log_dir, 'final_model.pth'))
