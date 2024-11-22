from torch.nn.modules import Module
from torch.utils.tensorboard import SummaryWriter
from .base import AbstractExperiment
from tqdm import tqdm, trange
from torch import nn
import torch.optim as optim
import torch
import os
import logging
import numpy as np
from sklearn.metrics import accuracy_score
from src.utils.tracking import plot_classes_preds


class CNNExperiment(AbstractExperiment):
    def __init__(self, model: Module, writer: SummaryWriter, log_interval: int, lr: float) -> None:
        super().__init__(model, writer, log_interval, lr)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.NAdam(model.parameters(), lr=lr)

    def train(self, train_loader, epoch):

        self.model.train()
        running_loss = 0.0

        with tqdm(train_loader, leave=False, desc="Running training phase") as pbar:
            for step, (data, targets) in enumerate(train_loader):
                inputs, labels = data.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()

                if step % self.log_interval == self.log_interval - 1:
                    self.writer.add_scalar('train/loss',
                                           running_loss / 100,
                                           epoch * len(train_loader) + step)

                    if train_loader.dataset.classes:
                        self.writer.add_figure('train/preds',
                                               plot_classes_preds(
                                                   self.model, inputs, labels, train_loader.dataset.classes),
                                               global_step=epoch * len(train_loader) + step)
                        running_loss = 0.0
                pbar.update()

    def test(self, test_loader):
        self.model.eval()

        val_correct = 0
        val_total = 0

        with torch.no_grad():
            with tqdm(test_loader, leave=False, desc="Running testing phase") as pbar:
                for data, targets in test_loader:
                    inputs, labels = data.to(
                        self.device), targets.to(self.device)
                    outputs = self.model(inputs)
                    _, preds = torch.max(outputs, 1)
                    val_total += labels.size(0)
                    val_correct += (preds == labels).sum().item()
                    pbar.update()
        accuracy = 100 * val_correct / val_total
        return accuracy

    def fit(self, train_loader, test_loader, num_epochs):
        pbar = trange(
            num_epochs, desc=f"Epoch 1/{num_epochs}")
        current_acc = 0
        for epoch in pbar:
            self.train(train_loader, epoch)
            test_acc = self.test(test_loader)
            current_acc = test_acc
            self.writer.add_scalar('test/accuracy', current_acc, epoch)

        torch.save(self.model.state_dict(), os.path.join(
            self.writer.log_dir, 'checkpoint.pth'))

        with open(self.writer.log_dir + '/parameters.txt', "a") as file:
            file.write(f"\nFinal Accuracy: {current_acc}\n")

        logging.info("Training Completed.")
        logging.info(f"Final test accuracy: {current_acc}%")
