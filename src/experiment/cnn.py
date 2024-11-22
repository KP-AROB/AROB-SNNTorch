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
        self.optimizer = optim.nadam(model.parameters(), lr=lr)

    def train(self, train_loader, epoch):

        self.model.train()
        all_preds = []
        all_labels = []
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
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

                if step % self.log_interval == self.log_interval - 1:
                    self.writer.add_scalar('training loss',
                                           running_loss / 100,
                                           epoch * len(train_loader) + step)

                    if train_loader.dataset.classes:
                        self.writer.add_figure('predictions vs. actuals',
                                               plot_classes_preds(
                                                   self.model, inputs, labels, train_loader.dataset.classes),
                                               global_step=epoch * len(train_loader) + step)
                        running_loss = 0.0
                pbar.update()

        return accuracy_score(all_labels, all_preds)

    def test(self, test_loader):
        self.model.eval()

        val_loss = 0.0
        val_preds = []
        val_labels = []

        with torch.no_grad():
            with tqdm(test_loader, leave=False, desc="Running testing phase") as pbar:
                for data, targets in test_loader:
                    inputs, labels = data.to(
                        self.device), targets.to(self.device)

                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)

                    val_loss += loss.item()
                    _, preds = torch.max(outputs, 1)
                    val_preds.extend(preds.cpu().numpy())
                    val_labels.extend(labels.cpu().numpy())

        mean_val_loss = val_loss / len(test_loader.dataset)
        val_accuracy = accuracy_score(val_labels, val_preds)
        return mean_val_loss, val_accuracy

    def fit(self, train_loader, test_loader, num_epochs):
        pbar = trange(
            num_epochs, desc=f"Epoch 1/{num_epochs} | Test Accuracy: ")
        current_acc = 0
        for epoch in pbar:
            self.train(train_loader, epoch)
            _, test_acc = self.test(test_loader)
            current_acc = test_acc
            pbar.set_description(
                f"Epoch {epoch+1}/{num_epochs} | Test Accuracy: {current_acc:.2f}%")
        torch.save(self.model.state_dict(), os.path.join(
            self.writer.log_dir, 'checkpoint.pth'))

        with open(self.writer.log_dir + '/parameters.txt', "a") as file:
            file.write(f"\nFinal Accuracy: {current_acc}\n")

        logging.info("Training Completed.")
        logging.info(f"Final test accuracy: {current_acc}%")
