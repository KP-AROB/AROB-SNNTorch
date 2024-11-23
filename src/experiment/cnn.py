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
        train_preds = []
        train_labels = []

        with tqdm(train_loader, leave=False, desc="Running training phase") as pbar:
            for step, (data, targets) in enumerate(train_loader):
                inputs, labels = data.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
                train_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
                train_labels.extend(labels.cpu().numpy())

                if step % self.log_interval == self.log_interval - 1:
                    self.writer.add_scalar('train/loss',
                                           running_loss / 100,
                                           epoch * len(train_loader) + step)

                    self.writer.add_scalar('train/acc',
                                           accuracy_score(
                                               train_labels, train_preds),
                                           epoch * len(train_loader) + step)
                    running_loss = 0.0
                pbar.update()

    def test(self, test_loader):
        self.model.eval()

        test_preds = []
        test_labels = []

        with torch.no_grad():
            with tqdm(test_loader, leave=False, desc="Running testing phase") as pbar:
                for data, targets in test_loader:
                    inputs, labels = data.to(
                        self.device), targets.to(self.device)
                    outputs = self.model(inputs)
                    test_preds.extend(torch.argmax(
                        outputs, dim=1).cpu().numpy())
                    test_labels.extend(labels.cpu().numpy())
                    pbar.update()
        test_acc = accuracy_score(test_labels, test_preds)
        return test_acc

    def fit(self, train_loader, test_loader, num_epochs):
        pbar = trange(
            num_epochs, desc=f"Epoch 1/{num_epochs}")
        current_acc = 0
        for epoch in pbar:
            self.train(train_loader, epoch)
            test_acc = self.test(test_loader)
            current_acc = test_acc
            self.writer.add_scalar('test/acc', current_acc, epoch)

        torch.save(self.model.state_dict(), os.path.join(
            self.writer.log_dir, 'checkpoint.pth'))

        with open(self.writer.log_dir + '/parameters.txt', "a") as file:
            file.write(f"\nFinal Accuracy: {current_acc}\n")

        logging.info("Training Completed.")
        logging.info(f"Final test accuracy: {current_acc}%")
