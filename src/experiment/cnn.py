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
from torchmetrics import Accuracy


class CNNExperiment(AbstractExperiment):
    def __init__(self, model: Module, writer: SummaryWriter, log_interval: int, lr: float) -> None:
        super().__init__(model, writer, log_interval, lr)
        self.criterion = nn.CrossEntropyLoss()
        self.metric = Accuracy(
            task="multiclass", num_classes=self.model.n_output).to(self.device)
        self.optimizer = optim.NAdam(model.parameters(), lr=lr)

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

        test_loss = 0
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
        test_acc = self.metric(torch.cat(test_preds),
                               torch.cat(test_labels))
        return test_loss, test_acc.item()

    def fit(self, train_loader, test_loader, num_epochs):
        pbar = trange(
            num_epochs, desc="Completed Epoch")
        train_acc = 0
        test_acc = 0

        for epoch in pbar:
            train_loss, train_acc = self.train(train_loader)
            test_loss, test_acc = self.test(test_loader)
            self.writer.add_scalar('train/acc', train_acc, epoch)
            self.writer.add_scalar('train/loss', train_loss, epoch)
            self.writer.add_scalar('test/acc', test_acc, epoch)
            self.writer.add_scalar('test/loss', test_loss, epoch)

        torch.save(self.model.state_dict(), os.path.join(
            self.writer.log_dir, 'checkpoint.pth'))

        with open(self.writer.log_dir + '/parameters.txt', "a") as file:
            file.write(f"\nFinal Train Accuracy: {train_acc}\n")
            file.write(f"\nFinal Test Accuracy: {test_acc}\n")

        logging.info("Training Completed.")
        logging.info(f"Final train accuracy: {train_acc} %")
        logging.info(f"Final test accuracy: {test_acc} %")
