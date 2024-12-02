import torch
from .base import AbstractExperiment
from snntorch import functional as SF
from tqdm import tqdm
import torch.nn.functional as F
from spikingjelly.activation_based import functional
import numpy as np


class SJellyExperiment(AbstractExperiment):
    def __init__(self,
                 model,
                 writer,
                 log_interval,
                 lr, class_weights, weight_decay) -> None:
        super().__init__(model, writer, log_interval, lr, class_weights, weight_decay)
        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay)

        if self.model.n_output == 2:
            self.criterion = torch.nn.BCELoss()
        else:
            self.criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

        np.int = np.int64

    def train(self, train_loader):
        self.model.train()

        train_loss = 0
        train_acc = 0
        train_samples = 0

        with tqdm(train_loader, leave=False, desc="Running training phase") as pbar:
            for data, targets in train_loader:
                inputs, labels = data.to(self.device), targets.to(self.device)
                self.optimizer.zero_grad()
                label_onehot = F.one_hot(labels, self.model.n_output).float()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, label_onehot)
                loss.backward()
                self.optimizer.step()

                train_samples += labels.numel()
                train_loss += loss.item() * labels.numel()
                train_acc += (outputs.argmax(1) == labels).float().sum().item()

                functional.reset_net(self.model)

                pbar.update(1)

        train_loss /= train_samples
        train_acc /= train_samples
        return train_loss, train_acc

    def test(self, test_loader):
        self.model.eval()

        test_loss = 0
        test_acc = 0
        test_samples = 0

        with torch.no_grad():
            with tqdm(test_loader, leave=False, desc="Running testing phase") as pbar:
                for data, targets in test_loader:
                    inputs, labels = data.to(
                        self.device), targets.to(self.device)
                    label_onehot = F.one_hot(
                        labels, self.model.n_output).float()
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, label_onehot)

                    test_samples += labels.numel()
                    test_loss += loss.item() * labels.numel()
                    test_acc += (outputs.argmax(1) ==
                                 labels).float().sum().item()
                    functional.reset_net(self.model)
                    pbar.update(1)

        test_loss /= test_samples
        test_acc /= test_samples
        return test_loss, test_acc
