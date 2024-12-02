import torch
from .base import AbstractExperiment
from snntorch import functional as SF
from tqdm import tqdm


class SNNExperiment(AbstractExperiment):
    def __init__(self,
                 model,
                 writer,
                 log_interval,
                 lr, class_weights, weight_decay) -> None:
        super().__init__(model, writer, log_interval, lr, class_weights, weight_decay)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay)
        self.criterion = SF.ce_rate_loss(
            weight=class_weights) if self.class_weights else SF.ce_rate_loss

    def train(self, train_loader):
        self.model.train()
        total = 0
        train_loss = 0
        train_acc = 0

        with tqdm(train_loader, leave=False, desc="Running training phase") as pbar:
            for data, targets in train_loader:
                inputs, labels = data.to(self.device), targets.to(self.device)
                self.optimizer.zero_grad()

                spk_rec, _ = self.model(inputs)
                train_acc += SF.accuracy_rate(spk_rec,
                                              labels) * spk_rec.size(1)
                loss = self.criterion(spk_rec, labels)

                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()
                total += spk_rec.size(1)
                pbar.update(1)

        train_loss /= len(train_loader)
        train_acc /= total
        return train_loss, train_acc.item()

    def test(self, test_loader):
        self.model.eval()
        total = 0
        test_loss = 0
        test_acc = 0

        with torch.no_grad():
            with tqdm(test_loader, leave=False, desc="Running testing phase") as pbar:
                for data, targets in test_loader:
                    inputs, labels = data.to(
                        self.device), targets.to(self.device)

                    spk_rec, _ = self.model(inputs)
                    test_acc += SF.accuracy_rate(spk_rec,
                                                 labels) * spk_rec.size(1)
                    loss = self.criterion(spk_rec, labels)
                    test_loss += loss.item()
                    total += spk_rec.size(1)
                    pbar.update(1)

        test_loss /= len(test_loader)
        test_acc = test_acc / total
        return test_loss, test_acc.item()
