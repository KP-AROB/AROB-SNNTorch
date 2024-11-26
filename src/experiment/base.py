import torch
import os
import logging
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from abc import ABC, abstractmethod
from torch.utils.data import DataLoader


class AbstractExperiment(ABC):

    def __init__(self, model: torch.nn.Module, writer: SummaryWriter, log_interval: int, lr: float, class_weights: torch.Tensor = None, weight_decay: float = 0) -> None:
        super().__init__()
        self.model = model
        self.writer = writer
        self.log_interval = log_interval
        self.lr = lr
        self.class_weights = class_weights
        self.device = torch.device(
            "cuda") if torch.cuda.is_available() else torch.device("cpu")

    @abstractmethod
    def train(self, dl: DataLoader):
        raise NotImplementedError

    @abstractmethod
    def test(self, dl: DataLoader):
        raise NotImplementedError

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
                checkpoint = {
                    'net': self.model.state_dict(),
                    'epoch': epoch,
                    'max_test_acc': best_val_loss
                }
                torch.save(checkpoint, os.path.join(
                    self.writer.log_dir, 'best_model.pth'))

            logging.info(
                f"Epoch {epoch+1} / {num_epochs}: Train/Val Acc: {train_acc:.4f} | {val_acc:4f}, Train/Val Loss: {train_loss:.4f} | {val_loss:4f}")

        torch.save(self.model.state_dict(), os.path.join(
            self.writer.log_dir, 'final_model.pth'))
