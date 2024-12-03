import torch
import os
import logging
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from abc import ABC, abstractmethod
from torch.utils.data import DataLoader
from torchmetrics import F1Score, AUROC, Recall, Specificity, Accuracy


class AbstractExperiment(ABC):

    def __init__(self, model: torch.nn.Module, writer: SummaryWriter, log_interval: int, lr: float, weight_decay: float = 0) -> None:
        super().__init__()
        self.model = model
        self.writer = writer
        self.log_interval = log_interval
        self.lr = lr
        self.weight_decay = weight_decay
        self.device = torch.device(
            "cuda") if torch.cuda.is_available() else torch.device("cpu")

        self.metrics = {
            'f1': F1Score(task='multiclass', num_classes=self.model.n_output),
            'auroc': AUROC(task='multiclass', num_classes=self.model.n_output),
            'specificity': Specificity(task='multiclass', num_classes=self.model.n_output),
            'acc': Accuracy(task='multiclass', num_classes=self.model.n_output),
        }

    @abstractmethod
    def train(self, dl: DataLoader):
        raise NotImplementedError

    @abstractmethod
    def test(self, dl: DataLoader):
        raise NotImplementedError

    def compute_metrics(self, preds, targets, mode):
        val_results = {}
        for name, metric in self.metrics.items():
            outputs = torch.argmax(preds, dim=1) if name == 'acc' else preds
            val_results[f'{mode}/{name}'] = metric(outputs, targets)
        return val_results

    def log_metrics(self, train_loss, val_loss, train_metrics, val_metrics, epoch):
        self.writer.add_scalar('train/loss', train_loss, epoch)
        self.writer.add_scalar('val/loss', val_loss, epoch)
        for (train_name, train_metric_value), (val_name, val_metric_value) in zip(train_metrics.items(), val_metrics.items()):
            self.writer.add_scalar(train_name, train_metric_value, epoch)
            self.writer.add_scalar(val_name, val_metric_value, epoch)
        return

    def fit(self, train_loader, val_loader, num_epochs):
        best_val_loss = float('inf')

        for epoch in range(num_epochs):
            train_loss, train_metrics = self.train(train_loader)
            val_loss, val_metrics = self.test(val_loader)
            self.log_metrics(train_loss, val_loss,
                             train_metrics, val_metrics, epoch)
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
                f"Epoch {epoch+1} / {num_epochs}: Train/Val Acc: {train_metrics['train/acc']:.4f} | {val_metrics['val/acc']:4f}, Train/Val Loss: {train_loss:.4f} | {val_loss:4f}")

        torch.save(self.model.state_dict(), os.path.join(
            self.writer.log_dir, 'final_model.pth'))
