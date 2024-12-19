import torch
import os
import logging
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from abc import ABC, abstractmethod
from torch.utils.data import DataLoader
from torchmetrics import F1Score, AUROC, Recall, Specificity, Accuracy
from src.utils.optimisation import EarlyStopping
from torch.optim.lr_scheduler import ExponentialLR, ReduceLROnPlateau
import torch.optim as optim


class AbstractExperiment(ABC):

    def __init__(self, model: torch.nn.Module, writer: SummaryWriter, log_interval: int, lr: float, early_stopping_patience: int = 5, weight_decay: float = 0, param_obj: dict = {}) -> None:
        super().__init__()
        self.model = model
        self.writer = writer
        self.log_interval = log_interval
        self.lr = lr
        self.param_obj = param_obj
        self.weight_decay = weight_decay
        self.early_stopping_patience = early_stopping_patience
        self.device = torch.device(
            "cuda") if torch.cuda.is_available() else torch.device("cpu")

        metric_task = 'binary' if self.model.n_output == 2 else 'multiclass'
        self.metrics = {
            'acc': Accuracy(task=metric_task, num_classes=self.model.n_output),
            'recall': Recall(task=metric_task, num_classes=self.model.n_output),
            'f1': F1Score(task=metric_task, num_classes=self.model.n_output),
            'auroc': AUROC(task=metric_task, num_classes=self.model.n_output),
            'specificity': Specificity(task=metric_task, num_classes=self.model.n_output),
        }
        self.early_stopping = EarlyStopping(
            patience=early_stopping_patience, verbose=False, param_obj=self.param_obj)

        self.optimizer = optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.lr_scheduler = ReduceLROnPlateau(self.optimizer)

    @abstractmethod
    def train(self, dl: DataLoader):
        raise NotImplementedError

    @abstractmethod
    def test(self, dl: DataLoader):
        raise NotImplementedError

    def compute_metrics(self, preds, targets, mode):
        val_results = {}
        for name, metric in self.metrics.items():
            outputs = torch.argmax(
                preds, dim=1) if name == 'acc' or self.model.n_output == 2 else preds
            val_results[f'{mode}/{name}'] = metric(outputs, targets)
        return val_results

    def log_metrics(self, train_loss, val_loss, train_metrics, val_metrics, epoch, scalar_prefix=None):
        prefix = f"{scalar_prefix}/" if scalar_prefix else ""
        self.writer.add_scalar(f'{prefix}train/loss', train_loss, epoch)
        self.writer.add_scalar(f'{prefix}val/loss', val_loss, epoch)
        for (train_name, train_metric_value), (val_name, val_metric_value) in zip(train_metrics.items(), val_metrics.items()):
            self.writer.add_scalar(train_name, train_metric_value, epoch)
            self.writer.add_scalar(val_name, val_metric_value, epoch)
        return

    def fit(self, train_loader, val_loader, num_epochs):
        with open(self.writer.log_dir + '/parameters.txt', "a") as file:
            file.write(f"\n{self.model}\n")

        for epoch in range(num_epochs):
            train_loss, train_metrics = self.train(train_loader)
            val_loss, val_metrics = self.test(val_loader)
            self.log_metrics(train_loss, val_loss,
                             train_metrics, val_metrics, epoch, None)
            self.lr_scheduler.step(val_loss)
            self.early_stopping(self.model, epoch,
                                val_metrics['val/acc'], self.writer.log_dir)

            logging.info(
                f"Epoch {epoch+1} / {num_epochs} - LR : {self.lr_scheduler.get_last_lr()[0]:6f} T/V Acc: {train_metrics['train/acc']:.4f} | {val_metrics['val/acc']:.4f}, T/V AUROC: {train_metrics['train/auroc']:.4f} | {val_metrics['val/auroc']:4f}, T/V Loss: {train_loss:.4f} | {val_loss:4f}")

            if self.early_stopping.early_stop:
                logging.info(
                    f"Val loss did not improve for {self.early_stopping.patience} epochs.")
                logging.info('Training stopped by early stopping mecanism.')
                break
