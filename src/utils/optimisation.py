import torch
import logging
import os


class EarlyStopping:
    def __init__(self, patience=5, delta=0, verbose=True):
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.counter = 0
        self.best_acc = None
        self.early_stop = False

    def __call__(self, model: torch.nn.Module, epoch: int, val_acc: float, log_dir: str):
        if self.best_acc is None or val_acc > self.best_acc - self.delta:
            self.best_acc = val_acc
            self.counter = 0
            checkpoint = {
                'net': model.state_dict(),
                'epoch': epoch,
                'max_test_acc': self.best_acc
            }
            torch.save(checkpoint, os.path.join(log_dir, 'best_model.pth'))
        else:
            self.counter += 1
            if self.verbose:
                logging.info(
                    f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
