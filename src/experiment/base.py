import torch
import os
from torch.utils.tensorboard import SummaryWriter
from snntorch import functional as SF
from tqdm import tqdm, trange
import numpy as np


class BaseExperiment(object):

    def __init__(self,
                 model,
                 writer: SummaryWriter,
                 log_interval: int = 50,
                 lr: float = .001,
                 device: str = 'cuda') -> None:
        self.model = model
        self.writer = writer
        self.log_interval = log_interval
        self.lr = lr
        self.device = device
        self.optimizer = torch.optim.Adam(
            self.model.net.parameters(),
            lr=lr,
            betas=(0.9, 0.999))
        self.criterion = SF.ce_rate_loss()

    def compute_test_accuracy(self, test_dataloader):
        with torch.no_grad():
            total = 0
            acc = 0
            self.model.net.eval()

            for data, targets in test_dataloader:
                data = data.to(self.device)
                targets = targets.to(self.device)
                spk_rec, _ = self.model.forward_pass(data)
                acc += SF.accuracy_rate(spk_rec, targets) * spk_rec.size(1)
                total += spk_rec.size(1)
        return acc/total

    def train(self, train_loader, epoch):
        losses = []
        self.model.net.train()

        for step, (data, targets) in enumerate(tqdm(train_loader, leave=False, desc="Running training phase")):
            data = data.to(self.device)
            targets = targets.to(self.device)
            self.optimizer.zero_grad()
            spk_rec, _ = self.model.forward_pass(data)
            loss_val = self.criterion(spk_rec, targets)
            loss_val.backward()
            self.optimizer.step()
            losses.append(loss_val.item())
            global_step = (len(train_loader) * train_loader.batch_size) * \
                epoch + train_loader.batch_size * step

            if step % self.log_interval == 0 and step > 0:
                self.writer.add_scalar(
                    'loss/train', np.mean(losses), global_step=global_step)

    def test(self, test_loader):
        self.model.net.eval()
        with torch.no_grad():
            test_acc = self.compute_test_accuracy(test_loader)
        return test_acc.item()

    def fit(self, train_loader, test_loader, num_epochs):
        current_acc = 0
        for epoch in trange(num_epochs, desc="Completed epochs"):
            self.train(train_loader, epoch)
            current_acc = self.test(test_loader)
        torch.save(self.model.net.state_dict(), os.path.join(
            self.writer.log_dir, 'checkpoint.pth'))

        final_acc = round(current_acc * 100, 2)
        with open(self.writer.log_dir + '/parameters.txt', "a") as file:
            file.write(f"\nFinal Accuracy: {final_acc}\n")
        print("\n\033[32mTraining Completed.\033[0m")
        print(
            f"\033[32mFinal test accuracy: {final_acc}%\033[0m\n")
