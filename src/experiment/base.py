import torch
import os
import logging
from torch.utils.tensorboard import SummaryWriter
from snntorch import functional as SF
from tqdm import tqdm, trange
import numpy as np
from snntorch import utils, spikegen
from src.utils.parameters import load_fnc


class BaseExperiment(object):

    def __init__(self,
                 model,
                 writer: SummaryWriter,
                 log_interval: int = 50,
                 lr: float = .001,
                 num_steps: int = 50,
                 encoding_type: str = 'rate',
                 device: str = 'cuda') -> None:
        self.model = model
        self.writer = writer
        self.num_steps = num_steps
        self.log_interval = log_interval
        self.lr = lr
        self.device = device
        self.optimizer = torch.optim.Adam(
            self.model.net.parameters(),
            lr=lr,
            betas=(0.9, 0.999))
        self.criterion = SF.ce_rate_loss()
        self.encoding = load_fnc('snntorch.spikegen', encoding_type)

    def compute_test_accuracy(self, test_dataloader):
        with torch.no_grad():
            total = 0
            acc = 0
            self.model.net.eval()
            with tqdm(test_dataloader, leave=False, desc="Running testing phase") as pbar:
                for data, targets in test_dataloader:
                    data = data.to(self.device)
                    targets = targets.to(self.device)
                    spk_rec, _ = self.compute_forward_pass(data)
                    acc += SF.accuracy_rate(spk_rec, targets) * spk_rec.size(1)
                    total += spk_rec.size(1)
                    pbar.update()
        return acc/total

    def compute_forward_pass(self, data):
        mem_rec = []
        spk_rec = []
        utils.reset(self.model.net)

        spike_data = self.encoding(data, num_steps=self.num_steps)

        for t in range(self.num_steps):
            spk_out, mem_out = self.model.net(spike_data[t])
            spk_rec.append(spk_out)
            mem_rec.append(mem_out)

        return torch.stack(spk_rec), torch.stack(mem_rec)

    def train(self, train_loader, epoch):
        losses = []
        self.model.net.train()

        with tqdm(train_loader, leave=False, desc="Running training phase") as pbar:
            for step, (data, targets) in enumerate(train_loader):
                data = data.to(self.device)
                targets = targets.to(self.device)
                self.optimizer.zero_grad()
                spk_rec, _ = self.compute_forward_pass(data)
                loss_val = self.criterion(spk_rec, targets)
                loss_val.backward()
                self.optimizer.step()
                losses.append(loss_val.item())
                global_step = (len(train_loader) * train_loader.batch_size) * \
                    epoch + train_loader.batch_size * step

                if step % self.log_interval == 0 and step > 0:
                    self.writer.add_scalar(
                        'loss/train', np.mean(losses), global_step=global_step)
                pbar.set_description(
                    f"Running training phase | loss/train : {np.mean(losses):.4f}")
                pbar.update()

    def test(self, test_loader):
        self.model.net.eval()
        with torch.no_grad():
            test_acc = self.compute_test_accuracy(test_loader)
        return test_acc.item()

    def fit(self, train_loader, test_loader, num_epochs):
        current_acc = 0
        pbar = trange(
            num_epochs, desc=f"Epoch 1/{num_epochs} | Test Accuracy: ")
        for epoch in pbar:
            self.train(train_loader, epoch)
            new_acc = self.test(test_loader)
            ev_ratio = round((new_acc - current_acc) * 2, 2)
            ev_ratio_str = f"+{ev_ratio:.2f}%" if ev_ratio > 0 else f"{ev_ratio:.2f}%"
            current_acc = new_acc
            pbar.set_description(
                f"Epoch {epoch+1}/{num_epochs} | Test Accuracy: {round(current_acc * 100, 2):.2f}% ({ev_ratio_str})")
        torch.save(self.model.net.state_dict(), os.path.join(
            self.writer.log_dir, 'checkpoint.pth'))

        final_acc = round(current_acc * 100, 2)
        with open(self.writer.log_dir + '/parameters.txt', "a") as file:
            file.write(f"\nFinal Accuracy: {final_acc}\n")

        logging.info("Training Completed.")
        logging.info("Final test accuracy: {final_acc}%")
