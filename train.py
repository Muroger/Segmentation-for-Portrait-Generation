# visualization library
from matplotlib import pyplot as plt

import numpy as np
import pandas as pd

# torch libraries
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
import segmentation_models_pytorch as smp

# others
import os
import time
import random
from tqdm import tqdm_notebook as tqdm
from pathlib import Path
import PIL

from dataloader import (
    mean,
    std,
    get_transform,
    PortraitDataset,
    PortraitDataloader,
)
from evaluation import dice_score, Scores, epoch_log

# *****************to reproduce same results fixing the seed and hash*******************
seed = 42
random.seed(seed)
os.environ["PYTHONHASHSEED"] = str(seed)
np.random.seed(seed)
torch.cuda.manual_seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True

#base_path = Path(__file__).parent.parent
#data_path = Path(base_path / "data/").resolve()
data_path = Path("./APDrawingDB/data/").resolve()


class Trainer(object):
    def __init__(self, model):
        self.num_workers = 4
        self.batch_size = {"train": 8, "val": 1}
        self.accumulation_steps = 24 // self.batch_size["train"]
        self.lr = 1e-3
        self.num_epochs = 50
        self.phases = ["train", "val"]
        self.best_loss = float("inf")
        self.device = torch.device("cuda:0")
        torch.set_default_tensor_type("torch.cuda.FloatTensor")
        self.net = model.to(self.device)
        cudnn.benchmark = True
        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr)
        #self.scheduler = ReduceLROnPlateau(self.optimizer, mode="min", patience=3, verbose=True)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=50, eta_min=1e-6, last_epoch=-1)
        self.dataloaders = {
            phase: PortraitDataloader(
                df[phase],
                img_fol[phase],
                mask_fol[phase],
                mean,
                std,
                phase=phase,
                batch_size=self.batch_size[phase],
                num_workers=self.num_workers,
            )
            for phase in self.phases
        }

        self.losses = {phase: [] for phase in self.phases}
        self.dice_score = {phase: [] for phase in self.phases}

    def forward(self, inp_images, tar_mask):
        #print(inp_images.shape)
        #print(inp_images[0].shape)
        inp_images = inp_images.to(self.device)
        tar_mask = tar_mask.to(self.device)

        pred_mask = self.net(inp_images)
        loss = self.criterion(pred_mask, tar_mask)
        return loss, pred_mask

    def iterate(self, epoch, phase):
        measure = Scores(phase, epoch)
        start = time.strftime("%H:%M:%S")
        print(f"Starting epoch: {epoch} | phase:{phase} | 🙊':{start}")
        batch_size = self.batch_size[phase]
        self.net.train(phase == "train")
        dataloader = self.dataloaders[phase]
        running_loss = 0.0
        total_batches = len(dataloader)
        self.optimizer.zero_grad()
        for itr, batch in enumerate(dataloader):
            images, mask_target = batch
            loss, pred_mask = self.forward(images, mask_target)
            loss = loss / self.accumulation_steps
            if phase == "train":
                loss.backward()
                if (itr + 1) % self.accumulation_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
            running_loss += loss.item()
            pred_mask = pred_mask.detach().cpu()
            measure.update(mask_target, pred_mask)
        epoch_loss = (running_loss * self.accumulation_steps) / total_batches
        dice = epoch_log(epoch_loss, measure)
        self.losses[phase].append(epoch_loss)
        self.dice_score[phase].append(dice)
        torch.cuda.empty_cache()
        return epoch_loss

    def start(self):
        for epoch in range(self.num_epochs):
            self.iterate(epoch, "train")
            state = {
                "epoch": epoch,
                "best_loss": self.best_loss,
                "state_dict": self.net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            }
            with torch.no_grad():
                val_loss = self.iterate(epoch, "val")
                self.scheduler.step()#val_loss)
            if val_loss < self.best_loss:
                print("******** optimal found, saving state ********")
                state["best_loss"] = self.best_loss = val_loss
                torch.save(state, "./model_office_384x384_effnet_2.pth")
            print()


if __name__ == "__main__":
    df_train = pd.DataFrame()
    df_train['image_name'] = list(os.listdir('./APDrawingDB/data/train_imgs/'))

    df_test = pd.DataFrame()
    df_test['image_name'] = list(os.listdir('./APDrawingDB/data/test_imgs/'))
    df = {'train': df_train, 'val': df_test}
    #df = pd.read_csv(data_path / "Metadata.csv")

    # location of original and mask image
    img_fol = {'train': data_path / "train_imgs", 'val': data_path / "test_imgs"}
    mask_fol = {'train': data_path / "train_masks", 'val': data_path / "test_masks"}
    #mask_fol = data_path / "train_masks"

    #model = smp.Unet("resnext50_32x4d", encoder_weights="imagenet", classes=1, activation='sigmoid')
    model = smp.Unet("timm-efficientnet-b4", encoder_weights="noisy-student", classes=1, activation='sigmoid')

    #print(model)
    model_trainer = Trainer(model)
    model_trainer.start()
