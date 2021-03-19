from typing import List, Dict

import pytorch_lightning as pl
import torch
from torch.nn import functional as F
from torch.optim.adam import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

from channel_order.config import Config
from channel_order.dataset import ChannelOrderDataset
from channel_order.model import get_model


class TrainSystem(pl.LightningModule):
    def __init__(self, config: Config):
        super(TrainSystem, self).__init__()
        # prepare model
        self.config = config
        self.model = get_model(config.model)
        self.epochs = config.epochs

    def forward(self, x):
        return self.model.forward(x)

    def training_step(self, batch, batch_idx):
        images, labels = batch
        labels = labels.unsqueeze(1).type(images.dtype)
        predictions = self.forward(images)
        loss = F.binary_cross_entropy_with_logits(predictions, labels)
        acc = torch.sum(labels == (predictions > 0.5)) * 1.0 / len(labels)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train_acc", acc, prog_bar=True, on_step=True, on_epoch=True)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        labels = labels.unsqueeze(1).type(images.dtype)
        predictions = self.forward(images)
        loss = F.binary_cross_entropy_with_logits(predictions, labels)
        acc = torch.sum(labels == (predictions > 0.5)) * 1.0 / len(labels)
        self.log("val_loss", loss, on_step=False, on_epoch=True)
        self.log("val_acc", acc, on_step=False, on_epoch=True)
        return {"val_loss": loss}

    def test_step(self, batch, batch_idx):
        images, labels = batch
        labels = labels.unsqueeze(1).type(images.dtype)
        predictions = self.forward(images)
        print(predictions, labels)
        loss = F.binary_cross_entropy_with_logits(predictions, labels)
        acc = torch.sum(labels == (predictions > 0.5)) * 1.0 / len(labels)
        self.log("test_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("test_acc", acc, prog_bar=True, on_step=False, on_epoch=True)
        return {"test_loss": loss}

    def configure_optimizers(self):
        optimizer = Adam(self.model.parameters())
        scheduler = StepLR(optimizer, step_size=10, gamma=0.5)
        return [optimizer], [scheduler]

    def train_dataloader(self):
        # REQUIRED
        train_dataset = ChannelOrderDataset(root_dir=self.config.dataset.train)
        return DataLoader(
            train_dataset,
            num_workers=8,
            batch_size=32,
        )

    def val_dataloader(self):
        # OPTIONAL
        val_dataset = ChannelOrderDataset(root_dir=self.config.dataset.val)
        return DataLoader(
            val_dataset,
            num_workers=4,
            batch_size=32,
        )

    def test_dataloader(self):
        # OPTIONAL
        test_dataset = ChannelOrderDataset(
            root_dir=self.config.dataset.test, glob_search_word="*HR.png"
        )
        return DataLoader(
            test_dataset,
            num_workers=4,
            batch_size=10,
        )
