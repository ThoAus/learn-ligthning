
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch import nn, optim
import torchmetrics
import torch.nn.functional as F

import lightning as pl


################################## DataModule ##################################

class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size, num_workers):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        entire_dataset = datasets.MNIST(
            root=self.data_dir,
            train=True,
            download=True,
            transform=transforms.ToTensor(),
            )
        self.train_ds, self.val_ds = random_split(entire_dataset, [50000, 10000])

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            persistent_workers=True,
        )

################################## Lightning Module ##################################

class GenericModule(pl.LightningModule):
    def __init__(self, model, num_classes, learning_rate):
        super().__init__()
        self.model = model
        self.lr = learning_rate
        self.num_classes = num_classes

        # Loss function and metrics
        self.loss_fn = nn.CrossEntropyLoss()
        self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=self.num_classes)
        
    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)

    # To avoid code duplication
    def _common_step(self, batch, batch_idx):
        x, y = batch
        x = x.reshape(x.size(0), -1)
        scores = self.model(x)
        loss = self.loss_fn(scores, y)
        accuracy = self.accuracy(scores, y)
        return y, scores, loss, accuracy

    def training_step(self, batch, batch_idx):
        y, scores, loss, accuracy = self._common_step(batch, batch_idx)
        self.log_dict({"train_loss": loss,"train_acc": accuracy},
                      on_step=False,
                      on_epoch=True,
                      prog_bar=True,
                      )
        return {"loss": loss, "scores": scores, "y": y}

    def validation_step(self, batch, batch_idx):
        y, scores, loss, accuracy = self._common_step(batch, batch_idx)
        self.log_dict({"val_loss": loss,"val_acc": accuracy},
                      on_step=False,
                      on_epoch=True,
                      prog_bar=True,
                      )
        return {"loss": loss, "scores": scores, "y": y}