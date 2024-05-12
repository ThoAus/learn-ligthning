import torch
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import torchmetrics
from torchmetrics import Metric
from tqdm import tqdm                           # Progress bar
import lightning as L                           # PyTorch Lightning
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.tuner import Tuner       # for hyperparameter tuning (lr_finder)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Training hyperparameters
INPUT_SIZE = 784
NUM_CLASSES = 10
LEARNING_RATE = 0.001
DROPOUT_RATE = 0.1
MC_DROPOUT_SAMPLES = 50
BATCH_SIZE = 1024
NUM_EPOCHS = 10

# Dataset
DATA_DIR = "dataset/"
NUM_WORKERS = 4

# DataModule (Dataloader)
class MnistDataModule(L.LightningDataModule):
    def __init__(self, data_dir, batch_size, num_workers):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self):
        datasets.MNIST(self.data_dir, train=True, download=True)
        datasets.MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage):
        entire_dataset = datasets.MNIST(
            root=self.data_dir,
            train=True,
            transform=transforms.ToTensor(),
            download=False,
        )
        self.train_ds, self.val_ds = random_split(entire_dataset, [50000, 10000])
        self.test_ds = datasets.MNIST(
            root=self.data_dir,
            train=False,
            transform=transforms.ToTensor(),
            download=False,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            persistent_workers=True,        # Considered to speed up the dataloader worker initialization
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            persistent_workers=True,        # Considered to speed up the dataloader worker initialization
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            persistent_workers=True,        # Considered to speed up the dataloader worker initialization
        )
    
    def predict_dataloader(self):
        return DataLoader(
            self.val_ds,                    # Use the entire validation dataset for prediction
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            persistent_workers=True,        # Considered to speed up the dataloader worker initialization
        )

# LightningModule (PyTorch Model)
class LightningModule(L.LightningModule):
    def __init__(self, input_size, learning_rate, num_classes, dropout_rate, dropout_samples):
        super().__init__()
        self.lr = learning_rate
        self.num_classes = num_classes
        self.dropout_samples = dropout_samples
        self.save_hyperparameters()

        # Layers
        self.fc1 = nn.Linear(input_size, 20)
        self.fc2 = nn.Linear(20, 20)
        self.fc3 = nn.Linear(20, self.num_classes)
        self.dropout = nn.Dropout(p=dropout_rate)
        
        # Loss function and metrics
        self.loss_fn = nn.CrossEntropyLoss()
        self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=self.num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x
    
    # Common step for training, validation, and test steps, to avoid code duplication
    def _common_step(self, batch, batch_idx):
        x, y = batch
        x = x.reshape(x.size(0), -1)
        scores = self.forward(x)
        loss = self.loss_fn(scores, y)
        return loss, scores, y

    def training_step(self, batch, batch_idx):
        loss, scores, y = self._common_step(batch, batch_idx)
        accuracy = self.accuracy(scores, y)
        self.log_dict({"train_loss": loss,"train_acc": accuracy},
                      on_step=False,
                      on_epoch=True,
                      prog_bar=True,
                      )
        return {"loss": loss, "scores": scores, "y": y}

    def validation_step(self, batch, batch_idx):
        loss, scores, y = self._common_step(batch, batch_idx)
        accuracy = self.accuracy(scores, y)
        self.log_dict({"val_loss": loss,"val_acc": accuracy},
                      on_step=False,
                      on_epoch=True,
                      prog_bar=True,
                      )
        return {"loss": loss, "scores": scores, "y": y}

    def test_step(self, batch, batch_idx):
        loss, scores, y = self._common_step(batch, batch_idx)
        self.log("test_loss", loss)
        return loss

    # Prediction step applying Monte Carlo Dropout
    def predict_step(self, batch, batch_idx):
        x, y = batch
        x = x.reshape(x.size(0), -1)
        # Enable dropout
        self.dropout.train()
        predictions = torch.zeros(self.dropout_samples, x.size(0), self.num_classes)
        for i in range(self.dropout_samples):
            scores = self.forward(x)
            predictions[i] = F.softmax(scores, dim=1)
        return predictions

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)

# Initialize and train model
dm = MnistDataModule(data_dir=DATA_DIR,
                     batch_size=BATCH_SIZE,
                     num_workers=NUM_WORKERS,
                     )

model = LightningModule(input_size=INPUT_SIZE,
                        learning_rate=LEARNING_RATE,
                        num_classes=NUM_CLASSES,
                        dropout_rate=DROPOUT_RATE,
                        dropout_samples=MC_DROPOUT_SAMPLES,
                        )

logger = CSVLogger("logs", name="FCM_MNIST")

trainer = L.Trainer(max_epochs=NUM_EPOCHS,
                    logger=logger,
                    )

trainer.fit(model, dm)