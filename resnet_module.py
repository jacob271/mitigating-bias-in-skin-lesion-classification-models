import torch
import torchmetrics
import torch.nn as nn
import pytorch_lightning as pl

from torchvision import models


class ResNetModel(pl.LightningModule):

    def __init__(self, in_channels=3, num_classes=4, lr=1e-4):
        super(ResNetModel, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.lr = lr

        self.model = models.resnet18(pretrained=True)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, self.num_classes)

        self.loss_fn = nn.CrossEntropyLoss()

        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = torchmetrics.classification.MulticlassAccuracy(num_classes=num_classes, average='weighted')
        self.test_acc = torchmetrics.classification.MulticlassAccuracy(num_classes=num_classes, average='weighted')

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2)
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        x, y = batch
        preds = self.model(x)
        loss = self.loss_fn(preds, y)
        self.train_acc(torch.argmax(preds, dim=1), y)

        self.log('train_loss', loss.item(), on_epoch=True)
        self.log('train_acc', self.train_acc, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        preds = self.model(x)
        loss = self.loss_fn(preds, y)
        self.val_acc(torch.argmax(preds, dim=1), y)

        self.log('val_loss', loss.item(), on_epoch=True)
        self.log('val_acc', self.val_acc, on_epoch=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        preds = self.model(x)
        self.test_acc(torch.argmax(preds, dim=1), y)

        self.log('test_acc', self.test_acc, on_epoch=True)

    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = 0):
        imgs, labels = batch
        preds = self.model(imgs)
        # return torch.argmax(preds, dim=1)
        return preds[:, :2]
