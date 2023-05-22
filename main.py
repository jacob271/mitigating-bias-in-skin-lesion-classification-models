import os

import torch
import torchmetrics
from pytorch_lightning.callbacks import ModelCheckpoint
from torch import optim
import torch.nn as nn
import pytorch_lightning as pl

from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
from torchvision import models

from dataset import get_dataset
from model import ResNet

wandb_logger = WandbLogger(project="bias-skin-lesion-detection")

classes = ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"]

CHECKPOINT_PATH = os.environ.get("PATH_CHECKPOINT", "./saved_models")
os.makedirs(CHECKPOINT_PATH, exist_ok=True)


class ResNetModel(pl.LightningModule):

    def __init__(self, pretrained=False, in_channels=3, num_classes=4, lr=3e-4, freeze=False):
        super(ResNetModel, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.lr = lr

        self.model = models.resnet18(pretrained=pretrained)

        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False

        self.model.fc = nn.Sequential(
            nn.Linear(self.model.fc.in_features, 128),
            nn.Dropout(0.3),
            nn.Linear(128, self.num_classes)
        )

        self.loss_fn = nn.CrossEntropyLoss()

        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=4)
        self.val_acc = torchmetrics.classification.MulticlassAccuracy(num_classes=4, average='weighted')
        self.test_acc = torchmetrics.classification.MulticlassAccuracy(num_classes=4, average='weighted')

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

        return torch.argmax(preds, dim=1)


class SkinLesionModule(pl.LightningModule):
    def __init__(self, model_hparams, optimizer_name, optimizer_hparams):
        """
        Inputs:
            model_hparams - Hyperparameters for the model, as dictionary.
            optimizer_name - Name of the optimizer to use. Currently supported: Adam, SGD
            optimizer_hparams - Hyperparameters for the optimizer, as dictionary. This includes learning rate, weight decay, etc.
        """
        super().__init__()
        # Exports the hyperparameters to a YAML file, and create "self.hparams" namespace
        self.save_hyperparameters()
        # Create model
        self.model = ResNet(**model_hparams)
        # Create loss module
        self.loss_module = nn.CrossEntropyLoss()
        # Example input for visualizing the graph in Tensorboard
        self.example_input_array = torch.zeros((1, 3, 450, 600), dtype=torch.float32)

    def forward(self, imgs):
        # Forward function that is run when visualizing the graph
        return self.model(imgs)

    def configure_optimizers(self):
        # We will support Adam or SGD as optimizers.
        if self.hparams.optimizer_name == "Adam":
            # AdamW is Adam with a correct implementation of weight decay (see here
            # for details: https://arxiv.org/pdf/1711.05101.pdf)
            optimizer = optim.AdamW(self.parameters(), **self.hparams.optimizer_hparams)
        elif self.hparams.optimizer_name == "SGD":
            optimizer = optim.SGD(self.parameters(), **self.hparams.optimizer_hparams)
        else:
            assert False, f'Unknown optimizer: "{self.hparams.optimizer_name}"'

        # We will reduce the learning rate by 0.1 after 100 and 150 epochs
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], gamma=0.1)
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        # "batch" is the output of the training data loader.
        imgs, labels = batch
        preds = self.model(imgs)
        loss = self.loss_module(preds, labels)
        acc = (preds.argmax(dim=-1) == labels).float().mean()

        # Logs the accuracy per epoch to tensorboard (weighted average over batches)
        self.log("train_acc", acc, on_step=False, on_epoch=True)
        self.log("train_loss", loss)
        return loss  # Return tensor to call ".backward" on

    def validation_step(self, batch, batch_idx):
        imgs, labels = batch
        preds = self.model(imgs).argmax(dim=-1)
        acc = (labels == preds).float().mean()
        # By default logs it per epoch (weighted average over batches)
        self.log("val_acc", acc)

    def test_step(self, batch, batch_idx):
        imgs, labels = batch
        preds = self.model(imgs).argmax(dim=-1)
        acc = (labels == preds).float().mean()
        # By default logs it per epoch (weighted average over batches), and returns it afterwards
        self.log("test_acc", acc)

    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = 0):
        imgs, labels = batch
        preds = self.model(imgs).argmax(dim=-1)
        return preds


def train_model(**kwargs):
    save_name = "ResNet"
    print("saving to ", os.path.join(CHECKPOINT_PATH, save_name))

    # Create a PyTorch Lightning trainer with the generation callback
    trainer = pl.Trainer(
        default_root_dir=os.path.join(CHECKPOINT_PATH, save_name),  # Where to save models
        # We run on a single GPU (if possible)
        accelerator="auto",
        devices=1,
        # How many epochs to train for if no patience is set
        max_epochs=180,
        callbacks=[
            ModelCheckpoint(
                save_weights_only=True, mode="max", monitor="val_acc"
            ),  # Save the best checkpoint based on the maximum val_acc recorded. Saves only weights and not optimizer]
        ],
        logger=wandb_logger
    )  # In case your notebook crashes due to the progress bar, consider increasing the refresh rate
    train_set = get_dataset("train", under_sampling=True, use_sample_probabilities=True)
    val_set = get_dataset("val")
    test_set = get_dataset("test")
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True, drop_last=True, pin_memory=False, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=32, shuffle=False, drop_last=False, num_workers=4)
    test_loader = DataLoader(test_set, batch_size=32, shuffle=False, drop_last=False, num_workers=4)

    model = ResNetModel()
    trainer.fit(model, train_loader, val_loader)
    model = ResNetModel.load_from_checkpoint(
        trainer.checkpoint_callback.best_model_path
    )  # Load best checkpoint after training

    # Test best model on validation and test set
    val_result = trainer.test(model, dataloaders=val_loader, verbose=False)
    test_result = trainer.test(model, dataloaders=test_loader, verbose=False)
    result = {"test": test_result[0]["test_acc"], "val": val_result[0]["test_acc"]}

    return model, result


if __name__ == "__main__":
    if torch.cuda.is_available():
        print("Using gpu for training")
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    resnet_model, resnet_results = train_model()

    print(resnet_results)
