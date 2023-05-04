import os

import torch
import torchvision
from lightning.pytorch.callbacks import ModelCheckpoint
from matplotlib import pyplot as plt
from torch import optim
from torch.utils.data import DataLoader
import torch.nn as nn
import lightning as pl
from torchvision.transforms import transforms

from lightning.pytorch.loggers import WandbLogger

from dataset import SkinLesionDataset
from model import ResNet

wandb_logger = WandbLogger(project="bias-skin-lesion-detection")

classes = ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"]


CHECKPOINT_PATH = os.environ.get("PATH_CHECKPOINT", "saved_models/SkinLesionNets")
os.makedirs(CHECKPOINT_PATH, exist_ok=True)


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

    # Check whether pretrained model exists. If yes, load it and skip training
    pretrained_filename = os.path.join(CHECKPOINT_PATH, save_name + ".ckpt")
    if os.path.isfile(pretrained_filename):
        print(f"Found pretrained model at {pretrained_filename}, loading...")
        # Automatically loads the model with the saved hyperparameters
        model = SkinLesionModule.load_from_checkpoint(pretrained_filename)
    else:
        # pl.seed_everything(42)  # To be reproducable
        model = SkinLesionModule(**kwargs)
        trainer.fit(model, train_loader, val_loader)
        model = SkinLesionModule.load_from_checkpoint(
            trainer.checkpoint_callback.best_model_path
        )  # Load best checkpoint after training

    # Test best model on validation and test set
    val_result = trainer.test(model, dataloaders=val_loader, verbose=False)
    test_result = trainer.test(model, dataloaders=test_loader, verbose=False)
    result = {"test": test_result[0]["test_acc"], "val": val_result[0]["test_acc"]}

    return model, result


def visualize_example_images():
    train_set = SkinLesionDataset("./data/ISIC2018_Task3_Training_GroundTruth/ISIC2018_Task3_Training_GroundTruth"
                                  ".csv", img_dir="./data/ISIC2018_Task3_Training_Input/")
                                  #transform=transforms.Compose([transforms.Resize((32, 32), antialias=True)]))
    NUM_IMAGES = 4
    images = [train_set[idx][0] / 255.0 for idx in range(NUM_IMAGES)]
    orig_images = [train_set[idx][0] for idx in range(NUM_IMAGES)]
    orig_images = [train_transform(img) for img in orig_images]

    img_grid = torchvision.utils.make_grid(torch.stack(images + orig_images, dim=0), nrow=4, pad_value=0.5)
    img_grid = img_grid.permute(1, 2, 0)

    plt.figure(figsize=(8, 8))
    plt.title("Augmentation examples on training data")
    plt.imshow(img_grid)
    plt.axis("off")
    plt.show()
    plt.close()


def dataset_mean_and_std():
    # Adapted from: https://www.binarystudy.com/2022/04/how-to-normalize-image-dataset-inpytorch.html
    train_set = SkinLesionDataset("./data/ISIC2018_Task3_Training_GroundTruth/ISIC2018_Task3_Training_GroundTruth"
                                  ".csv", img_dir="./data/ISIC2018_Task3_Training_Input/")
                                  # transform=transforms.Compose([transforms.Resize((32, 32), antialias=True)]))
    train_loader = DataLoader(train_set, batch_size=8, shuffle=True, drop_last=True, pin_memory=False, num_workers=1)
    cnt = 0
    fst_moment = torch.empty(3)
    snd_moment = torch.empty(3)

    for images, _ in train_loader:
        b, c, h, w = images.shape
        nb_pixels = b * h * w
        sum_ = torch.sum(images, dim=[0, 2, 3])
        sum_of_square = torch.sum(images ** 2,
                                  dim=[0, 2, 3])
        fst_moment = (cnt * fst_moment + sum_) / (
                cnt + nb_pixels)
        snd_moment = (cnt * snd_moment + sum_of_square) / (
                cnt + nb_pixels)
        cnt += nb_pixels

    mean, std = fst_moment, torch.sqrt(
        snd_moment - fst_moment ** 2)

    print("mean and std: \n", mean, std)


if __name__ == "__main__":
    if torch.cuda.is_available():
        print("Using gpu for training")
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    # Mean and Std without resize
    # DATA_MEANS = torch.tensor([194.6954, 139.2556, 145.4731])
    # DATA_STD = torch.tensor([36.0131, 38.9913, 43.4326])

    DATA_MEANS = torch.tensor([194.7129, 139.2769, 145.5000])
    DATA_STD = torch.tensor([35.1942, 37.9567, 42.1663])
    print("Data mean", DATA_MEANS)
    print("Data std", DATA_STD)

    test_transform = transforms.Compose([
        #transforms.Resize((32, 32), antialias=True),
        transforms.Normalize(DATA_MEANS, DATA_STD)])
    # For training, we add some augmentation. Networks are too powerful and would overfit.
    train_transform = transforms.Compose(
        [
            #transforms.Resize((32, 32), antialias=True),
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop((650, 400), scale=(0.8, 1.0), ratio=(0.75, 1.33), antialias=True),
            transforms.Normalize(DATA_MEANS, DATA_STD),
        ]
    )

    train_set = SkinLesionDataset("./data/ISIC2018_Task3_Training_GroundTruth/ISIC2018_Task3_Training_GroundTruth"
                                  ".csv", img_dir="./data/ISIC2018_Task3_Training_Input/", transform=train_transform)
    test_set = SkinLesionDataset("./data/ISIC2018_Task3_Test_GroundTruth/ISIC2018_Task3_Test_GroundTruth.csv",
                                 img_dir="./data/ISIC2018_Task3_Test_Input/", transform=test_transform)
    val_set = SkinLesionDataset("./data/ISIC2018_Task3_Validation_GroundTruth"
                                "/ISIC2018_Task3_Validation_GroundTruth.csv",
                                img_dir="./data/ISIC2018_Task3_Validation_Input/", transform=test_transform)

    train_loader = DataLoader(train_set, batch_size=8, shuffle=True, drop_last=True, pin_memory=False, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=8, shuffle=False, drop_last=False, num_workers=4)
    test_loader = DataLoader(test_set, batch_size=8, shuffle=False, drop_last=False, num_workers=4)

    resnet_model, resnet_results = train_model(
        model_hparams={"num_classes": 7, "c_hidden": [16, 32, 64], "num_blocks": [3, 3, 3], "act_fn_name": "relu"},
        optimizer_name="SGD",
        optimizer_hparams={"lr": 0.1, "momentum": 0.9, "weight_decay": 1e-4},
    )

    print(resnet_results)
