import os

from pytorch_lightning.callbacks import ModelCheckpoint
import pytorch_lightning as pl

from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

from dataset import get_dataset
from resnet_module import ResNetModule

wandb_logger = WandbLogger(project="bias-skin-lesion-detection")

classes = ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"]

CHECKPOINT_PATH = os.environ.get("PATH_CHECKPOINT", "./saved_models")
os.makedirs(CHECKPOINT_PATH, exist_ok=True)


def train_model():
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
    val_set = get_dataset("validation")
    test_set = get_dataset("test")
    train_loader = DataLoader(train_set, batch_size=32, shuffle=False, drop_last=False, pin_memory=False, num_workers=1)
    val_loader = DataLoader(val_set, batch_size=32, shuffle=False, drop_last=False, num_workers=4)
    test_loader = DataLoader(test_set, batch_size=32, shuffle=False, drop_last=False, num_workers=4)

    model = ResNetModule()
    trainer.fit(model, train_loader, val_loader)
    model = ResNetModule.load_from_checkpoint(
        trainer.checkpoint_callback.best_model_path
    )  # Load best checkpoint after training

    # Test best model on validation and test set
    val_result = trainer.test(model, dataloaders=val_loader, verbose=False)
    test_result = trainer.test(model, dataloaders=test_loader, verbose=False)
    result = {"test": test_result[0]["test_acc"], "val": val_result[0]["test_acc"]}

    return model, result


if __name__ == "__main__":
    resnet_model, resnet_results = train_model()
    print(resnet_results)
