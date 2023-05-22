import torch
from basic_vae_module import VAE
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

from dataset import get_dataset

wandb_logger = WandbLogger(project="bias-skin-lesion-detection")


def train_vae():
    trainer = Trainer(
        accelerator="auto",
        devices=1,
        max_epochs=100,
        callbacks=[
            ModelCheckpoint(
                save_weights_only=True, mode="min", monitor="val_loss"
            ),
        ],
        logger=wandb_logger
    )  # In case your notebook crashes due to the progress bar, consider increasing the refresh rate
    train_set = get_dataset("train", under_sampling=True)
    val_set = get_dataset("validation")
    test_set = get_dataset("test")
    train_loader = DataLoader(train_set, batch_size=12, shuffle=True, drop_last=True, pin_memory=False, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=12, shuffle=False, drop_last=False, num_workers=4)
    test_loader = DataLoader(test_set, batch_size=12, shuffle=False, drop_last=False, num_workers=4)

    model = VAE(input_height=360, enc_type="resnet18", enc_out_dim=512)
    trainer.fit(model, train_loader, val_loader)
    model = VAE.load_from_checkpoint(
        trainer.checkpoint_callback.best_model_path
    )

    val_result = trainer.test(model, dataloaders=val_loader, verbose=False)
    test_result = trainer.test(model, dataloaders=test_loader, verbose=False)

    return model, {"val": val_result, "test": test_result}


if __name__ == "__main__":
    if torch.cuda.is_available():
        print("Using gpu for training")
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    resnet_model, resnet_results = train_vae()

    print(resnet_results)
