import torch
from basic_vae_module import VAE
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

from dataset import train_set, test_set, val_set

wandb_logger = WandbLogger(project="bias-skin-lesion-detection")


def train_vae():
    trainer = Trainer(
        accelerator="auto",
        devices=1,
        max_epochs=100,
        callbacks=[
            ModelCheckpoint(
                save_weights_only=True, mode="max", monitor="val_acc"
            ),  # Save the best checkpoint based on the maximum val_acc recorded. Saves only weights and not optimizer]
        ],
        logger=wandb_logger
    )  # In case your notebook crashes due to the progress bar, consider increasing the refresh rate

    train_loader = DataLoader(train_set, batch_size=32, shuffle=True, drop_last=True, pin_memory=False, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=32, shuffle=False, drop_last=False, num_workers=4)
    test_loader = DataLoader(test_set, batch_size=32, shuffle=False, drop_last=False, num_workers=4)

    model = VAE(input_height=400, enc_type="resnet50", enc_out_dim=2048)
    trainer.fit(model, train_loader, val_loader)
    model = VAE.load_from_checkpoint(
        trainer.checkpoint_callback.best_model_path
    )

    val_result = trainer.test(model, dataloaders=val_loader, verbose=False)
    test_result = trainer.test(model, dataloaders=test_loader, verbose=False)
    result = {"test": test_result[0]["test_acc"], "val": val_result[0]["test_acc"]}

    return model, result


if __name__ == "__main__":
    if torch.cuda.is_available():
        print("Using gpu for training")
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    resnet_model, resnet_results = train_vae()

    print(resnet_results)
