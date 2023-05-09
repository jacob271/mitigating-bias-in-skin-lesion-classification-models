import torch

from dataloaders import val_loader, test_loader
from main import SkinLesionModule
import pytorch_lightning as pl

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

model = SkinLesionModule.load_from_checkpoint('checkpoint1.cpkt', map_location=device)

trainer = pl.Trainer(
    # We run on a single GPU (if possible)
    accelerator="auto",
    devices=1,
    # How many epochs to train for if no patience is set
    max_epochs=10,
    logger=False
)

val_result = trainer.test(model, dataloaders=val_loader, verbose=True)
test_result = trainer.test(model, dataloaders=test_loader, verbose=True)
result = {"test": test_result[0]["test_acc"], "val": val_result[0]["test_acc"]}

print(result)
