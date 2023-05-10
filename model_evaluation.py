import torch
from torch.utils.data import DataLoader
from torchmetrics import ConfusionMatrix

from dataloaders import test_set
from main import SkinLesionModule
import pytorch_lightning as pl

import matplotlib.pyplot as plt
import seaborn as sns

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

test_loader = DataLoader(test_set, batch_size=1, shuffle=False, drop_last=False, num_workers=1)

all_labels = []
for batch in test_loader:
    imgs, labels = batch
    all_labels.append(labels)

all_labels = torch.cat(all_labels)

print(all_labels)
predictions = trainer.predict(model, test_loader)
predictions = torch.cat(predictions)
print(predictions)

confm = ConfusionMatrix(task="multiclass", num_classes=7)

result = confm(predictions, all_labels)

labels = ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"]
sns.heatmap(result, annot=True, cmap="Blues", xticklabels=labels, yticklabels=labels)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()

print(confm(predictions, all_labels))
