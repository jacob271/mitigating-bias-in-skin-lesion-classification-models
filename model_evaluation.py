import math

import torch
import torchmetrics.classification
from torch.utils.data import DataLoader
from torchmetrics import ConfusionMatrix

from dataset import val_set_with_metadata
from main import ResNet50Model
import pytorch_lightning as pl

import matplotlib.pyplot as plt
import seaborn as sns
import wandb

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

model = ResNet50Model.load_from_checkpoint('charlie4.cpkt', map_location=device)

trainer = pl.Trainer(
    # We run on a single GPU (if possible)
    accelerator="auto",
    devices=1,
    # How many epochs to train for if no patience is set
    max_epochs=10,
    logger=False,
)


test_loader = DataLoader(val_set_with_metadata, batch_size=1, shuffle=False, drop_last=False, num_workers=1)

all_labels = []
for batch in test_loader:
    imgs, labels = batch
    all_labels.append(labels)

print(all_labels)
predictions = trainer.predict(model, test_loader)
predictions = torch.cat(predictions)
print(predictions)

confm = ConfusionMatrix(task="multiclass", num_classes=4)

confm_labels = []
for label in all_labels:
    confm_labels.append(label[0])

confm_labels = torch.cat(confm_labels)

result = confm(predictions, confm_labels)

labels = ["MEL", "NV", "BCC", "BKL"]
sns.heatmap(result, annot=True, cmap="Blues", xticklabels=labels, yticklabels=labels)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.savefig("conf_ma.png")

wandb.log({"confma": wandb.Image("conf_ma.png")})

print(confm(predictions, confm_labels))

# calculate accuracy based on sex
male_predictions = []
male_labels = []
female_predictions = []
female_labels = []

for i in range(len(predictions)):
    if all_labels[i][2][0] == 'male':
        male_predictions.append(torch.unsqueeze(predictions[i], dim=0))
        male_labels.append(all_labels[i][0])
    elif all_labels[i][2][0] == 'female':
        female_predictions.append(torch.unsqueeze(predictions[i], dim=0))
        female_labels.append(all_labels[i][0])
    else:
        print(f"unknown: {all_labels[i][2]}")

print(male_labels)
print(male_predictions)

metric = torchmetrics.classification.MulticlassAccuracy(num_classes=4, average='weighted')

male_accuracy = metric(torch.cat(male_predictions), torch.cat(male_labels))
female_accuracy = metric(torch.cat(female_predictions), torch.cat(female_labels))

overall_accuracy = metric(predictions, confm_labels)

print(f"male_acc: {male_accuracy}")
print(f"female_acc: {female_accuracy}")

bias = (math.pow(male_accuracy - overall_accuracy, 2) + math.pow(female_accuracy - overall_accuracy, 2)) / 2

print(f"bias: {bias}")
