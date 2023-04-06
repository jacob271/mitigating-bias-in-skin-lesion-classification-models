import os

import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from torchvision.io import read_image
import matplotlib.pyplot as plt
import numpy as np
import lightning as pl
from torchvision.transforms import transforms

classes = ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"]


class SkinLesionDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0] + ".jpg")
        image = read_image(img_path)
        label = self.img_labels.iloc[idx][1:].astype(float).argmax()
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


if __name__ == "__main__":
    train_dataset = SkinLesionDataset("./data/ISIC2018_Task3_Training_GroundTruth/ISIC2018_Task3_Training_GroundTruth"
                                      ".csv", img_dir="./data/ISIC2018_Task3_Training_Input/")
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_dataset = SkinLesionDataset("./data/ISIC2018_Task3_Test_GroundTruth/ISIC2018_Task3_Test_GroundTruth.csv",
                                     img_dir="./data/ISIC2018_Task3_Test_Input/")
    validation_dataset = SkinLesionDataset("./data/ISIC2018_Task3_Validation_GroundTruth"
                                           "/ISIC2018_Task3_Validation_GroundTruth.csv",
                                           img_dir="./data/ISIC2018_Task3_Validation_Input/")
    # Display image and label.
    train_features, train_labels = next(iter(train_dataloader))
    print(f"Feature batch shape: {train_features.size()}")
    print(f"Labels batch shape: {train_labels.size()}")
    img = train_features[0].squeeze()
    img = np.transpose(img, (1, 2, 0))
    label = train_labels[0]
    print(f"Label: {classes[label]}")
    plt.imshow(img)
    plt.show()
    print(f"Label: {label}")
