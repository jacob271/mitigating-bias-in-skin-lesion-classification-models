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
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    # DATA_MEANS = (train_dataset.data / 255.0).mean(axis=(0, 1, 2))
    # DATA_STD = (train_dataset.data / 255.0).std(axis=(0, 1, 2))
    # print("Data mean", DATA_MEANS)
    # print("Data std", DATA_STD)
    test_transform = transforms.Compose([])
    # For training, we add some augmentation. Networks are too powerful and would overfit.
    train_transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop((450, 600), scale=(0.8, 1.0), ratio=(0.75, 1.33)),
        ]
    )

    NUM_IMAGES = 4
    images = [train_dataset[idx][0] for idx in range(NUM_IMAGES)]
    to_pil = transforms.ToPILImage()
    orig_images = [train_dataset[idx][0] for idx in range(NUM_IMAGES)]
    orig_images = [train_transform(img) for img in orig_images]

    img_grid = torchvision.utils.make_grid(torch.stack(images + orig_images, dim=0), nrow=4, pad_value=0.5)
    img_grid = img_grid.permute(1, 2, 0)

    plt.figure(figsize=(8, 8))
    plt.title("Augmentation examples on training data")
    plt.imshow(img_grid)
    plt.axis("off")
    plt.show()
    plt.close()


