import os

import torch
from torch.utils.data import Dataset
import pandas as pd
from torchvision.io import read_image


class SkinLesionDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        dataframe = pd.read_csv(annotations_file)
        discarded_classes = ['AKIEC', 'DF', 'VASC']
        relevant_classes = ['MEL', 'NV', 'BCC', 'BKL']
        for discarded_class in discarded_classes:
            dataframe = dataframe[dataframe[discarded_class] != 1.0]
            dataframe = dataframe.drop(columns=[discarded_class])
        dataframe = dataframe.reset_index(drop=True)
        number_of_samples = dataframe[relevant_classes].sum(axis=0)
        min_samples = number_of_samples.min()
        for relevant_class in relevant_classes:
            other_rows = dataframe[dataframe[relevant_class] != 1.0]
            relevant_rows = dataframe[dataframe[relevant_class] == 1.0].head(int(min_samples))
            dataframe = pd.concat([other_rows, relevant_rows])

        self.img_labels = dataframe

        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0] + ".jpg")
        image = read_image(img_path)
        image = image.to(torch.float32)
        label = self.img_labels.iloc[idx][1:].astype(float).argmax()
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

