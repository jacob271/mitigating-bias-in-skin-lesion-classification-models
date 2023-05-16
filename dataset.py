import os

import torch
import torchvision
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from torchvision import transforms
from torchvision.io import read_image


class SkinLesionDataset(Dataset):
    def __init__(self, annotations_file, img_dir, metadata_file, transform=None, target_transform=None,
                 include_metadata=False, under_sampling=True):
        # under_sampling: if True, the dataset will be balanced by under-sampling the relevant classes

        self.include_metadata = include_metadata
        dataframe = pd.read_csv(annotations_file)
        discarded_classes = ['AKIEC', 'DF', 'VASC']
        relevant_classes = ['MEL', 'NV', 'BCC', 'BKL']
        for discarded_class in discarded_classes:
            dataframe = dataframe[dataframe[discarded_class] != 1.0]
            dataframe = dataframe.drop(columns=[discarded_class])
        dataframe = dataframe.reset_index(drop=True)
        if under_sampling:
            number_of_samples = dataframe[relevant_classes].sum(axis=0)
            min_samples = number_of_samples.min()
            for relevant_class in relevant_classes:
                other_rows = dataframe[dataframe[relevant_class] != 1.0]
                relevant_rows = dataframe[dataframe[relevant_class] == 1.0].head(int(min_samples))
                dataframe = pd.concat([other_rows, relevant_rows])

        metadata_sex = []
        metadata_age = []
        metadata = pd.read_csv(metadata_file)
        for isic_id in dataframe['image']:
            if len(metadata[metadata['isic_id'] == isic_id]['age_approx'].values) == 0:
                metadata_age.append(-10)
            else:
                metadata_age.append(metadata[metadata['isic_id'] == isic_id]['age_approx'].values[0])
            if len(metadata[metadata['isic_id'] == isic_id]['sex'].values) == 0:
                metadata_sex.append("unknown")
            else:
                metadata_sex.append(metadata[metadata['isic_id'] == isic_id]['sex'].values[0])

        dataframe['age'] = metadata_age
        dataframe['sex'] = metadata_sex

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
        label = self.img_labels.iloc[idx][1:5].astype(float).argmax()
        age = self.img_labels.iloc[idx]['age']
        sex = self.img_labels.iloc[idx]['sex']
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        if self.include_metadata:
            return image, [label, age, sex]
        return image, label


DATA_MEANS = torch.tensor([194.7155, 139.2602, 145.4779])
DATA_STD = torch.tensor([36.0167, 38.9894, 43.4381])

test_transform = transforms.Compose([transforms.CenterCrop((448, 448)), transforms.Normalize(DATA_MEANS, DATA_STD)])

train_transform = transforms.Compose([transforms.CenterCrop((448, 448)), transforms.RandomHorizontalFlip(),
                                      transforms.RandomResizedCrop((448, 448), scale=(0.8, 1.0), ratio=(0.75, 1.33),
                                                                   antialias=True),
                                      transforms.Normalize(DATA_MEANS, DATA_STD), ])

train_set = SkinLesionDataset("./data/ISIC2018_Task3_Training_GroundTruth/ISIC2018_Task3_Training_GroundTruth.csv",
                              img_dir="./data/ISIC2018_Task3_Training_Input/",
                              metadata_file="./data/ISIC2018_Task3_Training_GroundTruth/metadata.csv",
                              transform=train_transform)
train_set_with_metadata = SkinLesionDataset(
    "./data/ISIC2018_Task3_Test_GroundTruth/ISIC2018_Task3_Test_GroundTruth.csv",
    img_dir="./data/ISIC2018_Task3_Test_Input/", metadata_file="./data/ISIC2018_Task3_Test_GroundTruth/metadata.csv",
    include_metadata=True, transform=test_transform)
test_set = SkinLesionDataset("./data/ISIC2018_Task3_Test_GroundTruth/ISIC2018_Task3_Test_GroundTruth.csv",
                             img_dir="./data/ISIC2018_Task3_Test_Input/",
                             metadata_file="./data/ISIC2018_Task3_Test_GroundTruth/metadata.csv",
                             transform=test_transform, under_sampling=False)
test_set_with_metadata = SkinLesionDataset("./data/ISIC2018_Task3_Test_GroundTruth/ISIC2018_Task3_Test_GroundTruth.csv",
                                           img_dir="./data/ISIC2018_Task3_Test_Input/",
                                           metadata_file="./data/ISIC2018_Task3_Test_GroundTruth/metadata.csv",
                                           transform=test_transform, under_sampling=False, include_metadata=True)
val_set = SkinLesionDataset("./data/ISIC2018_Task3_Validation_GroundTruth/ISIC2018_Task3_Validation_GroundTruth.csv",
                            img_dir="./data/ISIC2018_Task3_Validation_Input/",
                            metadata_file="./data/ISIC2018_Task3_Validation_GroundTruth/metadata.csv",
                            transform=test_transform, under_sampling=False)
val_set_with_metadata = SkinLesionDataset(
    "./data/ISIC2018_Task3_Validation_GroundTruth/ISIC2018_Task3_Validation_GroundTruth.csv",
    img_dir="./data/ISIC2018_Task3_Validation_Input/",
    metadata_file="./data/ISIC2018_Task3_Validation_GroundTruth/metadata.csv", transform=test_transform,
    under_sampling=False, include_metadata=True)


def dataset_mean_and_std():
    # Adapted from: https://www.binarystudy.com/2022/04/how-to-normalize-image-dataset-inpytorch.html
    data_set = SkinLesionDataset("./data/ISIC2018_Task3_Training_GroundTruth/ISIC2018_Task3_Training_GroundTruth"
                                 ".csv", metadata_file="./data/ISIC2018_Task3_Training_GroundTruth/metadata.csv",
                                 img_dir="./data/ISIC2018_Task3_Training_Input/")
    # transform=transforms.Compose([transforms.Resize((32, 32), antialias=True)]))
    data_loader = DataLoader(data_set, batch_size=64, shuffle=True, drop_last=True, pin_memory=False, num_workers=1)
    cnt = 0
    fst_moment = torch.empty(3)
    snd_moment = torch.empty(3)

    for images, _ in data_loader:
        b, c, h, w = images.shape
        nb_pixels = b * h * w
        sum_ = torch.sum(images, dim=[0, 2, 3])
        sum_of_square = torch.sum(images ** 2, dim=[0, 2, 3])
        fst_moment = (cnt * fst_moment + sum_) / (cnt + nb_pixels)
        snd_moment = (cnt * snd_moment + sum_of_square) / (cnt + nb_pixels)
        cnt += nb_pixels

    mean, std = fst_moment, torch.sqrt(snd_moment - fst_moment ** 2)

    print("mean and std: \n", mean, std)


def visualize_example_images():
    data_set = SkinLesionDataset("./data/ISIC2018_Task3_Training_GroundTruth/ISIC2018_Task3_Training_GroundTruth"
                                 ".csv", metadata_file="./data/ISIC2018_Task3_Training_GroundTruth/metadata.csv",
                                 img_dir="./data/ISIC2018_Task3_Training_Input/")
    num_images = 4
    images = [data_set[idx][0] / 255.0 for idx in range(num_images)]
    orig_images = [data_set[idx][0] for idx in range(num_images)]
    orig_images = [train_transform(img) for img in orig_images]

    img_grid = torchvision.utils.make_grid(torch.stack(images + orig_images, dim=0), nrow=4, pad_value=0.5)
    img_grid = img_grid.permute(1, 2, 0)

    plt.figure(figsize=(8, 8))
    plt.title("Augmentation examples on training data")
    plt.imshow(img_grid)
    plt.axis("off")
    plt.show()
    plt.close()
