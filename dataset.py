import os

import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd
from torchvision import transforms
from torchvision.io import read_image


class SkinLesionDataset(Dataset):
    def __init__(self, annotations_file, img_dir, metadata_file, transform=None, target_transform=None,
                 include_metadata=False, under_sampling=True, id_as_label=False, sample_probabilities_file="",
                 metadata_hairiness_file="",
                 metadata_skin_tone_file=""):
        # under_sampling: if True, the dataset will be balanced by under-sampling the relevant classes

        self.id_as_label = id_as_label
        self.include_metadata = include_metadata
        dataframe = pd.read_csv(annotations_file)
        self.use_sample_probabilities = False
        if sample_probabilities_file:
            self.use_sample_probabilities = True
        self.sample_probabilities = None

        self.num_classes = 2
        discarded_classes = ['AKIEC', 'DF', 'VASC']
        relevant_classes = ['MEL', 'NV', 'BCC', 'BKL']
        if self.num_classes == 2:
            discarded_classes = ['AKIEC', 'DF', 'VASC', 'BCC', 'BKL']
            relevant_classes = ['MEL', 'NV']

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
            dataframe = dataframe.reset_index(drop=True)

        # Set sample probabilities
        if self.use_sample_probabilities:
            sp_df = pd.read_csv(sample_probabilities_file, index_col=False)
            sp_df = sp_df[sp_df['isic_id'].isin(dataframe['image'])]
            sp_df = sp_df.drop(sp_df.columns[0], axis=1)
            sp_df = sp_df.reset_index(drop=True)
            self.sample_probabilities = sp_df

        metadata_sex = []
        metadata_age = []
        metadata_hairiness = []
        metadata_skin_tone = []
        metadata = pd.read_csv(metadata_file)
        if metadata_hairiness_file:
            metadata_hairiness_df = pd.read_csv(metadata_hairiness_file)
        else:
            metadata_hairiness_df = pd.DataFrame(columns=['isic_id', 'hair_density', 'high_hair_density'])
        if metadata_skin_tone_file:
            metadata_skin_tone_df = pd.read_csv(metadata_skin_tone_file)
        else:
            metadata_skin_tone_df = pd.DataFrame(columns=['isic_id', 'skin_tone'])
        for isic_id in dataframe['image']:
            if len(metadata[metadata['isic_id'] == isic_id]['age_approx'].values) == 0:
                metadata_age.append(-10)
            else:
                metadata_age.append(metadata[metadata['isic_id'] == isic_id]['age_approx'].values[0])
            if len(metadata[metadata['isic_id'] == isic_id]['sex'].values) == 0:
                metadata_sex.append("unknown")
            else:
                metadata_sex.append(metadata[metadata['isic_id'] == isic_id]['sex'].values[0])
            
            if len(metadata_hairiness_df[metadata_hairiness_df['isic_id'] == isic_id]['high_hair_density'].values) == 0:
                metadata_hairiness.append(-1)
            else:
                metadata_hairiness.append(int(metadata_hairiness_df[metadata_hairiness_df['isic_id'] == isic_id]['high_hair_density'].values[0]))

            if len(metadata_skin_tone_df[metadata_skin_tone_df['isic_id'] == isic_id]['skin_tone'].values) == 0:
                metadata_skin_tone.append("Other")
            else:
                metadata_skin_tone.append(metadata_skin_tone_df[metadata_skin_tone_df['isic_id'] == isic_id]['skin_tone'].values[0])

        dataframe['age'] = metadata_age
        dataframe['sex'] = metadata_sex
        dataframe['high_hair_density'] = metadata_hairiness
        dataframe['skin_tone'] = metadata_skin_tone

        self.img_labels = dataframe

        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        if self.use_sample_probabilities:
            img_id1 = self.img_labels.iloc[idx, 0]
            img_id2 = self.sample_probabilities.iloc[idx, 0]
            if img_id1 != img_id2:
                print(self.img_labels.iloc[idx, 0])
                print(self.sample_probabilities[idx, 0])
                print("ERROR")
                raise Exception("Sample probabilities and image labels are not aligned")
            idx = self.get_random_index()
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0] + ".jpg")
        image = read_image(img_path)
        image = image.to(torch.float32)
        label = self.img_labels.iloc[idx][1:(self.num_classes + 1)].astype(float).argmax()
        if self.id_as_label:
            label = [self.img_labels.iloc[idx][0], label]
        age = self.img_labels.iloc[idx]['age']
        sex = self.img_labels.iloc[idx]['sex']
        hairiness = self.img_labels.iloc[idx]['high_hair_density']
        skin_tone = self.img_labels.iloc[idx]['skin_tone']
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        if self.include_metadata:
            return image, [label, age, sex, hairiness, skin_tone]
        return image, label

    def get_image_from_isic_id(self, isic_id):
        img_path = os.path.join(self.img_dir, isic_id + ".jpg")
        image = read_image(img_path)
        image = image.to(torch.float32)
        if self.transform:
            image = self.transform(image)
        return image

    def get_random_index(self):
        random_number = np.random.uniform(0, 1)
        cumulative_probabilities = np.cumsum(self.sample_probabilities['sample_probability'].values)
        selected_index = np.searchsorted(cumulative_probabilities, random_number)
        return selected_index


DATA_MEANS = torch.tensor([192.1314, 141.6559, 147.6526])
DATA_STD = torch.tensor([33.1019, 38.3609, 43.3686])

test_transform = transforms.Compose([
    transforms.CenterCrop((450, 450)),
    transforms.Resize((360, 360), antialias=False),
    transforms.Normalize(DATA_MEANS, DATA_STD),
])

plain_transform = transforms.Compose([
    transforms.CenterCrop((450, 450)),
    transforms.Resize((360, 360), antialias=False),
])

train_transform = transforms.Compose([
    transforms.CenterCrop((450, 450)),
    transforms.Resize((360, 360), antialias=False),
    transforms.RandomHorizontalFlip(),
    transforms.Normalize(DATA_MEANS, DATA_STD),
])


def get_dataset(dataset_name, include_metadata=False, under_sampling=False, id_as_label=False,
                use_sample_probabilities=False, use_plain_transform=False):
    metadata_hairiness_file = ""
    metadata_skin_tone_file = ""
    if dataset_name == "train":
        img_dir = "./data/ISIC2018_Task3_Training_Input/"
        metadata_file = "./data/ISIC2018_Task3_Training_GroundTruth/metadata.csv"
        csv_file = "./data/ISIC2018_Task3_Training_GroundTruth/ISIC2018_Task3_Training_GroundTruth.csv"
        sample_probabilities_file = "./data/ISIC2018_Task3_Training_GroundTruth/binary_sample_probabilities.csv"
        transform = train_transform
    elif dataset_name == "test":
        img_dir = "./data/ISIC2018_Task3_Test_Input/"
        metadata_file = "./data/ISIC2018_Task3_Test_GroundTruth/metadata.csv"
        csv_file = "./data/ISIC2018_Task3_Test_GroundTruth/ISIC2018_Task3_Test_GroundTruth.csv"
        sample_probabilities_file = ""
        metadata_hairiness_file = "./data/ISIC2018_Task3_Test_GroundTruth/binary_hair_densities.csv"
        metadata_skin_tone_file = "./data/ISIC2018_Task3_Test_GroundTruth/binary_skin_tones.csv"
        transform = test_transform
    elif dataset_name == "validation":
        img_dir = "./data/ISIC2018_Task3_Validation_Input/"
        metadata_file = "./data/ISIC2018_Task3_Validation_GroundTruth/metadata.csv"
        csv_file = "./data/ISIC2018_Task3_Validation_GroundTruth/ISIC2018_Task3_Validation_GroundTruth.csv"
        sample_probabilities_file = ""
        transform = test_transform
    else:
        raise ValueError("Invalid dataset name.")

    if not use_sample_probabilities:
        sample_probabilities_file = ""

    if use_plain_transform:
        transform = plain_transform

    return SkinLesionDataset(csv_file, img_dir=img_dir, metadata_file=metadata_file, transform=transform,
                             include_metadata=include_metadata, under_sampling=under_sampling, id_as_label=id_as_label,
                             sample_probabilities_file=sample_probabilities_file,
                             metadata_hairiness_file=metadata_hairiness_file,
                             metadata_skin_tone_file=metadata_skin_tone_file)
