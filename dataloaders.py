import torch
import torchvision
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from dataset import SkinLesionDataset

# Mean and Std with resize
# DATA_MEANS = torch.tensor([194.7129, 139.2769, 145.5000])
# DATA_STD = torch.tensor([35.1942, 37.9567, 42.1663])

DATA_MEANS = torch.tensor([194.7155, 139.2602, 145.4779])
DATA_STD = torch.tensor([36.0167, 38.9894, 43.4381])

test_transform = transforms.Compose([
    transforms.Normalize(DATA_MEANS, DATA_STD)])

train_transform = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop((650, 400), scale=(0.8, 1.0), ratio=(0.75, 1.33), antialias=True),
        transforms.Normalize(DATA_MEANS, DATA_STD),
    ]
)

train_set = SkinLesionDataset("./data/ISIC2018_Task3_Training_GroundTruth/ISIC2018_Task3_Training_GroundTruth"
                              ".csv", img_dir="./data/ISIC2018_Task3_Training_Input/", transform=train_transform)
test_set = SkinLesionDataset("./data/ISIC2018_Task3_Test_GroundTruth/ISIC2018_Task3_Test_GroundTruth.csv",
                             img_dir="./data/ISIC2018_Task3_Test_Input/", transform=test_transform)
val_set = SkinLesionDataset("./data/ISIC2018_Task3_Validation_GroundTruth"
                            "/ISIC2018_Task3_Validation_GroundTruth.csv",
                            img_dir="./data/ISIC2018_Task3_Validation_Input/", transform=test_transform)

train_loader = DataLoader(train_set, batch_size=32, shuffle=True, drop_last=True, pin_memory=False, num_workers=4)
val_loader = DataLoader(val_set, batch_size=32, shuffle=False, drop_last=False, num_workers=4)
test_loader = DataLoader(test_set, batch_size=32, shuffle=False, drop_last=False, num_workers=4)


def dataset_mean_and_std():
    # Adapted from: https://www.binarystudy.com/2022/04/how-to-normalize-image-dataset-inpytorch.html
    data_set = SkinLesionDataset("./data/ISIC2018_Task3_Training_GroundTruth/ISIC2018_Task3_Training_GroundTruth"
                                  ".csv", img_dir="./data/ISIC2018_Task3_Training_Input/")
    # transform=transforms.Compose([transforms.Resize((32, 32), antialias=True)]))
    data_loader = DataLoader(data_set, batch_size=64, shuffle=True, drop_last=True, pin_memory=False, num_workers=1)
    cnt = 0
    fst_moment = torch.empty(3)
    snd_moment = torch.empty(3)

    for images, _ in data_loader:
        b, c, h, w = images.shape
        nb_pixels = b * h * w
        sum_ = torch.sum(images, dim=[0, 2, 3])
        sum_of_square = torch.sum(images ** 2,
                                  dim=[0, 2, 3])
        fst_moment = (cnt * fst_moment + sum_) / (
                cnt + nb_pixels)
        snd_moment = (cnt * snd_moment + sum_of_square) / (
                cnt + nb_pixels)
        cnt += nb_pixels

    mean, std = fst_moment, torch.sqrt(
        snd_moment - fst_moment ** 2)

    print("mean and std: \n", mean, std)


def visualize_example_images():
    data_set = SkinLesionDataset("./data/ISIC2018_Task3_Training_GroundTruth/ISIC2018_Task3_Training_GroundTruth"
                                  ".csv", img_dir="./data/ISIC2018_Task3_Training_Input/")
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
