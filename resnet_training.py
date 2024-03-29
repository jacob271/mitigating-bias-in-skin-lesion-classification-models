import argparse
import os

from pytorch_lightning.callbacks import ModelCheckpoint
import pytorch_lightning as pl
import torchmetrics.classification
from torchmetrics import ConfusionMatrix
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
import wandb
import torch
import matplotlib.pyplot as plt
import seaborn as sns

from statistics import variance

from dataset import get_dataset
from resnet_module import ResNetModel

wandb_logger = WandbLogger(project="bias-skin-lesion-detection")

classes = ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"]

CHECKPOINT_PATH = os.environ.get("PATH_CHECKPOINT", "./saved_models")
os.makedirs(CHECKPOINT_PATH, exist_ok=True)


def train_resnet(debiasing=False, num_classes=2, transfer_learning=False, num_epochs=20, batch_size=32, lr=1e-4):
    save_name = "ResNet"
    print("saving to ", os.path.join(CHECKPOINT_PATH, save_name))

    trainer = pl.Trainer(
        default_root_dir=os.path.join(CHECKPOINT_PATH, save_name),
        accelerator="gpu",
        devices=[2],
        max_epochs=num_epochs,
        callbacks=[
            ModelCheckpoint(
                save_weights_only=True, save_last=True
                # save_weights_only=True, mode="max", monitor="val_acc"
            ),  # Save the best checkpoint based on the maximum val_acc recorded
        ],
        logger=wandb_logger
    )
    train_set = get_dataset("train", under_sampling=True, use_sample_probabilities=debiasing, num_classes=num_classes)
    val_set = get_dataset("validation", num_classes=num_classes)
    test_set = get_dataset("test", num_classes=num_classes)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=(not debiasing), drop_last=False, pin_memory=False, num_workers=1)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=4)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=4)

    model = ResNetModel(num_classes=num_classes, lr=lr, transfer_learning=transfer_learning)
    trainer.fit(model, train_loader, val_loader)
    model = ResNetModel.load_from_checkpoint(
        trainer.checkpoint_callback.best_model_path
    )  # Load best checkpoint after training

    # Test best model on validation and test set
    val_result = trainer.validate(model, dataloaders=val_loader, verbose=False)
    test_result = trainer.test(model, dataloaders=test_loader, verbose=False)
    result = {"test": test_result, "val": val_result}

    return model, result


def get_predictions(model, data_set_name="test", num_classes=2):
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)
    
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=[0],
        logger=False,
    )

    data_set_with_metadata = get_dataset(data_set_name, include_metadata=True, num_classes=num_classes)
    data_loader = DataLoader(data_set_with_metadata, batch_size=1, shuffle=False, drop_last=False, num_workers=0)

    all_labels = []
    for batch in data_loader:
        imgs, labels = batch
        all_labels.append(labels)

    predictions = trainer.predict(model, data_loader)
    predictions = torch.cat(predictions)
    return predictions, all_labels


def plot_confusion_matrix(predictions, all_labels, file_path="conf_ma.png", num_classes=4):
    class_labels = []
    for label in all_labels:
        class_labels.append(label[0])
    class_labels = torch.cat(class_labels)
    
    confm = ConfusionMatrix(task="multiclass", num_classes=num_classes)
    result = confm(predictions, class_labels)

    if num_classes == 4:
        labels = ["MEL", "NV", "BCC", "BKL"]
    else:
        labels = ["MEL", "NV"]
    sns.heatmap(result, annot=True, cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.savefig(file_path)

    
def calculate_gender_bias(predictions, all_labels, metric, num_classes=4):
    male_predictions = []
    male_labels = []
    female_predictions = []
    female_labels = []
    unknown_count = 0
    
    for i in range(len(predictions)):
        if all_labels[i][2][0] == 'male':
            male_predictions.append(torch.unsqueeze(predictions[i], dim=0))
            male_labels.append(all_labels[i][0])
        elif all_labels[i][2][0] == 'female':
            female_predictions.append(torch.unsqueeze(predictions[i], dim=0))
            female_labels.append(all_labels[i][0])
        else:
            unknown_count += 1
    print(f"Observed {unknown_count} labels out of {len(predictions)} to be unknown")

    male_accuracy = metric(torch.cat(male_predictions), torch.cat(male_labels)).item()
    female_accuracy = metric(torch.cat(female_predictions), torch.cat(female_labels)).item()
    bias = variance([male_accuracy, female_accuracy])
    results = {"male_acc": male_accuracy, "female_acc": female_accuracy, "gender_bias": bias}
    print(f"male_acc: {male_accuracy}")
    print(f"female_acc: {female_accuracy}")
    print(f"bias: {bias}")
    return results


def calculate_age_bias(predictions, all_labels, metric, num_classes=4):
    age_groups = ["upto30", "35to55", "60up", "unknown"]
    age_based_predictions = {"upto30": [], "35to55": [], "60up": [], "unknown": []}
    age_labels = {"upto30": [], "35to55": [], "60up": [], "unknown": []}
    
    unknown_counter = 0
    for i in range(len(predictions)):
        age = all_labels[i][1][0].item()
        if age <= 0.0:
            age_group = "unknown"
            unknown_counter += 1
        if age <= 30.0:
            age_group = "upto30"
        elif age <= 55.0:
            age_group = "35to55"
        else:
            age_group = "60up"

        age_based_predictions[age_group].append(torch.unsqueeze(predictions[i], dim=0))
        age_labels[age_group].append(all_labels[i][0])
        
    print(f"Observed {unknown_counter} ages out of {len(predictions)} to be unknown")
    
    accuracies = {}
    acc_list = []
    for age_group in age_groups:
        print(f"{age_group} has {len(age_based_predictions[age_group])} samples")
        if age_group == "unknown":
            continue
        if len(age_based_predictions[age_group]) > 0:
            accuracies[age_group] = metric(torch.cat(age_based_predictions[age_group]), torch.cat(age_labels[age_group])).item()
            acc_list.append(accuracies[age_group])
        else:
            print("WARNING: No samples for this age group")

    print(f"age_accuracies: {accuracies}")
    print(f"age_bias: {variance(acc_list)}")
    results = {"age_accuracies": accuracies, "age_bias": variance(acc_list)}
    return results


def calculate_hairiness_bias(predictions, all_labels, metric, num_classes=4):
    high_density_predictions = []
    high_density_labels = []
    low_density_predictions = []
    low_density_labels = []
    unknown_count = 0

    for i in range(len(predictions)):
        if all_labels[i][3][0] == 1:
            high_density_predictions.append(torch.unsqueeze(predictions[i], dim=0))
            high_density_labels.append(all_labels[i][0])
        elif all_labels[i][3][0] == 0:
            low_density_predictions.append(torch.unsqueeze(predictions[i], dim=0))
            low_density_labels.append(all_labels[i][0])
        else:
            unknown_count += 1
    print(f"Observed {unknown_count} labels out of {len(predictions)} to be unknown")

    high_density_accuracy = metric(torch.cat(high_density_predictions), torch.cat(high_density_labels)).item()
    low_density_accuracy = metric(torch.cat(low_density_predictions), torch.cat(low_density_labels)).item()
    bias = variance([high_density_accuracy, low_density_accuracy])

    results = {"high_density_acc": high_density_accuracy, "low_density_acc": low_density_accuracy, "hairiness_bias": bias}
    print(f"high_density_acc: {high_density_accuracy}")
    print(f"low_density_acc: {low_density_accuracy}")
    print(f"hairiness_bias: {bias}")
    return results


def calculate_skin_tone_bias(predictions, all_labels, metric, num_classes=4):
    skin_types = ["Type I", "Type II", "Type III", "Other"]
    type_based_predictions = {"Type I": [], "Type II": [], "Type III": [], "Other": []}
    type_based_labels = {"Type I": [], "Type II": [], "Type III": [], "Other": []}

    unknown_counter = 0
    for i in range(len(predictions)):
        skin_type = all_labels[i][4][0]
        if skin_type == "Other":
            unknown_counter += 1

        type_based_predictions[skin_type].append(torch.unsqueeze(predictions[i], dim=0))
        type_based_labels[skin_type].append(all_labels[i][0])

    print(f"Observed {unknown_counter} skin tones out of {len(predictions)} to be 'Other'")

    accuracies = {}
    acc_list = []
    for skin_type in skin_types:
        print(f"{skin_type} has {len(type_based_predictions[skin_type])} samples")
        if skin_type == "Other":
            continue
        if len(type_based_predictions[skin_type]) > 0:
            accuracies[skin_type] = metric(torch.cat(type_based_predictions[skin_type]), torch.cat(type_based_labels[skin_type])).item()
            acc_list.append(accuracies[skin_type])
        else:
            print("WARNING: No samples for this skin tone group")

    print(f"skin_accuracies: {accuracies}")
    print(f"skin_tone_bias: {variance(acc_list)}")
    results = {"skin_accuracies": accuracies, "skin_tone_bias": variance(acc_list)}
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_debiasing', type=bool, default=False, help='use debiasing')
    parser.add_argument('--num_classes', type=int, default=2, help='number of classes')
    parser.add_argument('--transfer_learning', type=bool, default=False, help='use transfer learning')
    parser.add_argument('--num_epochs', type=int, default=20, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    args = parser.parse_args()

    wandb.config.debiasing=args.use_debiasing
    wandb.config.pretrained = args.transfer_learning
    num_classes = args.num_classes
    metric = torchmetrics.classification.MulticlassAccuracy(num_classes=num_classes, average='weighted')
    resnet_model, resnet_results = train_resnet(debiasing=args.use_debiasing, num_classes=num_classes, transfer_learning=args.transfer_learning, num_epochs=args.num_epochs, batch_size=args.batch_size, lr=args.lr)
    predictions, all_labels = get_predictions(resnet_model, num_classes=num_classes)
    confm_path = "conf_matrix.png"
    plot_confusion_matrix(predictions, all_labels, confm_path, num_classes=num_classes)
    wandb.log({"confusion matrix": wandb.Image(confm_path)})
    gender_bias = calculate_gender_bias(predictions, all_labels, metric, num_classes=num_classes)
    wandb.log(gender_bias)
    age_bias = calculate_age_bias(predictions, all_labels, metric, num_classes=num_classes)
    wandb.log(age_bias)
    hairiness_bias = calculate_hairiness_bias(predictions, all_labels, metric, num_classes=num_classes)
    wandb.log(hairiness_bias)
    print(resnet_results)
    
