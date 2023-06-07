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


def train_resnet(debiasing=False):
    save_name = "ResNet"
    print("saving to ", os.path.join(CHECKPOINT_PATH, save_name))

    trainer = pl.Trainer(
        default_root_dir=os.path.join(CHECKPOINT_PATH, save_name),
        accelerator="gpu",
        devices=[3],
        max_epochs=20,
        callbacks=[
            ModelCheckpoint(
                save_weights_only=True, save_last=True
                #save_weights_only=True, mode="max", monitor="val_acc"
            ),  # Save the best checkpoint based on the maximum val_acc recorded
        ],
        logger=wandb_logger
    )
    train_set = get_dataset("train", under_sampling=True, use_sample_probabilities=debiasing)
    val_set = get_dataset("validation")
    test_set = get_dataset("test")
    train_loader = DataLoader(train_set, batch_size=32, shuffle=(not debiasing), drop_last=False, pin_memory=False, num_workers=1)
    val_loader = DataLoader(val_set, batch_size=32, shuffle=False, drop_last=False, num_workers=4)
    test_loader = DataLoader(test_set, batch_size=32, shuffle=False, drop_last=False, num_workers=4)

    model = ResNetModel()
    trainer.fit(model, train_loader, val_loader)
    model = ResNetModel.load_from_checkpoint(
        trainer.checkpoint_callback.best_model_path
    )  # Load best checkpoint after training

    # Test best model on validation and test set
    val_result = trainer.validate(model, dataloaders=val_loader, verbose=False)
    test_result = trainer.test(model, dataloaders=test_loader, verbose=False)
    result = {"test": test_result, "val": val_result}

    return model, result


def get_predictions(model, data_set_name="test"):
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)
    
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=[0],
        logger=False,
    )

    data_set_with_metadata = get_dataset(data_set_name, include_metadata=True)
    data_loader = DataLoader(data_set_with_metadata, batch_size=1, shuffle=False, drop_last=False, num_workers=0)

    all_labels = []
    for batch in data_loader:
        imgs, labels = batch
        all_labels.append(labels)

    #print(all_labels)
    predictions = trainer.predict(model, data_loader)
    predictions = torch.cat(predictions)
    #print(predictions)
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

    
def calculate_gender_bias(predictions, all_labels, num_classes=4):
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

    metric = torchmetrics.classification.MulticlassAccuracy(num_classes=num_classes, average='weighted')
    male_accuracy = metric(torch.cat(male_predictions), torch.cat(male_labels)).item()
    female_accuracy = metric(torch.cat(female_predictions), torch.cat(female_labels)).item()
    bias = variance([male_accuracy, female_accuracy])
    results = {"male_acc": male_accuracy, "female_acc": female_accuracy, "gender_bias": bias}
    print(f"male_acc: {male_accuracy}")
    print(f"female_acc: {female_accuracy}")
    print(f"bias: {bias}")
    return results


def calculate_age_bias(predictions, all_labels, num_classes=4):
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
    
    metric = torchmetrics.classification.MulticlassAccuracy(num_classes=num_classes, average='weighted')

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

    print(f"accuracies: {accuracies}")
    print(f"age_bias: {variance(acc_list)}")
    results = {"accuracies": accuracies, "age_bias": variance(acc_list)}
    return results


def calculate_hairiness_bias(predictions, all_labels, num_classes=4):
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

    metric = torchmetrics.classification.MulticlassAccuracy(num_classes=num_classes, average='weighted')
    high_density_accuracy = metric(torch.cat(high_density_predictions), torch.cat(high_density_labels)).item()
    low_density_accuracy = metric(torch.cat(low_density_predictions), torch.cat(low_density_labels)).item()
    bias = variance([high_density_accuracy, low_density_accuracy])
    results = {"high_density_acc": high_density_accuracy, "low_density_acc": low_density_accuracy, "hairiness_bias": bias}
    print(f"high_density_acc: {high_density_accuracy}")
    print(f"low_density_acc: {low_density_accuracy}")
    print(f"hairiness_bias: {bias}")
    return results


if __name__ == "__main__":
    num_classes = 2
    resnet_model, resnet_results = train_resnet(debiasing=False)
    predictions, all_labels = get_predictions(resnet_model)
    confm_path = "conf_matrix.png"
    plot_confusion_matrix(predictions, all_labels, confm_path, num_classes=num_classes)
    wandb.log({"confusion matrix": wandb.Image(confm_path)})
    gender_bias = calculate_gender_bias(predictions, all_labels, num_classes=num_classes)
    wandb.log(gender_bias)
    age_bias = calculate_age_bias(predictions, all_labels, num_classes=num_classes)
    wandb.log(age_bias)
    hairiness_bias = calculate_hairiness_bias(predictions, all_labels, num_classes=num_classes)
    wandb.log(hairiness_bias)
    
    print(resnet_results)
