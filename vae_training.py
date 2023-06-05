from vae_module import VAE
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from dataset import get_dataset
import wandb

wandb_logger = WandbLogger(project="bias-skin-lesion-detection")

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


def train_vae():
    trainer = Trainer(
        accelerator="auto",
        devices=1,
        max_epochs=200,
        callbacks=[
            ModelCheckpoint(
                save_weights_only=True, mode="min", monitor="val_loss"
            ),
        ],
        logger=wandb_logger
    )
    train_set = get_dataset("train", under_sampling=True)
    val_set = get_dataset("validation")
    test_set = get_dataset("test")
    train_loader = DataLoader(train_set, batch_size=12, shuffle=True, drop_last=True, pin_memory=False, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=12, shuffle=False, drop_last=False, num_workers=4)
    test_loader = DataLoader(test_set, batch_size=12, shuffle=False, drop_last=False, num_workers=4)

    model = VAE(input_height=360, enc_type="resnet18", enc_out_dim=512)
    trainer.fit(model, train_loader, val_loader)
    model = VAE.load_from_checkpoint(
        trainer.checkpoint_callback.best_model_path, map_location=device
    )

    val_result = trainer.validate(model, dataloaders=val_loader, verbose=False)
    test_result = trainer.test(model, dataloaders=test_loader, verbose=False)

    return model, {"val": val_result, "test": test_result}


def calculate_sample_probabilities(dataset_name, model, visualize_latent_variables=False):
    model = model.to(device)
    data_set = get_dataset(dataset_name=dataset_name, under_sampling=True,  id_as_label=True)
    
    data_loader = DataLoader(data_set, batch_size=1, shuffle=False, drop_last=False, num_workers=1)
    latent_repr_chunks = []
    all_isic_ids = [[], [], [], []]
    latent_repr_lists = [[], [], [], []]
    counter = 0
    for batch in data_loader:
        counter += 1
        if counter % 100 == 0:
            print(counter)
       
        imgs, labels = batch
        ids = labels[0]
        class_label = labels[1][0].item()
        imgs = imgs.to(device)
        original_imgs = imgs

        chunk_latent_repr = model.encode(imgs)
        latent_repr_lists[class_label].append(chunk_latent_repr)
        all_isic_ids[class_label].extend(ids)    

    latent_repr = []
    for i in range(4):
        latent_repr.append(torch.cat(latent_repr_lists[i], dim=0))
    
    sample_p = []
    
    for j in range(4):
        
        sample_p.append(np.zeros(latent_repr[j].shape[0]))
        bins = 10

        for i in range(256):
            latent_distribution = latent_repr[j][:,i].cpu()
            hist_density, bin_edges =  np.histogram(latent_distribution, density=True, bins=bins)

            bin_edges[0] = -float('inf')
            bin_edges[-1] = float('inf')

            smoothing_fac = 0.0

            bin_idx = np.digitize(latent_distribution, bin_edges) 
            hist_density = hist_density / np.sum(hist_density)

            p = 1.0/(hist_density[bin_idx-1] + smoothing_fac)
            p = p/np.sum(p) * 0.25

            # This is a bit different to what is described in the paper
            sample_p[j] = np.maximum(p, sample_p[j])

            if i <= 0 and visualize_latent_variables:
                plt.bar(bin_edges[:-1], hist_density, width=np.diff(bin_edges), align='edge')
                plt.xlabel('Bins')
                plt.ylabel('Density')
                plt.title(f"Histogram for latent variable {i+1}")
                plt.show()


        sample_p[j] = sample_p[j]/np.sum(sample_p[j])*0.25
        print(np.sum(sample_p[j]))
    
    print(np.concatenate(sample_p))
    print(np.concatenate(all_isic_ids))
    
    return np.concatenate(sample_p), np.concatenate(all_isic_ids)

if __name__ == "__main__":
    resnet_model, resnet_results = train_vae()
    sample_probs, isic_ids = calculate_sample_probabilities("train", resnet_model)
    data_dict = {"isic_id": isic_ids, "sample_probability":sample_probs.tolist()}
    dataframe = pd.DataFrame(data_dict)
    wandb_table = wandb.Table(dataframe=dataframe)
    table_artifact = wandb.Artifact(
        "sample_probs_artifact", 
        type="dataset"
    )        
    table_artifact.add(wandb_table, "sample_probs_table")

    dataframe.to_csv("train_sample_probabilities.csv")
    table_artifact.add_file("train_sample_probabilities.csv")
    
    wandb.log_artifact(table_artifact)
    
    print(resnet_results)
