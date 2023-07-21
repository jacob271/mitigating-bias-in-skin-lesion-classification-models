# Mitigating Bias in Skin Lesion Classification Models using Variational Autoencoders

This project provides an implementation of an adapted version of the bias mitigation method by [Amini et. al](https://dl.acm.org/doi/10.1145/3306618.3314243) which was developed as part of my bachelor thesis with the title above. The thesis is available in the `thesis` folder.

You can also check out the [GitHub Page](https://jacob271.github.io/mitigating-bias-in-skin-lesion-classification-models/) of this repository for a summary of my bachelor thesis.

## Abstract of the thesis

Leveraging deep learning for early detection of skin cancer could help prevent deaths. Current skin lesion classification algorithms include biases and perform worse for patients with rarer skin features. An existing bias mitigation method automatically detects rare skin features in a dataset using a Variational Autoencoder and takes them into account when training a classifier. We propose an adaptation of this method that allows having multiple classes. We show that the adaptation is effective in experiment setups similar to those in previous research. Bias with respect to age and skin tone of the patient was successfully reduced by more than 45%, with a significance of p < 0.0005. Further, we observe that using transfer learning diminishes the bias mitigation effects while providing decreased biases on its own. Lastly, we find that the method is not effective for a more complex multi-class skin lesion classification task. We discuss potential reasons and areas for future work.

## Overview of the Method

The method is based on the following steps:

1. Train a Variational Autoencoder (VAE) on the training dataset.
2. Use the trained VAE to extract debiasing sample probabilities for the training dataset. Images with rare skin features are assigned higher probabilities.
3. Use the debiasing sample probabilities to train a classifier. 
4. Evaluate the classifier on the test dataset. This is done using the metric weighted accuracy.
5. Evaluate the bias of the classifier using the attributes sex, age, visible hair, and skin tone. Bias is measured as the variance of weighted accuracy across the different values of the attribute.

To evaluate the effectiveness of the bias mitigation method, we also train a classifier with regular sample probabilities and evaluate its bias.
During the experiments, we applied this method in different configurations.

1. A binary classification task.
2. A binary classification task using transfer learning.
3. A multi-class classification task.


## Setup

For this project, you need to have Python 3.10 installed.

Install the requirements with `pip3.10 install -r requirements.txt`.

We recommend using a virtual environment such as `venv` or `conda`.

### Getting the dataset

Download the datasets for the ISIC2018 challenge task 3 [see here](https://challenge.isic-archive.com/data/#2018) and paste it into a folder `data`.

Run `pip3.10 install isic-cli` followed by these three commands ([see here if that does not work](https://stackoverflow.com/questions/35898734/pip-installs-packages-successfully-but-executables-not-found-from-command-line)):

`isic metadata download --collections 66 >> data/ISIC2018_Task3_Training_GroundTruth/metadata.csv`

`isic metadata download --collections 67 >> data/ISIC2018_Task3_Test_GroundTruth/metadata.csv`

`isic metadata download --collections 73 >> data/ISIC2018_Task3_Validation_GroundTruth/metadata.csv`

### Additional Metadata

Apart from the metadata provided by the ISIC Archive, we consider additional metadata about the skin tone and the hairiness of the patients.
You find the respective metadata files in the `metadata` folder.

We also provide notebooks to reproduce the determination of the skin tone and hairiness of the patients.

To execute these, first, start the notebooks using the following commands respectively:

`jupyter notebook detect_hairs.ipynb`

`jupyter notebook detect_skin_tone.ipynb`

Then follow the instructions provided in the notebooks.

### Logging

We use `wandb` to log important information during the training process and keep an overview of performed runs. You need to create an account on [wandb.ai](https://wandb.ai/) and create a project. Then, you can run `wandb login` and enter your API key.

## Training

The training is separated into two steps: training the VAE and training the classifier.

### Training the VAE

You can train a VAE for two or four classes respectively for the binary and multi-class classification tasks.

To train the VAE, run `python3.10 vae_training.py --num_classes <NUM_CLASSES>`
where `<NUM_CLASSES>` is either 2 or 4.
You can specify additional parameters such as the batch size or the number of epochs. Run `python3.10 vae_training.py --help` to see all available parameters.

This step automatically calculates debiasing sample probabilities for the training dataset and saves them in the `data` folder.

### Training the classifier

You can now decide whether you want to train a classifier based on the ResNet18 architecture with or without bias mitigation. Also, you can specify whether you want to use transfer learning and whether you consider a binary or multi-class classification task.

Run `python3.10 resnet_training.py --help` to see all available parameters.

You could run for instance `python3.10 resnet_training.py --num_classes 2 --use_transfer_learning False --use_debiasing True` to train a classifier with transfer learning and bias mitigation for the binary classification task.

Feel free to change the architecture and experiment with your own models.

## Evaluation

After the training, all relevant information is logged to wandb.
To further evaluate the effectiveness of the bias mitigation method check out the `bias_evaluation.ipynb` notebook.

To do so, first, start the notebook using `jupyter notebook bias_evaluation.ipynb` and follow the instructions in the notebook.
