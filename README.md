# Mitigating Bias in Skin Lesion Classification Models using Learned Latent Structures

This project provides an implementation of an adapted version of the bias mitigation method by [Amini et. al](https://dl.acm.org/doi/10.1145/3306618.3314243) which was developed as part of my bachelor thesis with the same title.

## Abstract of the thesis

Leveraging deep learning for early detection of skin cancer could help prevent deaths. Current skin lesion classification algorithms include biases and perform worse for patients with rarer skin features. An existing bias mitigation method automatically detects rare skin features in a dataset using a Variational Autoencoder and takes them into account when training a classifier. We propose an adaptation of this method that allows having multiple classes. We show that the adaptation is effective in experiment setups similar to those in previous research. Bias with respect to age and skin tone of the patient was successfully reduced by more than 45%, with a significance of p < 0.0005. Further, we observe that using transfer learning diminishes the bias mitigation effects while providing decreased biases on its own. Lastly, we find that the method is not effective for a more complex multi-class skin lesion classification task. We discuss potential reasons and areas for future work.

## Setup

Install the requirements with `pip3.10 install -r requirements.txt`.

Download the datasets for the ISIC2018 challenge task 3 [see here](https://challenge.isic-archive.com/data/#2018) and paste it into a folder `data`.

Run `pip3 install isic-cli` followed by these three commands ([see here if that does not work](https://stackoverflow.com/questions/35898734/pip-installs-packages-successfully-but-executables-not-found-from-command-line)):

`isic metadata download --collections 66 >> data/ISIC2018_Task3_Training_GroundTruth/metadata.csv`

`isic metadata download --collections 67 >> data/ISIC2018_Task3_Test_GroundTruth/metadata.csv`

`isic metadata download --collections 73 >> data/ISIC2018_Task3_Validation_GroundTruth/metadata.csv`

You now have all the metadata you need for the dataset.

## Training


## Evaluation

Check your wandb account to see some metrics about your model.

Additional metrics are provided in the `model_evaluation.ipynb` notebook. You might want to check out [this link](https://docs.anaconda.com/free/anaconda/jupyter-notebooks/remote-jupyter-notebook/) if you want to run the notebook on a remote server.
