# mitigating-bias-in-skin-lesion-detection-models

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

Check your wandb account to see some metrics about your model or run `python3.10 model_evaluation.py`.
