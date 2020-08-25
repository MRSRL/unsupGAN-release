# unsupGAN-release

# Unsupervised MRI Reconstruction

Unsupervised MRI Reconstruction with Generative Adversarial Networks

- 2020 Elizabeth Cole, Stanford University (ekcole@stanford.edu)

## Setup

Make sure the python requirements are installed

    pip3 install -r requirements.txt

The setup assumes that the latest Berkeley Advanced Reconstruction Toolbox is installed [1]. The scripts have all been tested with v0.4.01.

## Data preparation

We will first download data, generate sampling masks, and generate TFRecords for training. The datasets downloaded are fully sampled volumetric knee scans from mridata [2]. The setup script uses the BART binary. In a new folder, run the follwing script:

    python3 mri_util/setup_mri.py -v

## Training/Testing Unsupervised GAN

The training of the unsupervised GAN can be ran using the following script:

    python3 train_unsupervised.py dataset_dir model_dir

where dataset_dir is the folder where the knee datasets were saved to,
and model_dir will be the top directory where the models will be saved to.

Testing can be ran using:

    python3 test_unsupervised.py dataset_dir model_dir

## Training/Testing Supervised GAN

The training of the supervised GAN can be ran using the following script:

    python3 train_supervised.py dataset_dir model_dir

where dataset_dir is the folder where the knee datasets were saved to,
and model_dir will be the top directory where the models will be saved to.

Testing can be ran using:

    python3 test_supervised.py dataset_dir model_dir

## Questions/Issues

For any issues or questions, please open an issue on the github repo or contact
Elizabeth at ekcole@stanford.edu.