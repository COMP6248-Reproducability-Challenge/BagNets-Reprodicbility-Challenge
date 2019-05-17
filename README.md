## BagNets Reprodicbility Challenge

Original Paper: [APPROXIMATING CNNS WITH BAG-OF-LOCALFEATURES MODELS WORKS SURPRISINGLY WELL ON IMAGENET](https://openreview.net/pdf?id=SkfMWhAqYQ)

## Usage

#### 1. 
The "sa_" jupyter notebook only separated training/valisation/test sets, the results has already uploaded to this repo (test1.zip, training.zip, validation.zip).

#### 2. 
All "sb_" jupyter notebooks can be directly performed on Colab for reimplementation and result checking. The nootbooks are the process of training, evaluation and testing. 

BagNets9, BagNets17, BagNets33, densenet121, densenet169, resnet50 and Cbam-resnet50 are included.

#### 3. 
The "sc_" jupyter notebook includes the codes of generation of scramble images and their accuracy.

#### 4.
The class of BagNet model is rewrite by us in the file "BagNet_model.py", and a training script to train the model on Stanford Dogs Dataset is in file "train.py".
