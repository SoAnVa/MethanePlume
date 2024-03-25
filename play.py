import os
import numpy as np
import torch
from torch import nn, optim
from tqdm.autonotebook import tqdm
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, random_split, RandomSampler
from torch.utils.tensorboard import SummaryWriter
import argparse
from torchmetrics import JaccardIndex, Accuracy
import time
import random
from torch.utils.data import ConcatDataset

from model_unet import *
from data import create_dataset

bands = [1,2,3,4,5,6,7,8,9,10,11,12,13]
# create the datasets
PATH_D_TRAIN="data/DataTrain/augment_input_tiles/"
PATH_S_TRAIN="data/DataTrain/augment_output_matrix/"
PATH_D_TEST=os.getcwd()+"/data/DataTest/input_tiles/"
PATH_S_TEST=os.getcwd()+"/data/DataTest/output_matrix/"

data_train = create_dataset(
    datadir=PATH_D_TRAIN,
    segdir=PATH_S_TRAIN,
    band=bands,
apply_transforms=True)

# data_val = create_dataset(
#     datadir=PATH_D_TEST,
#     segdir=PATH_S_TEST,
#     band=bands,
#     apply_transforms=False)


# Concatenate the two datasets
#combined_dataset = ConcatDataset([data_train, data_val])

print(len(data_train))
#print(len(data_train))
