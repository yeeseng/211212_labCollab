
import os
import random
import numpy as np
import math
import pandas as pd
#import albumentations as albu
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset
from tqdm.notebook import trange, tqdm
from torchvision import models
import torchvision.transforms as transforms
import pytorch_lightning as pl
import torchmetrics
from pytorch_lightning.callbacks import ModelCheckpoint
#from pytorch_lightning.callbacks import LearningRateMonitor
#from pytorch_lightning.plugins.training_type import  DDPPlugin
import wandb
from pytorch_lightning.loggers import WandbLogger
# from torchsummary import summary
import torch.nn.functional as F
import glob
import pickle
from datetime import datetime
import yaml
import argparse

class labCollabDataset(BaseDataset):
    def __init__(
            self,
            dataframe=None,
            patientList=None,
            args=None
    ):
        self.dataframe = dataframe
        self.patientList = patientList

    def __getitem__(self, i):
        thisPatientMRN = self.patientList[i]
        thisSeries = self.dataframe.loc[thisPatientMRN]

        if thisSeries.loc['IDA']:
            diagnosisReelFrame = thisSeries.loc['reelframe']-random.randint(6,12)
        else:
            diagnosisReelFrame = random.randint(1,350)-random.randint(6,12)

        endFrame = max(diagnosisReelFrame,0)
        startFrame = max(diagnosisReelFrame-12,0)
        frameLength = endFrame-startFrame
        offset = 12-frameLength

        selectedReel = np.zeros((52,12))

        patientReel = np.load('Data/lab_data_patientReels/'+str(thisPatientMRN)+'.npy')
        selectedReel[:, offset:] = patientReel[:,startFrame:endFrame]

        return selectedReel, thisSeries.loc['IDA']

    def __len__(self):
        return len(self.patientList)

class labCollabDM(pl.LightningDataModule):
    def __init__(self, args=None):
        super().__init__()
        self.args = args

        self.dataDF = pd.read_csv('Data/Fe_def_outcome_cleanedAndStratified_YN.csv', index_col='mrn')
        self.dataDF = self.dataDF[self.dataDF['blacklist'] == False]

        # get train list
        trainDF = self.dataDF[self.dataDF['fold'] != self.args.fold]
        self.trainList = trainDF.index.to_list()
        self.trainList = [eachStudyNum for eachStudyNum in self.trainList if eachStudyNum not in self.blacklist]

        # get valid list
        validDF = self.dataDF[self.dataDF['fold'] == self.args.fold]
        self.validList = validDF.index.to_list()
        self.validList = [eachStudyNum for eachStudyNum in self.validList if eachStudyNum not in self.blacklist]

        trainAugmentation = get_training_augmentation()
        self.trainDataset = labCollabDataset(dataframe=self.dataDF, patientList=self.trainList, args=args)
        self.validDataset = labCollabDataset(dataframe=self.dataDF, patientList=self.validList, args=args)

    def setup(self, stage = None):
        return

    def train_dataloader(self):
        trainDataloader = DataLoader(self.trainDataset, batch_size=self.args.batch_size, shuffle=True, num_workers=self.args.num_workers)
        return trainDataloader

    def val_dataloader(self):
        validDataloader = DataLoader(self.validDataset, batch_size=self.args.batch_size, shuffle=False, num_workers=self.args.num_workers)
        return validDataloader

    def test_dataloader(self):
        return None

if __name__ == "__main__":
    print('hello world')

    '''
    #listOfFiles = glob.glob('Data/lab_data_patientReels/*')
    #print(len(listOfFiles))
    #print(listOfFiles[:10])

    blacklist = []
    for eachItem in thisDataset:
        try:
            filename = 'Data/lab_data_patientReels/'+str(eachItem)+'.csv'
            patientReelDF = pd.read_csv(filename)
            #patientReelDF = pd.read_csv('Data/lab_data_patientReels/'+eachItem+'.csv')
        except:
            blacklist.append(eachItem)

    print('number of study patients:', len(thisDataset))
    print('number in blacklist:', len(blacklist))
    #with open("outputs/blacklist.txt", "w") as output:
    #    output.write(str(blacklist))

    dataDF['blacklist'] = dataDF.apply(lambda x: True if x.mrn in blacklist else False, axis=1)
    print(dataDF.head())
    dataDF.to_csv('Data/Fe_def_outcome_cleanedAndStratified_YN.csv')
    '''