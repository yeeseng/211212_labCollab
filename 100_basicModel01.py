import pandas
import pandas as pd
import numpy as np
import statistics
import glob
import tqdm
from datetime import datetime, timedelta
import random
import yaml
import argparse

# PyTorch Modules
from torch.utils.data import Dataset as BaseDataset
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.nn.functional as F

import torchvision
from torchvision import models
from torchvision import transforms
import torchvision.transforms as transforms
import torchvision.datasets as dsets

import torchmetrics
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

class BiGRUver2(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(BiGRUver2, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.linear1 = nn.Linear(input_size, input_size)
        self.linear2 = nn.Linear(input_size, input_size)

        self.GRU = nn.GRU(
            input_size, hidden_size, num_layers, batch_first=True, bidirectional=True
        )
        self.linear3 = nn.Linear(hidden_size * 2, hidden_size)
        self.linear4 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        batch_size = x.size()[0]

        h0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size).cuda()
        # c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).cuda()

        # Forward pass
        x = self.linear1(x)
        x = self.linear2(x)
        out, h_n = self.GRU(x, h0)
        h_n = h_n.transpose(0,1)
        h_n = torch.flatten(h_n, start_dim=1, end_dim=-1)
        studyLevelOutputs = F.relu(self.linear3(h_n))
        studyLevelOutputs = self.linear4(studyLevelOutputs)

        return studyLevelOutputs

class simpleANN(nn.Module):
    def __init__(self, sequence_size, input_size):
        super(simpleANN, self).__init__()
        self.linear1 = nn.Linear(sequence_size*input_size, 312)
        self.linear2 = nn.Linear(312, 156)
        self.linear3 = nn.Linear(156, 1)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1, end_dim=-1)

        # Forward pass
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))

        return x

class labCollabLM(pl.LightningModule):
    def __init__(self, params):
        super().__init__()
        self.save_hyperparameters(params)
        self.mainModel = BiGRUver2(input_size=52, hidden_size=32, num_layers=1)
        #self.mainModel = simpleANN(sequence_size=12, input_size=52)

        # Loss
        self.criterion = torch.nn.BCEWithLogitsLoss()
        #self.criterion = torch.nn.BCELoss()

        # Metrics
        self.accuracy = torchmetrics.Accuracy(task='binary')
        self.learning_rate = self.hparams.lr

    def forward(self, x):
        z_class = self.mainModel(x)
        return z_class

    def training_step(self, batch, batch_idx):
        x, y = batch
        z_class = self.mainModel(x)
        y_class = torch.sigmoid(z_class)

        # class loss
        totalLoss = self.criterion(z_class, y)

        self.log('train_totalLoss', totalLoss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return totalLoss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        z_class = self.mainModel(x)
        y_class = torch.sigmoid(z_class)

        #print(y_class.dtype)
        #print(y.dtype)
        #breakpoint()

        '''
        if torch.sum(y)==0 or torch.sum(y)==self.hparams.batch_size:
            totalLoss = 0
            self.log('error', 1, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True, batch_size=self.hparams.batch_size)
        else:
            totalLoss = self.criterion(z_class, y)
            self.log('error', 0, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True, batch_size=self.hparams.batch_size)
        '''

        totalLoss = self.criterion(z_class, y)
        self.log('valid_totalLoss', totalLoss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        # Additional classification metrics
        #y_class = torch.sigmoid(z_class)

        class_score = y_class
        class_true = y.type(torch.int32)

        return {
            'class_score' : class_score,
            'class_true' : class_true,
        }

    def validation_epoch_end(self, outputs):
        y_score = torch.cat([eachBatchOutput['class_score'] for eachBatchOutput in outputs])
        y_true = torch.cat([eachBatchOutput['class_true'] for eachBatchOutput in outputs])
        acc_score = self.accuracy(y_score, y_true)
        self.log('valid_class_accuracy', acc_score, prog_bar=True, logger=True, sync_dist=True)
        return

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.mainModel.parameters(), lr=self.learning_rate)
        return optimizer

    def comp_loss(self, z_hat, y):
        return F.binary_cross_entropy_with_logits(z_hat, y, reduction = 'mean')

class labCollabDataset(BaseDataset):
    def __init__(
            self,
            dataframe=None,
            patientList=None,
            args=None
    ):
        self.dataframe = dataframe
        self.patientList = dataframe.index.to_list()

    def __getitem__(self, i):
        thisPatientMRN = self.patientList[i]
        thisSeries = self.dataframe.loc[thisPatientMRN]

        if thisSeries.loc['IDA']:
            diagnosisReelFrame = int(thisSeries.loc['reelFrame'])-random.randint(6,12)
        else:
            diagnosisReelFrame = random.randint(1,350)-random.randint(6,12)

        endFrame = max(diagnosisReelFrame,0)
        startFrame = max(diagnosisReelFrame-12,0)
        frameLength = endFrame-startFrame
        offset = 12-frameLength

        selectedReel = np.zeros((52,12))

        patientReel = np.load('Data/lab_data_patientReels/'+str(thisPatientMRN)+'.npy')
        patientReel = np.nan_to_num(patientReel)
        selectedReel[:, offset:] = patientReel[:,startFrame:endFrame]
        selectedReel=np.transpose(selectedReel).astype(np.float32)
        GTlabels = np.array([thisSeries.loc['IDA'].astype(np.float16)]) # pytorch lightning like this as a numpy array, size (batch, 1)

        return selectedReel, GTlabels

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

        # get valid list
        validDF = self.dataDF[self.dataDF['fold'] == self.args.fold]
        self.validList = validDF.index.to_list()

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
    print(pl.__version__)
    torch.set_float32_matmul_precision('medium')
    # load config file
    with open('100_config.yaml') as file:
        defaultConfigDict = yaml.safe_load(file)

    parser = argparse.ArgumentParser()
    for eachKey, eachValue in defaultConfigDict.items():
        parser.add_argument('--' + eachKey, default=eachValue, type=type(eachValue))

    args = parser.parse_args()
    print(args)

    # setting up
    timeStamp = datetime.now().strftime('%y%m%d%H%M')
    runName = timeStamp + '_' + args.tag + '_cv' + str(args.fold)
    print(runName)

    labCollabLightningModule = labCollabLM(args)
    labCollabDataModule = labCollabDM(args=args)

    wandb_logger = WandbLogger(project=args.projectName, name=timeStamp, tags=[args.tag])
    #wandb_logger = None

    '''
    checkpoint_callback = ModelCheckpoint(dirpath='checkpoints/' + runName + '/',
                                          filename=runName + '-{epoch}-{valid_loss_total:.4f}',
                                          monitor='valid_loss_total',
                                          save_top_k=100,
                                          mode='min')
    '''

    trainer = pl.Trainer(logger=wandb_logger, log_every_n_steps=10,
                         accumulate_grad_batches=10,
                         accelerator='gpu', devices=args.num_gpus, strategy='ddp_find_unused_parameters_false', precision = 16,
                         #accelerator='ddp', plugins=DDPPlugin(find_unused_parameters=False)],
                         max_epochs=args.max_epochs, num_sanity_val_steps=10)
    trainer.fit(labCollabLightningModule, labCollabDataModule)