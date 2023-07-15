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

from sklearn.metrics import accuracy_score

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
        self.mainModel = BiGRUver2(input_size=52, hidden_size=32, num_layers=self.hparams.num_layers)
        #self.mainModel = simpleANN(sequence_size=12, input_size=52)

        # Loss
        self.criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(self.hparams.pos_weight))
        #self.criterion = torch.nn.BCELoss()

        # Metrics
        self.accuracy = torchmetrics.Accuracy(task='binary')
        self.AUROC = torchmetrics.classification.BinaryAUROC()
        self.learning_rate = self.hparams.lr

    def forward(self, x):
        z_class = self.mainModel(x)
        return z_class

    def training_step(self, batch, batch_idx):
        x, y, _, _ = batch
        z_class = self.mainModel(x)
        y_class = torch.sigmoid(z_class)

        # class loss
        totalLoss = self.criterion(z_class, y)

        self.log('train_totalLoss', totalLoss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return totalLoss

    def validation_step(self, batch, batch_idx):
        x, y, mrn, frame = batch
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
            'mrn' : mrn,
            'frame' : frame
        }

    def validation_epoch_end(self, outputs):
        y_score = torch.cat([eachBatchOutput['class_score'] for eachBatchOutput in outputs])
        y_true = torch.cat([eachBatchOutput['class_true'] for eachBatchOutput in outputs])
        mrn = torch.cat([eachBatchOutput['mrn'] for eachBatchOutput in outputs])
        frame = torch.cat([eachBatchOutput['frame'] for eachBatchOutput in outputs])

        acc_score = self.accuracy(y_score, y_true)
        AUROC_score = self.AUROC(y_score, y_true)
        self.log('valid_class_accuracy', acc_score, prog_bar=True, logger=True, sync_dist=True)
        self.log('valid_class_AUROC', AUROC_score, prog_bar=True, logger=True, sync_dist=True)
        self.saveScores(y_score, y_true, mrn, frame)

        return

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.mainModel.parameters(), lr=self.learning_rate)
        return optimizer

    def saveScores(self, class_score, class_true, mrn, frame):
        class_score = class_score.cpu().detach().numpy().flatten()
        class_true = class_true.cpu().detach().numpy().flatten()
        mrn = mrn.cpu().detach().numpy().flatten()
        frame = frame.cpu().detach().numpy().flatten()

        savePath = 'outputs/tempScores/epoch.csv'.replace('epoch', str(self.current_epoch).zfill(3))

        scoreDict = {'class_score': class_score, 'class_true': class_true, 'mrn':mrn, 'frame':frame}
        scoreDF = pd.DataFrame(scoreDict)
        scoreDF.to_csv(savePath)

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
        self.patientList = patientList

    def __getitem__(self, i):
        thisPatientMRN = self.patientList[i]
        thisSeries = self.dataframe.loc[thisPatientMRN]

        if thisSeries.loc['IDA']:
            startFrame = int(thisSeries.loc['reelFrame'])-random.randint(18,24)
        else:
            startFrame = random.randint(thisSeries.loc['startNonZero'],338)

        endFrame = startFrame+12

        selectedReel = np.zeros((52,12))

        patientReel = np.load('Data/lab_data_patientReels/'+str(thisPatientMRN)+'.npy')
        patientReel = np.nan_to_num(patientReel)
        selectedReel[:,:] = patientReel[:,startFrame:endFrame]
        selectedReel=np.transpose(selectedReel).astype(np.float32)
        GTlabels = np.array([thisSeries.loc['IDA'].astype(np.float16)]) # pytorch lightning like this as a numpy array, size (batch, 1)

        return selectedReel, GTlabels, thisPatientMRN, startFrame

    def __len__(self):
        return len(self.patientList)

def loadConfig(configPath):
    with open(configPath) as file:
        defaultConfigDict = yaml.safe_load(file)

    parser = argparse.ArgumentParser()
    for eachKey, eachValue in defaultConfigDict.items():
        parser.add_argument('--' + eachKey, default=eachValue, type=type(eachValue))

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    print(pl.__version__)
    torch.set_float32_matmul_precision('medium')

    # load config file
    args = loadConfig('120_config.yaml')

    # load and prepare model from checkpoint
    myCollabLM = labCollabLM(args)
    myCollabLM = myCollabLM.load_from_checkpoint('checkpoints/cv5_200epochs.ckpt')
    myCollabLM = myCollabLM.cuda()
    myCollabLM = myCollabLM.eval()

    # prepare validation set
    dataDF = pd.read_csv('Data/Fe_def_outcome_cleanedAndStratified_YN_230701.csv', index_col='mrn')
    dataDF = dataDF[dataDF['blacklist2'] == 'False']

    validDF = dataDF[dataDF['fold'] == args.fold]
    validList = validDF.index.to_list()

    # analyze each patient reel from validation set
    for patientMRN in tqdm.tqdm(validList):
        patientReel = np.load('Data/lab_data_patientReels/'+str(patientMRN)+'.npy')
        patientReel = np.transpose(patientReel).astype(np.float32)
        patientReel = patientReel[np.newaxis,:,:]

        # inference
        totalFrames = 380-12
        arrayOfPreds = np.empty(shape = (totalFrames))
        for eachIndex in range(totalFrames):
            selectedReel = patientReel[:,eachIndex:eachIndex+12,:]
            selectedReel = torch.tensor(selectedReel).cuda()

            pred = myCollabLM(selectedReel)
            pred = torch.sigmoid(pred)
            pred = pred.detach().cpu().numpy().flatten()
            arrayOfPreds[eachIndex] = pred[0]

        dataDF = pd.DataFrame(arrayOfPreds, columns=['pred'])
        dataDF.to_csv('outputs/completeFramePredictions/'+str(patientMRN)+'.csv')