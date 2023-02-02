import pandas
import pandas as pd
import numpy as np
import statistics
import glob
import tqdm
from datetime import datetime, timedelta
import random

class BiGRUver2(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
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

        h0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size)
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

class labCollabLM(pl.LightningModule):
    def __init__(self, params):
        super().__init__()
        self.save_hyperparameters(params)
        self.mainModel = BiGRUver2()
        #EfficientNet3D.from_name("efficientnet-b2", override_params={'num_classes': 8}, in_channels=2)

        #Metrics
        self.accuracy = torchmetrics.Accuracy()
        self.learning_rate = self.hparams.lr

    def forward(self, x):
        z_class = self.mainModel(x)
        return z_class

    def training_step(self, batch, batch_idx):
        x, y = batch
        z_class = self.mainModel(x)

        # class loss
        totalLoss = self.comp_loss(z_class, y)

        self.log('train_totalLoss', totalLoss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return totalLoss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        z_class = self.mainModel(x)

        totalLoss = self.comp_loss(z_class, y)
        self.log('valid_totalLoss', totalLoss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        # Additional classification metrics
        y_class = torch.sigmoid(z_class)

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
        return F.binary_cross_entropy_with_logits(z_hat, y)

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