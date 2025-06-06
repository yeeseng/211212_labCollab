import pandas
import pandas as pd
import numpy as np
import statistics
import glob
import tqdm
from datetime import datetime, timedelta
import random

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
    print(pl.__version__)
    # load config file
    with open('200-defaultConfig.yaml') as file:
        defaultConfigDict = yaml.safe_load(file)

    parser = argparse.ArgumentParser()
    for eachKey, eachValue in defaultConfigDict.items():
        parser.add_argument('--' + eachKey, default=eachValue, type=type(eachValue))

    args = parser.parse_args()
    print(args)

    '''
    # setting up
    timeStamp = datetime.now().strftime('%y%m%d%H%M')
    runName = timeStamp + '_' + args.tag + '_cv' + str(args.fold)

    blacklist = []
    # blacklist = ['1.2.826.0.1.3680043.9447', '1.2.826.0.1.3680043.28990','1.2.826.0.1.3680043.28606','1.2.826.0.1.3680043.6714','1.2.826.0.1.3680043.31328','1.2.826.0.1.3680043.11192','1.2.826.0.1.3680043.24281']

    myRSNAcspineLightningModule = RSNAcspineLightningModule(args)
    myRSNAcspineDataModule = RSNAcspineDataModule(blacklist=blacklist, args=args)

    wandb_logger = WandbLogger(project=args.projectName, name=timeStamp, tags=[args.tag])
    #wandb_logger = None

    checkpoint_callback = ModelCheckpoint(dirpath='checkpoints/' + runName + '/',
                                          filename=runName + '-{epoch}-{valid_loss_total:.4f}',
                                          monitor='valid_loss_total',
                                          save_top_k=100,
                                          mode='min')
    # lr_monitor = LearningRateMonitor(logging_interval='epoch')
    trainer = pl.Trainer(logger=wandb_logger, log_every_n_steps=10, callbacks=[checkpoint_callback],
                         accumulate_grad_batches=4,
                         accelerator='gpu', devices=args.num_gpus, strategy='ddp_find_unused_parameters_false', precision = 16,
                         #accelerator='ddp', plugins=DDPPlugin(find_unused_parameters=False)],
                         max_epochs=args.max_epochs, num_sanity_val_steps=10)
    trainer.fit(myRSNAcspineLightningModule, myRSNAcspineDataModule)
    '''