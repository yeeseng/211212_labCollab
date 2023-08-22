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
        self.linear3 = nn.Linear(self.num_layers * hidden_size * 2, hidden_size)
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

class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(BiLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.linear1 = nn.Linear(input_size, input_size)
        self.linear2 = nn.Linear(input_size, input_size)

        self.LSTM = nn.LSTM(
            input_size, hidden_size, num_layers, batch_first=True, bidirectional=True
        )
        self.linear3 = nn.Linear(self.num_layers * hidden_size * 2, hidden_size)
        self.linear4 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        batch_size = x.size()[0]

        h0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size).cuda()
        c0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size).cuda()

        # Forward pass
        x = self.linear1(x)
        x = self.linear2(x)
        out, (h_n, c_n) = self.LSTM(x, (h0, c0))
        h_n = h_n.transpose(0, 1)
        h_n = torch.flatten(h_n, start_dim=1, end_dim=-1)
        studyLevelOutputs = F.relu(self.linear3(h_n))
        studyLevelOutputs = self.linear4(studyLevelOutputs)

        return studyLevelOutputs

class simpleANN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(simpleANN, self).__init__()

        self.sequence_size = 12
        self.output_size = 1
        self.input_size = input_size
        self.num_hidden_layers = num_layers-2 # minus input and output layer


        # Create the input layer
        self.input_layer = nn.Linear(self.sequence_size * input_size, hidden_size)

        # Create hidden layers
        self.hidden_layers = nn.ModuleList()
        for _ in range(self.num_hidden_layers):
            self.hidden_layers.append(nn.Linear(hidden_size, hidden_size))

        # Create the output layer
        self.output_layer = nn.Linear(hidden_size, self.output_size)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1, end_dim=-1)

        # Forward pass through input layer
        x = F.relu(self.input_layer(x))

        # Forward pass through hidden layers
        for layer in self.hidden_layers:
            x = F.relu(layer(x))

        # Forward pass through output layer
        x = self.output_layer(x)

        return x

class TransformerModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(TransformerModel, self).__init__()
        self.hidden_size = hidden_size
        self.sequence_length = 12
        self.num_layers = num_layers

        self.embedding = nn.Linear(input_size, hidden_size)

        self.transformer = nn.Transformer(
            d_model=hidden_size, nhead=4, num_encoder_layers=num_layers, num_decoder_layers=num_layers, batch_first=True
        )

        self.linear = nn.Linear(self.sequence_length*hidden_size, 1)

    def forward(self, x):
        #batch_size = x.size(0)
        x = self.embedding(x)  # Input embedding

        # Transformer expects input shape of (sequence_length, batch_size, hidden_size)
        #x = x.permute(1, 0, 2)
        transformer_output = self.transformer(x, x)

        # Back to original shape (batch_size, sequence_length, hidden_size)
        #transformer_output = transformer_output.permute(1, 0, 2)

        transformer_output = torch.flatten(transformer_output, start_dim=1, end_dim=-1)

        studyLevelOutputs = F.relu(self.linear(transformer_output))

        return studyLevelOutputs
class labCollabLM(pl.LightningModule):
    def __init__(self, params):
        super().__init__()
        self.save_hyperparameters(params)
        if self.hparams.baseModel == 'GRU':
            self.mainModel = BiGRUver2(input_size=52, hidden_size=self.hparams.hidden_size, num_layers=self.hparams.num_layers)
        elif self.hparams.baseModel == 'LSTM':
            self.mainModel = BiLSTM(input_size=52, hidden_size=self.hparams.hidden_size, num_layers=self.hparams.num_layers)
        elif self.hparams.baseModel == 'ANN':
            self.mainModel = simpleANN(input_size=52, hidden_size=self.hparams.hidden_size, num_layers=self.hparams.num_layers)
        elif self.hparams.baseModel == 'TRANSFORMER':
            self.mainModel = TransformerModel(input_size=52, hidden_size=self.hparams.hidden_size, num_layers=self.hparams.num_layers)
        else:
            raise Exception("Model type is not recognized")

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

class labCollabDM(pl.LightningDataModule):
    def __init__(self, args=None):
        super().__init__()
        self.args = args

        self.dataDF = pd.read_csv('Data/Fe_def_outcome_cleanedAndStratified_YN_230701.csv', index_col='mrn')
        self.dataDF = self.dataDF[self.dataDF['blacklist2'] == 'False']

        # get train list
        trainDF = self.dataDF[(self.dataDF['fold'] != self.args.fold) & (self.dataDF['fold'] != self.args.test)]
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
    with open('130_config.yaml') as file:
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

    if args.wandb == True:
        wandb_logger = WandbLogger(project=args.projectName, name=timeStamp, tags=[args.tag])
    else:
        wandb_logger = None

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
    #trainer.save_checkpoint('checkpoints/cv5_200epochs.ckpt')