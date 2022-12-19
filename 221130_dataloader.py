import pandas
import pandas as pd
import numpy as np
import statistics
import glob
import tqdm
from datetime import datetime, timedelta
import random
from torch.utils.data import Dataset as BaseDataset
import random

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

if __name__ == "__main__":
    dataDF = pd.read_csv('Data/Fe_def_outcome_cleanedAndStratified_YN.csv', index_col='mrn')
    dataDF = dataDF[dataDF['blacklist']==False]
    print(dataDF.head())

    thisDataset = labCollabDataset(dataframe=dataDF)
    print(thisDataset.__getitem__(3)[0].shape)

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