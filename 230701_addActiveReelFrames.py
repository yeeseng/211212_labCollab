import pandas as pd
import numpy as np
from tqdm import tqdm

'''
def findFirstNonZeroIndex(thisReel):
    for eachIndex in range(len(thisReel)):
        if thisReel[eachIndex] != 0:
            return(eachIndex)
    return 380

dataDF = pd.read_csv('Data/Fe_def_outcome_cleanedAndStratified_YN.csv', index_col='mrn')
listOfMRNs = dataDF.index.to_list()
for eachMRN in tqdm(listOfMRNs):
    if dataDF.loc[eachMRN].loc['blacklist'] == False:
        thisReelPath = 'Data/lab_data_patientReels/' + str(eachMRN) + '.npy'
        thisReel = np.load(thisReelPath)
        thisReel = np.sum(thisReel, axis=0)
        #startNonZero = str(findFirstNonZeroIndex(thisReel))
        dataDF.at[eachMRN,'startNonZero'] = findFirstNonZeroIndex(thisReel)

        if dataDF.loc[eachMRN].loc['IDA'] == True:
            if (dataDF.loc[eachMRN].loc['reelFrame']-24-dataDF.loc[eachMRN].loc['startNonZero'])>0:
                dataDF.at[eachMRN, 'blacklist2'] = 'False'
            else:
                dataDF.at[eachMRN, 'blacklist2'] = 'notEnoughHistoricLabs'
        else:
            if (338-dataDF.loc[eachMRN].loc['startNonZero'])>0:
                dataDF.at[eachMRN, 'blacklist2'] = 'False'
            else:
                dataDF.at[eachMRN, 'blacklist2'] = 'notEnoughHistoricLabs'
    else:
        dataDF.at[eachMRN, 'blacklist2'] = 'noLabsRecorded'

dataDF.to_csv('Data/Fe_def_outcome_cleanedAndStratified_YN_230701.csv')
'''

dataDF = pd.read_csv('Data/Fe_def_outcome_cleanedAndStratified_YN_230701.csv', index_col='mrn')
dataDF = dataDF[dataDF['blacklist2']=='False']
dataDF = dataDF[dataDF['IDA']==False]
print(len(dataDF))