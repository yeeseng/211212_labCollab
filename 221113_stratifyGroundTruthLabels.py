import pandas
import pandas as pd
import numpy as np
import statistics
import glob
import tqdm
from datetime import datetime, timedelta
import random

def parseResultTime(resultTime):
    thisDate = resultTime.split(' ')[0]
    thisDate = datetime.strptime(thisDate, '%Y-%m-%d')
    return(thisDate)

def timeToString(resultTime):
    return(resultTime.strftime('%Y-%m-%d'))

def calcReelFrame(resultDate, referenceTime=datetime(2008, 9, 30)):
    return (resultDate-referenceTime).days//14

def cleanCSV():
    dataDF = pd.read_csv('Data/Fe_def_outcome_df_07152021.csv')
    uniqueListOfMRNs = dataDF['mrn'].unique()
    toSave = []
    for eachMRN in uniqueListOfMRNs:
        thisDF = dataDF[dataDF['mrn'] == eachMRN]
        thisDF = thisDF[thisDF['iron_deficiency_only_FLAG'] == True]
        if (len(thisDF)) > 0:
            listOfDates = thisDF['ferritin_result_date'].tolist()
            listOfDates = [parseResultTime(eachDate) for eachDate in listOfDates]
            listOfDates.sort()
            dateString = timeToString(listOfDates[0])
            reelFrame = calcReelFrame(listOfDates[0])
            toSave.append({'mrn': eachMRN, 'IDA': True, 'date': dateString, 'reelFrame': reelFrame})
        else:
            toSave.append({'mrn': eachMRN, 'IDA': False, 'date': 'NA', 'reelFrame': 'NA'})
    toSaveDataDF = pandas.DataFrame(toSave)
    return toSaveDataDF

def stratify(dataDF):
    # Stratify positive patients
    dataDF['fold'] = 6
    for eachBool in [True, False]:
        listOfIDApatients = dataDF[dataDF['IDA']==eachBool]['mrn'].tolist()
        random.shuffle(listOfIDApatients)
        numInEachFold = len(listOfIDApatients)//5
        for eachFold in range(5):
            listForThisFold = listOfIDApatients[eachFold*numInEachFold:(eachFold+1)*numInEachFold]
            dataDF['fold'] = dataDF.apply(lambda x: eachFold if x.mrn in listForThisFold else x.fold, axis = 1)
    return dataDF

if __name__ == "__main__":
    dataDF = cleanCSV()
    dataDF = stratify(dataDF)
    dataDF.to_csv('Data/Fe_def_outcome_cleanedAndStratified_YN.csv')

