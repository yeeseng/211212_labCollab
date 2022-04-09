import pandas as pd
import numpy as np
import statistics
import glob
import tqdm
from datetime import datetime, timedelta

def parseResultTime(resultTime):
    thisDate = resultTime.split(' ')[0]
    thisDate = datetime.strptime(thisDate, '%d-%b-%Y')
    return(thisDate)

def calcReelFrame(resultDateString, referenceTime=datetime(2008, 9, 30)):
    return parseResultTime(resultDateString)-referenceTime

def normalizeHelper(ORD_VALUE, FINAL_REF_LOW, FINAL_REF_HIGH):
    normalizedValue = 0
    try:
        normalizedValue = (float(ORD_VALUE)-float(FINAL_REF_LOW))/(float(FINAL_REF_HIGH)-float(FINAL_REF_LOW))-0.5
    except:
        normalizedValue = 0
    return normalizedValue

# NOTE: needs enc from #2 in table of contents where hospital encounters are excluded
def lab_clean(df, use_ref_ranges=True):
    # Restrict to non-hospital through inner join
    #df = pd.merge(df, enc['pat_enc_csn_id'], on='pat_enc_csn_id', how='inner')
    # Restrict to >9/2008
    df['REEL_FRAME'] = df.apply(lambda x: calcReelFrame(x['RESULT_DATE']).days//7, axis=1)
    df = df[df['REEL_FRAME'] > 0]
    # Remove external orders/results
    df = df[(df.ORDER_CLASS_C != 30) & (df.ORDER_CLASS_C != 143)]
    # Remove point of care
    df = df[(~df.PROC_NAME.str.contains('(ABL)|(iStat)|(POC)|(POCT)')) & (df.ORDER_CLASS != 'Point of Care')]

    # Remove if no ref range if TRUE
    if use_ref_ranges == True:
        df['REF_LOW_2'] = df.REF_NORMAL_VALS.apply(lambda x: x.split('-')[0] if (isinstance(x, str)) else x)
        df['REF_HIGH_2'] = df.REF_NORMAL_VALS.apply(
            lambda x: x.split('-')[1] if ((isinstance(x, str)) and (len(x.split('-')) == 2)) else x)
        df.REF_LOW_2 = pd.to_numeric(df['REF_LOW_2'], errors='coerce')
        df.REF_HIGH_2 = pd.to_numeric(df['REF_HIGH_2'], errors='coerce')
        df['FINAL_REF_LOW'] = df.apply(lambda x: x.REF_LOW_2 if ~np.isnan(x.REF_LOW_2) else x.REFERENCE_LOW, axis=1)
        df['FINAL_REF_HIGH'] = df.apply(lambda x: x.REF_HIGH_2 if ~np.isnan(x.REF_HIGH_2) else x.REFERENCE_HIGH, axis=1)
        df = df[(~df.FINAL_REF_LOW.isna()) & (~df.FINAL_REF_HIGH.isna())]
        df['ORD_VALUE_NORM'] = df.apply(lambda x: normalizeHelper(ORD_VALUE = x.ORD_VALUE, FINAL_REF_LOW=x.FINAL_REF_LOW, FINAL_REF_HIGH=x.FINAL_REF_HIGH), axis=1)
    return df

def singleLabComponentReel(df, labComp):
    #print(labComp)
    thisDF = df[df['COMMON_NAME']==labComp]
    #print(thisDF.head())
    labREEL = np.zeros(580)
    if len(thisDF)>0:
        for eachIndex in range(580):
            listOfLabs = thisDF[thisDF['REEL_FRAME']==eachIndex].ORD_VALUE_NORM.values
            listOfLabs = [float(eachItem) for eachItem in listOfLabs]
            if len(listOfLabs)>0:
                labREEL[eachIndex] = statistics.mean(listOfLabs)
            elif eachIndex ==0:
                labREEL[eachIndex]=0
            else:
                labREEL[eachIndex]=labREEL[eachIndex-1]
    return labREEL

if __name__ == "__main__":
    listOfFileNames = [eachItem.split('/')[-1][:-3] for eachItem in glob.glob('Data/lab_data_patientLvl/*.csv')]
    listOfIDs = list(set([eachItem.split('_')[0] for eachItem in listOfFileNames]))
    listOfIDs.sort(reverse=True)

    labComponentsDF = pd.read_csv('Data/lab_stats_Yee_JW_edit.csv')
    labCompList = labComponentsDF[labComponentsDF['include']==1]['component name'].values

    csvFilepathTemplate = 'Data/lab_data_patientLvl/id_*.csv'
    npySavePath = 'Data/lab_data_patientReels/id.npy'

    #print(len(listOfIDs))
    for eachID in tqdm.tqdm(listOfIDs):
        #print(eachID)
        thisPath = csvFilepathTemplate.replace('id',eachID)
        thisPathList = glob.glob(thisPath)
        thisListOfDataFrames = [pd.read_csv(eachItem, usecols=['MRN', 'PAT_ENC_CSN_ID', 'ORDER_PROC_ID', 'PROC_NAME', 'RESULT_DATE', 'COMMON_NAME',
                                                'COMPONENT_NAME', 'BASE_NAME', 'ORD_VALUE', 'REFERENCE_LOW', 'REFERENCE_HIGH', 'REFERENCE_UNIT',
                                                'REF_NORMAL_VALS', 'ORDER_CLASS_C', 'ORDER_CLASS'])
                                for eachItem in thisPathList]

        idDataFrame = pd.concat(thisListOfDataFrames)
        idDataFrame = lab_clean(idDataFrame)
        #print(idDataFrame.dtypes)

        labReelList = []
        for eachItem in labCompList:
            labReel = singleLabComponentReel(idDataFrame, labComp=eachItem)
            labReelList.append(labReel)
        patientReel = np.stack(labReelList)

        thisSavePath = npySavePath.replace('id', eachID)
        #print(thisSavePath)
        np.save(thisSavePath, patientReel)

        #print(labReelList.shape)
        #print(labReelList[:,579].shape)

        #print(thisListOfDataFrames)