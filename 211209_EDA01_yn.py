#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import math
import statistics
import glob
import tqdm

listOfYearlyImportFiles = glob.glob('Data/lab_data/*')
print(len(listOfYearlyImportFiles))
print(listOfYearlyImportFiles)

componentDataDF = pd.read_csv('Data/211209_Lab_Crosswalk_YN.csv')
componentDataDF.head()

listOfLabComponents = []
for eachIndex, eachRow in componentDataDF.iterrows():
    listOfLabComponents.append(eachRow['First Alphabetical Component'])
    for eachColumnName in ['Related Components 0', 'Related Components 1', 'Related Components 2', 'Related Components 3']:
        if isinstance(eachRow[eachColumnName], str):
            listOfLabComponents.append(eachRow[eachColumnName])

print('number of lab components:', len(listOfLabComponents))

print(listOfLabComponents[250])

listOfLabComponents=listOfLabComponents[250:]

# function to preprocess list to remove nonsensical inputs and get only numerical values
def filterList(oldList):
    stripedList = [x.strip('<>') for x in oldList]
    filteredList = []
    for eachItem in stripedList:
        try:
            floatItem = float(eachItem)
            filteredList.append(floatItem)
        except:
            pass
    noNanList = [x for x in filteredList if math.isnan(x) == False]
    return noNanList

# get basic parameters of list like min, max, mean, standard deviation
def analyzeList(thisList):
    try:
        thisMin = min(thisList)
    except:
        thisMin = -1
        
    try:
        thisMax = max(thisList)
    except:
        thisMax = -1
        
    try:
        thisMean = statistics.mean(thisList)
    except:
        thisMean = -1
        
    try:
        thisStdev = statistics.stdev(thisList)
    except:
        thisStdev = -1
        
    return {'count':len(thisList),'min':thisMin, 'max':thisMax, 'mean':thisMean, 'stdev':thisStdev}

# generate string
def genString(componentName, paramDict):
    thisString = componentName.strip(',')
    thisString +=','+str(paramDict['count'])
    thisString +=','+str(paramDict['min'])
    thisString +=','+str(paramDict['max'])
    thisString +=','+str(paramDict['mean'])
    thisString +=','+str(paramDict['stdev'])
    thisString +='\n'
    return thisString


# In[ ]:


# For EDA, to figure out which column was the ex
with open("outputs/results.csv", "a") as text_file:
    text_file.write('component name,count,min,max,mean,stdev\n')

for eachComponent in listOfLabComponents:
    print(eachComponent)
    completeListOfORDvalues = []
    for eachYearlyImportFile in listOfYearlyImportFiles:
        workingDataFrame = pd.read_csv(eachYearlyImportFile, usecols = ['COMMON_NAME','COMPONENT_NAME','BASE_NAME','ORD_VALUE','REFERENCE_UNIT'])
        workingDataFrame['ORD_VALUE'] = workingDataFrame['ORD_VALUE'].astype('str') 
        workingDataFrame['nameMatched'] = workingDataFrame.apply(lambda x: x['COMMON_NAME']==eachComponent, axis=1)
        workingDataFrame = workingDataFrame[workingDataFrame['nameMatched']==True]
        eachYearListOfORDvalues = workingDataFrame['ORD_VALUE'].values
        eachYearListOfORDvalues = filterList(eachYearListOfORDvalues)
        print(len(eachYearListOfORDvalues))
        completeListOfORDvalues += eachYearListOfORDvalues
        print(len(completeListOfORDvalues))
    with open("outputs/results.csv", "a") as text_file:
        stringToSave = genString(eachComponent,analyzeList(completeListOfORDvalues))
        text_file.write(stringToSave)