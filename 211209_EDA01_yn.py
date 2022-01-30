#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import math
import statistics
import glob
import tqdm

listOfYearlyImportFiles = ['Data/lab_data/req1512_lab_2009.csv',
                           'Data/lab_data/req1512_lab_2010.csv',
                           'Data/lab_data/req1512_lab_2011.csv',
                           'Data/lab_data/req1512_lab_2012.csv',
                           'Data/lab_data/req1512_lab_2013.csv',
                           'Data/lab_data/req1512_lab_2014.csv',
                           'Data/lab_data/req1512_lab_2015.csv',
                           'Data/lab_data/req1512_lab_2016.csv',
                           'Data/lab_data/req1512_lab_2017.csv',
                           'Data/lab_data/req1512_lab_2018.csv',
                           'Data/lab_data/req1512_lab_2019.csv',
                           'Data/lab_data/req1512_lab_2020.csv']

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
#print(listOfLabComponents[250])
#listOfLabComponents=listOfLabComponents[250:]

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
        
    return {'acceptedCount':len(thisList),'min':thisMin, 'max':thisMax, 'mean':thisMean, 'stdev':thisStdev}

paramList = ['totalCount', 'acceptedCount', 'min', 'max', 'mean', 'stdev']

# generate string
def genString(componentName, paramDict):
    thisString = componentName.strip(',')
    for eachParam in paramList:
        thisString +=','+str(paramDict[eachParam])
    thisString +='\n'
    return thisString

# For EDA, to figure out which column was the ex
with open("outputs/results.csv", "a") as text_file:
    thisString = 'component name'
    for eachParam in paramList:
        thisString += ',' + eachParam
    thisString += '\n'
    text_file.write(thisString)

for eachComponent in listOfLabComponents:
    print(eachComponent)
    completeListOfORDvalues = []
    totalCount = 0
    for eachYearlyImportFile in listOfYearlyImportFiles:
        workingDataFrame = pd.read_csv(eachYearlyImportFile, usecols = ['COMMON_NAME','COMPONENT_NAME','BASE_NAME','ORD_VALUE','REFERENCE_UNIT'])
        workingDataFrame['ORD_VALUE'] = workingDataFrame['ORD_VALUE'].astype('str') 
        workingDataFrame['nameMatched'] = workingDataFrame.apply(lambda x: x['COMMON_NAME']==eachComponent, axis=1)
        workingDataFrame = workingDataFrame[workingDataFrame['nameMatched']==True]
        eachYearListOfORDvalues = workingDataFrame['ORD_VALUE'].values
        totalCount += len(eachYearListOfORDvalues) # get count of cases before processing
        eachYearListOfORDvalues = filterList(eachYearListOfORDvalues)
        print(len(eachYearListOfORDvalues))
        completeListOfORDvalues += eachYearListOfORDvalues
        print(len(completeListOfORDvalues))
    with open("outputs/results.csv", "a") as text_file:
        resultDict = analyzeList(completeListOfORDvalues)
        resultDict['totalCount'] = totalCount
        stringToSave = genString(eachComponent,resultDict)
        text_file.write(stringToSave)