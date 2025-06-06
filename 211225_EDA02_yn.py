#!/usr/bin/env python
# coding: utf-8

# This EDA to explore patient count

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

listOfMRNs = []

for eachYearlyImportFile in listOfYearlyImportFiles:
    workingDataFrame = pd.read_csv(eachYearlyImportFile, usecols = ['MRN', 'PAT_ENC_CSN_ID', 'ORDER_PROC_ID','ORDERING_DATE','COMMON_NAME','COMPONENT_NAME','BASE_NAME','ORD_VALUE','REFERENCE_UNIT'])
    workingDataFrame['MRN'] = workingDataFrame['MRN'].astype('str')
    thisListOfMRNs = workingDataFrame['MRN'].unique().tolist()
    print(len(thisListOfMRNs))
    listOfMRNs += thisListOfMRNs

print('total before removal of duplicates:',len(listOfMRNs))

# remove duplicates
listOfMRNs = list(set(listOfMRNs))
print('total after removal of duplicates:', len(listOfMRNs))