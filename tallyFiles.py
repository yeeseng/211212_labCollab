import pandas as pd
import numpy as np
import math
import statistics
import glob
import tqdm
import os

listOfFiles = glob.glob('Data/lab_data_patientLvl/*')
listOfYears = [eachItem.split('_')[-1][:-4] for eachItem in listOfFiles]
listOfYears = list(set(listOfYears))
print(len(listOfFiles))

'''
for eachFile in listOfFiles:
    year = eachFile.split('_')[-1][:-4]
    if year == '2009' or year == '2016':
        os.remove(eachFile)

listOfFiles = glob.glob('Data/lab_data_patientLvl/*')
listOfYears = [eachItem.split('_')[-1][:-4] for eachItem in listOfFiles]
listOfYears = list(set(listOfYears))
print(listOfYears)
'''
