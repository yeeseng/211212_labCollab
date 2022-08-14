import pandas as pd
import numpy as np
import statistics
import glob
import tqdm
from datetime import datetime, timedelta

if __name__ == "__main__":
    listOfFiles = glob.glob('Data/lab_data_patientReels/*.npy')
    for eachFile in listOfFiles:
        ptID = eachFile.split('/')[-1][:-4]
        thisNpArray = np.load(eachFile)
        thisNpArray = np.logical_or(thisNpArray<-0.001, thisNpArray>0.001)
        sum = thisNpArray.sum()
        percent = sum/(380*52)
        with open('Data/scarcityMetric.txt', 'a') as file:
            file.write(ptID + ',' + str(sum) + ',' + str(percent) + '\n')