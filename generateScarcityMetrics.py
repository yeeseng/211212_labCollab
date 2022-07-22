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
        thisNpArray = thisNpArray != 0
        sum = thisNpArray.sum()
        percent = sum/(312*52)
        with open('Data/scarcityMetric.txt', 'a') as file:
            file.write(ptID + ',' + str(sum) + ',' + str(percent) + '\n')