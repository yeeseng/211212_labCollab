import pandas
import pandas as pd
import numpy as np
import statistics
import glob
import tqdm
from datetime import datetime, timedelta
import random
from torch.utils.data import Dataset as BaseDataset

class labCollabDataset(BaseDataset):
    def __init__(
            self,
            dataframe=None,
            patientList=None,
            transform_mask=None,
            args=None
    ):
        self.dataframe = dataframe
        self.vertBodyList = vertBodyList
        self.args = args
        self.resizeTransformPadCrop = monai.transforms.ResizeWithPadOrCrop([args.imageSize, args.imageSize, args.numSlices])
        self.augmentation_both = augmentation_both
        self.augmentation_img = augmentation_img
        self.transform_mask = transform_mask

    def __getitem__(self, i):
        thisVertBody = self.vertBodyList[i]
        thisStudyID = thisVertBody.split('_')[0]
        thisCervicalLvl = int(thisVertBody[-1])

        fxMask_exists = torch.tensor(self.dataframe.loc[thisStudyID].loc['fxMask_exists'], dtype=torch.float)
        bb_exists = torch.tensor(self.dataframe.loc[thisStudyID].loc['bb_exists'], dtype=torch.float)

        target = np.array(self.dataframe.loc[thisStudyID].iloc[thisCervicalLvl:thisCervicalLvl+1].tolist()).astype(float)

        thisfilePath = self.args.imgFilePath.replace('vertBody', thisVertBody)
        imgVol = np.load(thisfilePath)['arr_0']
        imgVol = self.resizeTransformPadCrop(imgVol)

        if bb_exists:
            thisSegFilePath = self.args.segFilePath_fx.replace('vertBody', thisVertBody)
            segVol_fx = np.load(thisSegFilePath)['arr_0']
            segVol_fx = self.resizeTransformPadCrop(segVol_fx)
        else:
            segVol_fx = np.zeros((1, 256, 256, 256))

        combinedVol = np.concatenate((imgVol, segVol_fx), axis=0)

        # apply augmentations
        if self.augmentation_both:
            combinedVol = self.augmentation_both(combinedVol)
        else:
            combinedVol = torch.tensor(combinedVol)  # monai segmentation turns numpy into tensor

        imgVol = combinedVol[0:1, :, :, :]
        segVol_fx = combinedVol[-1:, :, :, :] > 0.05

        if self.augmentation_img:
            imgVol = self.augmentation_img(imgVol)

        segVol_fx = self.transform_mask(segVol_fx)

        return imgVol.type(torch.FloatTensor), segVol_fx.type(torch.FloatTensor), target, fxMask_exists, bb_exists, thisVertBody

    def __len__(self):
        return len(self.vertBodyList)

if __name__ == "__main__":
    dataDF = pd.read_csv('Data/Fe_def_outcome_cleanedAndStratified_YN.csv')