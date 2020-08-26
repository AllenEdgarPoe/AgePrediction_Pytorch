from torch.utils.data import Dataset
import os
from torchvision.transforms import transforms
import csv  # Python module for csv file processing
import pandas as pd
from scipy import ndimage as sio # Library for image processing

class AgeGenderDataset(Dataset):
    def __init__(self, file_path, transform=None, LAP=True):
        self.dataframe = pd.read_csv(file_path)
        self.transform = transform
        self.LAP = LAP


    ## Returns the size of the datasets
    def __len__(self):
        return len(self.dataframe)

    ## Support the indexing s.t. dataset[i] retrieves i-th sample
    def __getitem__(self, idx):
        img_path = os.path.join('./data', self.dataframe.iloc[idx,2])
        img = sio.imread(img_path, mode='RGB')
        img = transforms.ToPILImage()(img)
        if self.LAP:
            age,std = self.dataframe.iloc[idx, 0].astype('float'), self.dataframe.iloc[idx, 3].astype('float')
        else:
            age = self.dataframe.iloc[idx, 0].astype('float')

        if age not in range(0,100):
            age = 0

        if not self.transform==None:
            img = self.transform(img) # ndarray => torch.Tensor  + Normalization

        if self.LAP:
            return img, age, std
        else:
            return img, age