from torch.utils.data import Dataset
from os import path
import pandas as pd
import torch
import sys
import numpy as np

class TrainDataSet(Dataset):
    def __init__(self, config):
        self.train_data_path = config['datapath']
        self.train_list = get_id(Filepath='trainID_1.txt')
        self.data, self.label = read_data(self.train_data_path)


    def __getitem__(self, index):
        case_id = int(self.train_list[index])
        data = self.data[case_id]
        label = self.label[case_id,:]

        return torch.FloatTensor(data), torch.FloatTensor(label)

    def __len__(self):
        return len(self.train_list)


class TestDataSet(Dataset):
    def __init__(self, config):
        self.test_data_path = config['datapath']
        self.test_list = get_id(Filepath='testID_1.txt')
        self.data, self.label = read_data(self.test_data_path)


    def __getitem__(self, index):
        case_id = int(self.test_list[index])
        data = self.data[case_id]
        label = self.label[case_id,:]

        return torch.FloatTensor(data), torch.FloatTensor(label)

    def __len__(self):
        return len(self.test_list)

def read_data(data_path):

    # load data
    makerPath = path.join(data_path, 'genetic_marker_matrix_M.csv')
    yieldPath = path.join(data_path, 'Grain.yield_adjusted.csv')

    yieldData = pd.read_csv(yieldPath, usecols=['Adj.Grain.Yield']).values
    makerData = pd.read_csv(makerPath).values

    #index = np.loadtxt('index8000.txt')
    #makerData = np.delete(makerData, index, axis=1)


   # makerData = makerData.reshape(makerData.shape[0], 1, 1, makerData.shape[1])

    yieldData = noramlization(yieldData)
    #yieldData = yieldData.reshape(yieldData.shape[0], 1, yieldData.shape[1])

    # makerData = yieldData
    if makerData.shape[0] != yieldData.shape[0]:
        print('The size of input and label are not aligned')
        sys.exit(0)

    return makerData, yieldData

def noramlization(data):
    # minVals = data.min(axis=0)
    # maxVals = data.max(axis=0)
    # ranges = (maxVals - minVals) + 1e-8 # +1e-8 to avoid 0 value
    # normData = (data)/ranges
    #
    mean = np.mean(data,axis=0)
    std  = np.std(data,axis=0)
    normData = (data)/std
    return normData



def get_id(Filepath):
    with open(Filepath, 'r') as f:
        ids = [line.strip() for line in f.readlines()]
    return ids


# if __name__ == '__main__':
#     get_id('/home/ssd_2t/adl/code/trainID.txt')