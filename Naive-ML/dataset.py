from torch.utils.data import Dataset
from os import path
import pandas as pd
import torch
import sys
import numpy as np

class TrainDataSet(Dataset):
    def __init__(self, config):
        self.train_data_path = config['datapath']
        self.train_list = get_id(Filepath='trainID.txt')
        self.data, self.label = read_data(self.train_data_path)

    def __getitem__(self, index):
        case_id = int(self.train_list[index])
        data = self.data[case_id]
        label = self.label[case_id,:]

        return torch.FloatTensor(data), torch.FloatTensor(label)

    def __len__(self):
        return len(self.train_list)


class ValidateDataSet(Dataset):
    def __init__(self, config):
        self.validate_data_path = config['datapath']
        self.data, self.label = read_data(self.validate_data_path)
        self.validate_list = get_id(Filepath='testID.txt')

    def __getitem__(self, index):

        Vcase_id = int(self.validate_list[index])
        Vali_data = self.data[Vcase_id]
        Vali_label = self.label[Vcase_id,:]

        return torch.FloatTensor(Vali_data), torch.FloatTensor(Vali_label)

    def __len__(self):
        return len(self.validate_list)

class TestDataSet(Dataset):
    def __init__(self, config):
        self.test_data_path = config['datapath']
        self.test_list = get_id(Filepath='testID.txt')
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

    # load the useless column indexes of genetic marker matrix
    index= np.loadtxt('No-useIndex.txt')
    # delete the useless column in genetic marker matrix
    #makerData = np.delete(makerData, index, axis=1)

    if makerData.shape[0] != yieldData.shape[0]:
        print('The size of input and label are not aligned')
        sys.exit(0)

    return makerData, yieldData




def get_id(Filepath):
    with open(Filepath, 'r') as f:
        ids = [line.strip() for line in f.readlines()]
    return ids


# if __name__ == '__main__':
#     get_id('/home/ssd_2t/adl/code/trainID.txt')