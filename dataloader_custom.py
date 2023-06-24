import pandas as pd
from torch.utils import data
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import torch
import os
from PIL import Image
import random
import cv2
import math
import sys

# def getData(data_path_x):

#     x_file_list = os.listdir(data_path_x)
#     x_data = []
#     y_data = []
#     for file_name in x_file_list:
#         data = torch.load(data_path_x + file_name)
#         if file_name.split("_")[0] == "normal":
#             x_data.append(data)
#             # y_data.append(torch.tensor([1, 0]))
#             y_data.append(torch.tensor([0]))


#         else:
#             x_data.append(data)
#             # y_data.append(torch.tensor([0, 1]))
#             y_data.append(torch.tensor([1]))

#     return x_data, y_data

def getData(data_path_x, label, x_data, y_data):

    x_file_list = os.listdir(data_path_x)

    for file_name in x_file_list:
        data = torch.load(data_path_x + file_name)
        x_data.append(data)
        
        if label == 0:
            # y_data.append(torch.tensor([1, 0]))
            y_data.append(torch.tensor([0]))

        else:
            # y_data.append(torch.tensor([0, 1]))
            y_data.append(torch.tensor([1]))

    return x_data, y_data

class AC_Data_Loader(data.Dataset):
    def __init__(self, data_path_x_normal, data_path_x_abnormal):
        self.data_path_x_normal = data_path_x_normal
        self.data_path_x_abnormal = data_path_x_abnormal

        self.feature_data = []
        self.label_data = []

        self.feature_data, self.label_data = getData(self.data_path_x_normal, label = 0, x_data = self.feature_data, y_data = self.label_data)
        self.feature_data, self.label_data = getData(self.data_path_x_abnormal, label = 1, x_data = self.feature_data, y_data = self.label_data)

    def __len__(self):
        """'return the size of dataset"""
        return len(self.feature_data)

    def __getitem__(self, index):
        x_data = self.feature_data[index]
        y_data = self.label_data[index]
        
       
        return x_data, y_data


class Image_Data_Loader(data.Dataset):
    def __init__(self, data_path_x_normal, data_path_x_abnormal):
        self.data_path_x_normal = data_path_x_normal
        self.data_path_x_abnormal = data_path_x_abnormal

        self.feature_data, self.label_data = getData(self.data_path_x, label = 0)

    def __len__(self):
        """'return the size of dataset"""
        return len(self.feature_data)

    def __getitem__(self, index):
        x_data = self.feature_data[index]
        y_data = self.label_data[index]
        
       
        return x_data, y_data

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('GPU Use : ',torch.cuda.is_available())

    data_path_x = "./dataset/train/features/"
    data_loader = AC_Data_Loader(data_path_x)

    # train_loader = DataLoader(dataset = data_loader, num_workers = 4, batch_size=1, shuffle=True)
    train_loader = DataLoader(dataset = data_loader, batch_size=4, shuffle=True)


    for i, (x_data, y_data) in enumerate(train_loader):
        print(i)
        print(x_data.shape)
        print(y_data)
