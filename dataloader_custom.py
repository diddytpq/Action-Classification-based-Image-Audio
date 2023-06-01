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


HEIGHT=288
WIDTH=512
mag = 1
sigma = 2.5

TP = TN = FP1 = FP2 = FN = 0

path = os.path.dirname(os.path.abspath(__file__))


def getData(data_path_x):

    x_file_list = os.listdir(data_path_x)
    x_data = []
    y_data = []
    for file_name in x_file_list:
        data = torch.load(data_path_x + file_name)
        if file_name.split("_")[0] == "normal":
            x_data.append(data)
            y_data.append(0)

        else:
            x_data.append(data)
            y_data.append(1)



    return x_data, y_data

class AC_Data_Loader(data.Dataset):
    def __init__(self, data_path_x):
        self.data_path_x = data_path_x

        self.feature_data, self.label_data = getData(self.data_path_x)

    def __len__(self):
        """'return the size of dataset"""
        return len(self.feature_data)

    def __getitem__(self, index):
        x_data = self.feature_data[index]
        y_data = self.label_data[index]
        
       
        return x_data, y_data


def outcome(y_pred, y_true, tol):
    n = y_pred.shape[0]
    i = 0
    tp = tn = fp1 = fp2 = fn = 0

    while i < n:
        if np.max(y_pred[i]) == 0 and np.max(y_true[i]) == 0:
            tn += 1
        elif np.max(y_pred[i]) > 0 and np.max(y_true[i]) == 0:
            fp2 += 1
        elif np.max(y_pred[i]) == 0 and np.max(y_true[i]) > 0:
            fn += 1
        elif np.max(y_pred[i]) > 0 and np.max(y_true[i]) > 0:
            #h_pred

            ball_cand_score = []

            y_pred_img = y_pred[i].copy()
            y_true_img = y_true[i].copy()


            nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(y_pred_img, connectivity = 8)

            if len(stats): 
                stats = np.delete(stats, 0, axis = 0)
                centroids = np.delete(centroids, 0, axis = 0)

            for i in range(len(stats)):
                x, y, w, h, area = stats[i]

                score = np.mean(y_pred_img[y:y+h, x:x+w])

                ball_cand_score.append(score)

            (cx_pred, cy_pred) = centroids[np.argmax(ball_cand_score)]

            #h_true
            (cnts, _) = cv2.findContours(y_true_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            rects = [cv2.boundingRect(ctr) for ctr in cnts]
            max_area_idx = 0
            max_area = rects[max_area_idx][2] * rects[max_area_idx][3]
            for j in range(len(rects)):
                area = rects[j][2] * rects[j][3]
                if area > max_area:
                    max_area_idx = j
                    max_area = area
            target = rects[max_area_idx]
            (cx_true, cy_true) = (int(target[0] + target[2] / 2), int(target[1] + target[3] / 2))

            #print((cx_pred, cy_pred))
            #print((cx_true, cy_true))
            dist = math.sqrt(pow(cx_pred-cx_true, 2)+pow(cy_pred-cy_true, 2))

            if dist > tol:
                fp1 += 1
            else:
                tp += 1
        i += 1
    return (tp, tn, fp1, fp2, fn)


def evaluation(TP, TN, FP1, FP2, FN):
    
    try:
        accuracy = (TP + TN) / (TP + TN + FP1 + FP2 + FN)
    except:
        accuracy = 0

    try:
        precision = TP / (TP + FP1 + FP2)
    except:
        precision = 0

    try:
        recall = TP / (TP + FN)
    except:
        recall = 0
    
    try:
        accuracy_2 = (TP + TN) / (TP + TN + FP1 + FN)
    except:
        accuracy_2 = 0

    try:
        accuracy_3 = (TP) / (TP + FP1 + FN)
    except:
        accuracy_3 = 0
        
    try:
        f1_score = 2 * (precision * recall)/(precision + recall)
    except:
        f1_score = 0

    return (accuracy, precision, recall,  f1_score, accuracy_2, accuracy_3)

def display(TP, TN, FP1, FP2, FN):
    print('======================Evaluate=======================')
    print("Number of true positive:", TP)
    print("Number of true negative:", TN)
    print("Number of false positive FP1:", FP1)
    print("Number of false positive FP2:", FP2)
    print("Number of false negative:", FN)
    (accuracy, precision, recall,  f1_score, accuracy_2, accuracy_3)= evaluation(TP, TN, FP1, FP2, FN)
    
    print(" ")
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 score:", f1_score)
    print(" ")
    print("Accuracy:", accuracy)
    print("Accuracy_2:", accuracy_2)
    print("Accuracy_3:", accuracy_3)
    print('=====================================================')


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
