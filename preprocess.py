#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @file: pre_process.py
# @author: Yilong Yang
# @time: 2017/12/21 15:58
########################################################
import scipy.io as sio
import argparse
import os
import sys
import numpy as np
import time
import pickle

np.random.seed(0)

def data_1Dto2D(data, Y=3, X=3):
    data_2D = np.zeros([Y, X])
    data_2D[0] = (data[0], 0, 0)
    data_2D[1] = (0, 0, data[1])
    data_2D[2] = (0, data[2], 0)

    return data_2D

def norm_dataset(dataset_1D):
    norm_dataset_1D = np.zeros([dataset_1D.shape[0], 3])
    for i in range(dataset_1D.shape[1]):
        norm_dataset_1D[:, i] = feature_normalize(dataset_1D[:, i])

    return norm_dataset_1D

def feature_normalize(data):
    mean = data[data.nonzero()].mean()
    sigma = data[data.nonzero()].std()
    data_normalized = data
    data_normalized[data_normalized.nonzero()] = (data_normalized[data_normalized.nonzero()] - mean) / sigma
    return data_normalized

def dataset_1Dto2D(dataset_1D):
    dataset_2D = np.zeros([dataset_1D.shape[0], 3, 3])
    for i in range(dataset_1D.shape[0]):
        dataset_2D[i] = data_1Dto2D(dataset_1D[i])
    return dataset_2D

def norm_dataset_1Dto2D(dataset_1D):
    norm_dataset_2D = np.zeros([dataset_1D.shape[0], 3, 3])
    for i in range(dataset_1D.shape[0]):
        norm_dataset_2D[i] = feature_normalize(data_1Dto2D(dataset_1D[i]))
    # return shape: m*9*9
    return norm_dataset_2D

def windows(data, size):
    start = 0
    while ((start + size) < data.shape[0]):
        yield int(start), int(start + size)
        start += size


def segment_signal_without_transition(data, label, label_index, window_size):
    # get data file name and label file name
    for (start, end) in windows(data, window_size):
        # print(data.shape)
        if ((len(data[start:end]) == window_size)):
            if (start == 0):
                segments = data[start:end]
                segments = np.vstack([segments, data[start:end]])

                labels = np.array(label[label_index])
                labels = np.append(labels, np.array(label[label_index]))
            else:
                segments = np.vstack([segments, data[start:end]])
                labels = np.append(labels, np.array(
                    label[label_index]))  # labels = np.append(labels, stats.mode(label[start:end])[0][0])
    return segments, labels


def apply_mixup(dataset_file, window_size):  # initial empty label arrays
    print("Processing", dataset_file, "..........")
    data_file_in = sio.loadmat(dataset_file)
    data_in = data_file_in["data"].transpose(0, 2, 1)
    label_in = data_file_in["labels"][:, 0]

    for i in range(label_in.shape[0]):
        if (label_in[i] == 9):
            label_in[i] = 2
        elif (label_in[i] == 1):
            label_in[i] = 0
        elif (label_in[i] == 5):
            label_in[i] = 1

    label_inter = np.empty([0])  # initial empty data arrays
    data_inter = np.empty([0, window_size, 3, 3])
    trials = data_in.shape[0]

    # Data pre-processing
    for trial in range(0, trials):
        data = data_in[trial, :, 0:3]
        label_index = trial
        # read data and label
        data = norm_dataset(data)
        data, label = segment_signal_without_transition(data, label_in, label_index, window_size)
        data = dataset_1Dto2D(data)
        data = data.reshape(int(data.shape[0] / window_size), window_size, 3, 3)
        # append new data and label
        data_inter = np.vstack([data_inter, data])
        label_inter = np.append(label_inter, label)

    '''
    print("total data size:", data_inter.shape)
    print("total label size:", label_inter.shape)
    '''
    # shuffle data
    index = np.array(range(0, len(label_inter)))
    np.random.shuffle(index)
    shuffled_data = data_inter[index]
    shuffled_label = label_inter[index]
    return shuffled_data, shuffled_label, record


if __name__ == '__main__':
    begin = time.time()
    print("time begin:", time.localtime())
    dataset_dir = "./raw/"
    window_size = 963
    output_dir = "./preprocessd/"
    # get directory name for one subject
    record_list = [task for task in os.listdir(dataset_dir) if os.path.isfile(os.path.join(dataset_dir, task))]
    output_dir = output_dir + "/"
    if os.path.isdir(output_dir) == False:
        os.makedirs(output_dir)
    # print(record_list)

    for record in record_list:
        file = os.path.join(dataset_dir, record)
        shuffled_data, shuffled_label, record = apply_mixup(file, window_size)
        output_data = output_dir + record + "_dataset.pkl"
        output_label = output_dir + record + "_labels.pkl"

        with open(output_data, "wb") as fp:
            pickle.dump(shuffled_data, fp, protocol=4)
        with open(output_label, "wb") as fp:
            pickle.dump(shuffled_label, fp)
        end = time.time()
        print("end time:", time.localtime())
        print("time consuming:", (end - begin))

        break
