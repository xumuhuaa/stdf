import scipy.io

from GCForest import gcForest
import _pickle as pickle
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import numpy as np
import datetime
import pandas as pd
import scipy.io as sio

starttime = datetime.datetime.now()

file_list=['P01','P02']

test_accuracy_all=np.zeros(shape=[0],dtype=float)
mean_accuracy_all=0
for data_file in file_list:
    print('sub:', data_file)
    data_suffix = ".mat_dataset.pkl"
    label_suffix = ".mat_labels.pkl"

    dataset_dir = "./preprocessed/"
    data = pickle.load(open(dataset_dir + data_file + data_suffix, 'rb'), encoding='utf-8')
    label = pickle.load(open(dataset_dir + data_file + label_suffix, 'rb'), encoding='utf-8')
    # data = scipy.io.loadmat('/home/xumh/P01.mat')

    x = data
    y = label

    fold = 10
    count = 0
    test_accuracy_all_fold = np.zeros(shape=[0], dtype=float)
    for curr_fold in range(fold):
        fold_size = x.shape[0] // fold
        indexes_list = [i for i in range(len(x))]
        indexes = np.array(indexes_list)
        split_list = [i for i in range(curr_fold * fold_size, (curr_fold + 1) * fold_size)]
        split = np.array(split_list)

        X_test = x[split]
        y_test = y[split]

        split = np.array(list(set(indexes_list) ^ set(split_list)))
        X_train = x[split]
        y_train = y[split]
        print(count)
        print("train_x shape:", X_train.shape)
        print("test_x shape:", X_test.shape)

        train_sample = y_train.shape[0]
        test_sample = y_test.shape[0]

        X_train = X_train.transpose(1, 0, 2, 3)
        X_test = X_test.transpose(1, 0, 2, 3)

        X_train_mgs = np.zeros(shape=[0], dtype=float)
        X_test_mgs = np.zeros(shape=[0], dtype=float)

        for i in range(0, X_train.shape[0]):
            X_train_ = X_train[i]
            X_train_ = X_train_.reshape(X_train.shape[1], 9)
            gcf = gcForest(shape_1X=[3, 3], window=2, tolerance=0.0)
            X_train_mgs_ = gcf.mg_scanning(X_train_, y_train)
            X_train_mgs = np.concatenate((X_train_mgs, X_train_mgs_), axis=1)
            #print('X_tr_mgs_.shape:', X_tr_mgs_.shape)
            X_test_ = X_test[i]
            X_test_ = X_test_.reshape(X_test.shape[1], 9)
            #print('X_te_mgs_.shape:', X_te_mgs_.shape)
            X_te_mgs_ = gcf.mg_scanning(X_test_)
            X_te_mgs = np.concatenate((X_test_mgs, X_te_mgs_), axis=1)
        print('X_te_mgs.shape:', X_test_mgs.shape)

        gcf = gcForest(shape_1X=[3, 3], window=2, tolerance=0.0)
        _ = gcf.cascade_forest(X_train_mgs, y_train)
        pred_proba = np.mean(gcf.cascade_forest(X_test_mgs), axis=0)
        # print(X_te_mgs.shape)
        preds = np.argmax(pred_proba, axis=1)

        test_sample = y_test.shape[0]
        print("test_x shape:", X_test.shape)
        # evaluating accuracy
        accuracy = accuracy_score(y_true=y_test, y_pred=preds)
        print('accuracy = '.format(accuracy))
        Confusion_result = confusion_matrix(y_test, preds)
        print('confusion_matrix:', Confusion_result)
        test_accuracy_all = np.append(test_accuracy_all, accuracy)
        count += 1

    summary = pd.DataFrame({'fold': range(1, fold + 1), 'test_accuracy': test_accuracy_all_fold})
    writer = pd.ExcelWriter("./result/"  + "/" + data_file  + ".xlsx")
    summary.to_excel(writer, 'summary', index=False)
    writer.save()

endtime = datetime.datetime.now()
print(endtime - starttime)