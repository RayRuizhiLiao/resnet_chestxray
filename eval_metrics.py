'''
Author: Ruizhi Liao

Script for evaluation metric methods
'''

from scipy.stats import logistic
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from sklearn.metrics import matthews_corrcoef, confusion_matrix
from sklearn.metrics import mean_squared_error as mse
import numpy as np
import sklearn
from scipy.special import softmax


def compute_pairwise_auc(labels, preds):
    ''' Compute pariwise AUCs given
        labels (a batch of 4-class one-hot labels) and
        preds (a batch of predictions as 4-class probabilities)
    '''

    pairwise_aucs = {}

    def _pairwise_auc(y_all, pred_all, channel0, channel1):
        num_datapoints = np.shape(pred_all)[0]

        y = []
        pred = []
        for j in range(num_datapoints):
            if y_all[j][channel0] == 1 or y_all[j][channel1] == 1: # Only includer "relavent" predictions/labels
                y.append(y_all[j][channel1])
                pred.append(pred_all[j][channel1]/(pred_all[j][channel0]+pred_all[j][channel1]))
        fpr, tpr, thresholds = sklearn.metrics.roc_curve(y, pred, pos_label=1)
        return round(sklearn.metrics.auc(fpr, tpr), 4)

    pairwise_aucs['0v1'] = _pairwise_auc(labels, preds, 0, 1)
    pairwise_aucs['0v2'] = _pairwise_auc(labels, preds, 0, 2)
    pairwise_aucs['0v3'] = _pairwise_auc(labels, preds, 0, 3)
    pairwise_aucs['1v2'] = _pairwise_auc(labels, preds, 1, 2)
    pairwise_aucs['1v3'] = _pairwise_auc(labels, preds, 1, 3)
    pairwise_aucs['2v3'] = _pairwise_auc(labels, preds, 2, 3)

    return pairwise_aucs

def compute_ordinal_auc(labels, preds):
    ''' Compute ordinal AUCs given
        labels (a batch of 4-class one-hot labels) and
        preds (a batch of predictions as 4-class probabilities)
    '''

    assert np.shape(labels) == np.shape(preds) # size(labels)=(N,C);size(preds)=(N,C)

    num_datapoints = np.shape(preds)[0]
    num_channels = np.shape(preds)[1]
    cutoff_channels = num_channels-1

    ordinal_aucs = [] # 0v123, 01v23, 012v3
    for i in range(cutoff_channels):
        y = []
        pred = []
        for j in range(num_datapoints):
            y.append(sum(labels[j][i+1:])) # P(severity >=1) = P(severity=1) + P(severity=2) + P(severity=3)
            pred.append(sum(preds[j][i+1:])) # P(severity >=1) = P(severity=1) + P(severity=2) + P(severity=3)
        fpr, tpr, thresholds = sklearn.metrics.roc_curve(y, pred, pos_label=1)
        ordinal_aucs.append(round(sklearn.metrics.auc(fpr, tpr), 4))
    
    return ordinal_aucs

def compute_multiclass_auc(labels, preds):
    ''' Compute multiclass AUCs given
        labels (a batch of 4-class one-hot labels) and
        preds (a batch of predictions as 4-class probabilities)
    '''

    assert np.shape(labels) == np.shape(preds) # size(labels)=(N,C);size(preds)=(N,C)

    num_datapoints = np.shape(preds)[0]
    num_channels = np.shape(preds)[1]

    labels = np.array(labels)
    preds = np.array(preds)
    aucs = []
    for i in range(num_channels):
        fpr, tpr, thresholds = sklearn.metrics.roc_curve(labels[:,i], preds[:,i], pos_label=1)
        aucs.append(round(sklearn.metrics.auc(fpr, tpr), 4))

    return aucs

def compute_acc_f1_metrics(labels, preds):
    ''' Compute accuracy, F1, and other metrics given
        labels (a batch of integers between 0 and 3) and
        preds (a batch of predictions as 4-class probabilities)
    '''

    assert len(labels) == np.shape(preds)[0] # size(labels)=(N,1);size(preds)=(N,C)

    pred_classes = np.argmax(preds, axis=1)

    precision, recall, f1, _ = precision_recall_fscore_support(labels, pred_classes)
    accuracy = accuracy_score(labels, pred_classes)
    macro_f1 = np.mean(f1)

    def round_nd_array(array):
        return [round(val, 4) for val in array]

    return {
        "accuracy": round(accuracy, 4),
        "f1": round_nd_array(f1),
        "precision": round_nd_array(precision),
        "recall": round_nd_array(recall),
        'macro_f1': round(macro_f1, 4) 
    }, labels, pred_classes 

def compute_mse(labels, preds):
    ''' Compute MSE given
        labels (a batch of integers between 0 and 3) and
        preds (a batch of predictions as 4-class probabilities)
    '''

    assert len(labels) == np.shape(preds)[0] # size(labels)=(N,1);size(preds)=(N,C)

    num_datapoints = np.shape(preds)[0]
    num_channels = np.shape(preds)[1]

    expect_preds = np.zeros(num_datapoints)
    for i in range(num_datapoints):
        for j in range(num_channels):
            expect_preds[i] += j * preds[i][j]

    return round(mse(labels, expect_preds), 4)
