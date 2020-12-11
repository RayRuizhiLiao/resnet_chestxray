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

'''
Evaluation related helpers
'''
# in the case of gold labels, convert ordinal labels to predictions
def convert_ordinal_label_to_labels(ordinal_label):
    return np.sum(ordinal_label)
    # ordinal_label = "".join(str(v) for v in ordinal_label.tolist())
    # if ordinal_label == '000':
    #     return 0
    # elif ordinal_label == '100':
    #     return 1
    # elif ordinal_label == '110':
    #     return 2
    # elif ordinal_label == '111':
    #     return 3
    # else:
    #     raise Exception("No other possibilities of ordinal labels are possible")

def convert_sigmoid_prob_to_labels(pred):
    sigmoid_pred = logistic.cdf(pred)
    threshold = 0.5
    if sigmoid_pred[0] > threshold:
        if sigmoid_pred[1] > threshold:
            if sigmoid_pred[2] > threshold:
                return 3
            else:
                    return 2
        else:
            return 1
    else:
        return 0

# Given the 4 channel output of multiclass, compute the 3 channel ordinal auc
def compute_ordinal_auc_from_multiclass(labels_raw, preds):
    if len(labels_raw) != len(preds):
        raise ValueError('The size of the labels does not match the size the preds!')
    num_datapoints = len(labels_raw) # labels_raw needs to be between 0 and 1
    if len(preds[0]) != 4:
        raise ValueError('This auc can only be computed for multiclass')
    desired_channels = 3
    ordinal_aucs = [] # 0v123, 01v23, 012v3
    for i in range(desired_channels):
        y = []
        pred = []
        for j in range(num_datapoints):
            y.append(min(1.0, max(0.0, labels_raw[j] - i))) # if gold is 3 and channel is 0v123, then y is 1
            pred.append(sum(preds[j][i+1 : desired_channels+1])) # P(severity >=1) = P(severity=1) + P(severity=2) + P(severity=3)
        fpr, tpr, thresholds = sklearn.metrics.roc_curve(y, pred, pos_label=1)
        ordinal_auc_val = round(sklearn.metrics.auc(fpr, tpr), 4)
        ordinal_aucs.append(ordinal_auc_val)
    return ordinal_aucs

# Given the 4 channel output of multiclass, compute the 3 channel ordinal auc
def compute_ordinal_auc_onehot_encoded(labels, preds):
    if len(labels) != len(preds):
        raise ValueError('The size of the labels does not match the size the preds!')
    num_datapoints = len(labels)
    if len(preds[0]) != 4:
        raise ValueError('This auc can only be computed for multiclass')
    desired_channels = 3
    ordinal_aucs = [] # 0v123, 01v23, 012v3
    for i in range(desired_channels):
        y = []
        pred = []
        for j in range(num_datapoints):
            y.append(sum(labels[j][i+1 : desired_channels+1])) # P(severity >=1) = P(severity=1) + P(severity=2) + P(severity=3)
            pred.append(sum(preds[j][i+1 : desired_channels+1])) # P(severity >=1) = P(severity=1) + P(severity=2) + P(severity=3)
        fpr, tpr, thresholds = sklearn.metrics.roc_curve(y, pred, pos_label=1)
        ordinal_auc_val = round(sklearn.metrics.auc(fpr, tpr), 4)
        ordinal_aucs.append(ordinal_auc_val)
    return ordinal_aucs

def compute_auc(labels, preds):
    ''' Compute AUCs given
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
