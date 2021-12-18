import numpy as np


def confusion_matrix(predict_one: list, predict_zero: list):
    true_one = labels[predict_one]
    true_zero = labels[predict_zero]

    tp = np.sum(np.where(true_one == 1, 1, 0))
    tn = np.sum(np.where(true_zero == 0, 1, 0))
    fp = np.sum(np.where(true_zero == 1, 1, 0))
    fn = np.sum(np.where(true_one == 0, 1, 0))

    return tp, tn, fp, fn


def f1_score(beta, tp, tn, fp, fn):
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    return (1+beta**2)*(precision*recall)/(beta**2*precision+recall)
