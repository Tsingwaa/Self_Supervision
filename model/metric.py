"""
@Author: Tsingwaa Tsang
@Date: 2020-02-06 15:09:19
@LastEditors: Tsingwaa Tsang
@LastEditTime: 2020-02-16 23:19:50
@Description: Null
"""

import torch
import numpy as np


def cal_acc(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)


def conf_matrix(pred, target):
    pred = pred.tolist()
    target = target.tolist()
    diff_cls = len(set(target))
    conf_mat = np.zeros((diff_cls, diff_cls)
    for idx, pred_lb in enumerate(pred):
        tgt_lb=target[idx]
        conf_mat[pred_lb][tgt_lb] += 1

    return conf_mat

def cal_mean_recall(conf_mat):
    class_correct_nums=np.diag(conf_mat)
    class_all_nums=np.sum(conf_mat, axis=1).transpose()
    class_recall=np.around(class_correct_nums / class_all_nums, decimals=2)

    return class_recall

def my_metric2(output, target, k=3):
    with torch.no_grad():
        pred=torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct=0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)
