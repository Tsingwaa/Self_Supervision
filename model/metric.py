"""
@Author: Tsingwaa Tsang
@Date: 2020-02-06 15:09:19
@LastEditors: Tsingwaa Tsang
@LastEditTime: 2020-02-16 23:19:50
@Description: Null
"""

import torch


def cal_acc(output, target):
    with torch.no_grad():
        _pred = torch.argmax(output, dim=1)
        assert _pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(_pred == target).item()
    return correct / len(target)


def conf_matrix(_pred, _true, _class_num, is_print=False, _name_list=None):
    import numpy as np
    _pred = [int(i) for i in _pred]
    _true = [int(i) for i in _true]

    conf_mat = np.zeros((_class_num, _class_num), dtype=np.int)

    for idx, pred_lb in enumerate(_pred):
        tgt_lb = _true[idx]
        conf_mat[tgt_lb][pred_lb] += 1

    if is_print & (_name_list is not None):
        for idx in range(len(_name_list)):
            if idx == 0:
                print('\n\t' + '\t'.join('p_' + str(i) for i in _name_list))
            print('t_' + str(_name_list[idx]) + '\t' + '\t'.join(str(i) for i in conf_mat[idx]))

    return conf_mat


def cal_recall_precision(_conf_mat, is_print=True, _name_list=None):
    import numpy as np

    class_correct_nums = np.diag(_conf_mat)
    class_true_nums = np.sum(_conf_mat, axis=1).transpose()
    class_pred_nums = np.sum(_conf_mat, axis=0)
    class_recall = np.around(class_correct_nums / class_true_nums, decimals=2)
    class_precision = np.around(class_correct_nums / class_pred_nums, decimals=2)

    if is_print & (_name_list is not None):
        print('\nclass    \t' + '\t'.join(str(i) for i in _name_list))
        print('recall   \t' + '\t'.join(str(i) for i in class_recall))
        print('precision\t' + '\t'.join(str(i) for i in class_precision))

    return class_recall, class_precision


def save_ckt(model, opt, epoch, best_mean_recall, save_name):
    print('Saving the state of  model and optimizer...')
    save_dir = r'./backup/models/'
    save_path = save_dir + '_' + save_name
    torch.save({
        "model": model.state_dict(),
        "epoch": epoch,
        "opt": opt.state_dict(),
        "recall": best_mean_recall,
    }, save_path)


def load_ckt(model, opt, load_path):
    print('Loading model and optimizer from saved checkpoint...')
    ckt = torch.load(load_path)
    model.load_state_dict(ckt['model'])
    opt.load_state_dict(ckt['opt'])
    last_epoch = ckt['epoch']
    last_mean_recall = ckt['recall']

    return model, opt, last_epoch, last_mean_recall


if __name__ == '__main__':
    pred = torch.randint(8, [10], dtype=torch.uint8)
    true = torch.randint(8, [10], dtype=torch.uint8)
    conf_mat_ = conf_matrix(pred, true, 8, is_print=True, _name_list=[i for i in range(8)])
    cal_recall_precision(conf_mat_, True, [i for i in range(8)])
