import os
from copy import deepcopy

import numpy as np
from torch.nn import CrossEntropyLoss
from torch.nn.functional import cross_entropy
from torch.optim import SGD, lr_scheduler
from torchvision.models import resnet18
from tqdm import tqdm

from data_loader.my_dataloader import get_dataloader
from model.metric import *
from model.model import Net8FC

os.environ["CUDA_kVISIBLE_DEVICES"] = "7"


def train():
    # 定义变量
    # all_classes = 32
    aug_classes = 5
    num_classes = 8

    epochs = 300

    torch.set_printoptions(precision=2, threshold=100000, linewidth=10000)

    # get dataloader
    mean_std_path = './data/mean_std.json'
    data_root = './data/'
    # loader dict:'train','valid', 'test'
    loader = get_dataloader(mean_std_path, data_root)

    copy_resnet18 = deepcopy(resnet18(pretrained=True))
    # model = Net1FC(copy_resnet18, all_classes).cuda()
    model = Net8FC(copy_resnet18, num_classes, aug_classes).cuda()

    # 初始化损失函数
    # criterion0 = CrossEntropyLoss()
    # criterion1 = CrossEntropyLoss()
    # criterion2 = CrossEntropyLoss()
    # criterion3 = CrossEntropyLoss()
    # criterion4 = CrossEntropyLoss()
    # criterion5 = CrossEntropyLoss()
    # criterion6 = CrossEntropyLoss()
    # criterion7 = CrossEntropyLoss()
    #
    # criterion_list = [criterion0, criterion1, criterion2, criterion3, criterion4, criterion5, criterion6, criterion7]
    criterion_list = [CrossEntropyLoss() for i in range(8)]

    # 设置优化器
    opt = SGD(model.parameters(), lr=1e-2, momentum=0.9)
    scheduler = lr_scheduler.StepLR(opt, step_size=30, gamma=0.5)

    best_mean_recall = 0.0
    recent_recall_list = []
    for epoch in range(1, epochs + 1):
        model.train()
        print('\n++++++++++++++++++++++++++++++++++ [ EPOCH {} ] ++++++++++++++++++++++++++++++++++\n'.format(epoch))
        print("learning rate:", model.optimizer.state_dict()['param_group'][0]['lr'])

        pred_list = []
        label_list = []
        total_loss = 0.0
        # class_loss = [0 for i in range(8)]:wq

        for batch_idx, (data, label, tgt) in tqdm(enumerate(loader['train'])):
            # 收集rot_label序列
            label_list.extend(label)

            data, tgt = data.cuda(), tgt.cuda()
            # print(label, tgt)

            tgt_batch_list = []
            for i in range(8):
                tgt_batch_list.append(tgt.clone())

                for j, tgt_elem in enumerate(tgt):
                    if label[j] != i:
                        # 其他分类器若输入非本类图片，则target修改为4，即非本类
                        tgt_batch_list[i][j] = 4

            opt.zero_grad()
            output = model(data, label, "train")  # 得到含有8个分类器输出的列表
            # 收集预测标签序列，与目的标签一起进行评估
            # pred_list.extend(torch.argmax(output, dim=0).tolist())

            with torch.no_grad():
                for k in range(len(label)):  # 对每张图片的八个输出求8个loss，loss最小的分类器输出为该类
                    loss_k = []  # 记录第k张图片的8个loss
                    for idx in range(8):
                        one_out = output[idx][k].unsqueeze(0)
                        one_tgt = tgt_batch_list[idx][k].unsqueeze(0)
                        # if k == 1 & idx == 1:
                        #     print(one_out, one_tgt)
                        loss_k.append(cross_entropy(one_out, one_tgt))

                    # print(loss_k)

                    pred_list.append(np.argmin(loss_k))  # loss值最小的索引判定为该类

            batch_loss = criterion_list[0](output[0], tgt_batch_list[0])
            for idx in range(1, 8):
                batch_loss += criterion_list[idx](output[idx], tgt_batch_list[idx])

            total_loss += batch_loss.item()

            batch_loss.backward()

            opt.step()

        conf_mat = conf_matrix(pred_list, label_list, 8, False, [i for i in range(8)])
        tr_cls_recall, tr_cls_precision = cal_recall_precision(conf_mat, False, [i for i in range(8)])

        # 每几个epoch结束，进行测试调节优化，

        recent_recall_list.append(np.mean(tr_cls_recall))
        if epoch > 5:
            del recent_recall_list[0]

        if epoch > 100:
            if recent_recall_list[-1] > best_mean_recall:
                best_mean_recall = recent_recall_list[-1]
                save_name = 'resnet18' + str(best_mean_recall)
                save_ckt(model, opt, epoch, best_mean_recall, save_name)

        scheduler.step(epoch)

        if epoch % 10 == 0:
            valid(model, loader['valid'])


def valid(model, val_loader):
    print('\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Validation process...')
    model.eval()
    val_pred_list = []
    val_label_list = []
    with torch.no_grad():
        for index, (data, label) in tqdm(enumerate(val_loader)):
            val_label_list.append(int(label))
            labels = torch.tensor([label for i in range(4)]).cuda()
            # print(data)
            data = data.squeeze(1).cuda()
            data = torch.stack([data.clone(),
                                data.clone().rot90(1, [1, 2]),
                                data.clone().rot90(2, [1, 2]),
                                data.clone().rot90(3, [1, 2])])
            # print(data[0]-data[1])
            output = model(data, labels, "valid")

            val_loss = []  # 记录第k张图片的8个loss
            targets = torch.tensor([0, 1, 2, 3]).cuda()
            for idx in range(8):
                # if k == 1 & idx == 1:
                #     print(one_out, one_tgt)
                val_loss.append(cross_entropy(output[idx], targets))

            pred_label = np.argmin(val_loss)
            val_pred_list.append(int(pred_label))

        # print(val_pred_list)
        # print(val_label_list)
        val_conf_mat = conf_matrix(val_pred_list, val_label_list, 8, True, [i for i in range(8)])
        cal_recall_precision(val_conf_mat, True, [i for i in range(8)])


def get_threshold(ckt_path):
    aug_classes = 5
    num_classes = 8
    torch.set_printoptions(precision=2, threshold=100000, linewidth=10000)

    # get dataloader
    mean_std_path = './data/mean_std.json'
    data_root = './data/'
    # loader dict:'train','valid', 'test'
    loader = get_dataloader(mean_std_path, data_root)

    copy_resnet18 = deepcopy(resnet18(pretrained=False))
    # model = Net1FC(copy_resnet18, all_classes).cuda()
    model = Net8FC(copy_resnet18, num_classes, aug_classes).cuda()

    path = './backup/models/resnet180.09625'
    ckt = torch.load(path)
    model.load_state_dict(ckt['model'])

    model.eval()

    loss_list = [[] for i in range(8)]
    _pred_list = []
    _label_list = []
    with torch.no_grad():
        for index, (data, label) in tqdm(enumerate(loader['threshold'])):
            _label_list.append(int(label))
            labels = torch.tensor([label for i in range(4)]).cuda()
            # print(data)
            data = data.squeeze(1).cuda()
            data = torch.stack([data.clone(),
                                data.clone().rot90(1, [1, 2]),
                                data.clone().rot90(2, [1, 2]),
                                data.clone().rot90(3, [1, 2])])

            output = model(data, labels, "valid")

            val_loss = []  # 记录第k张图片的8个loss
            targets = torch.tensor([0, 1, 2, 3]).cuda()
            for idx in range(8):
                val_loss.append(cross_entropy(output[idx], targets))

            pred_label = np.argmin(val_loss)
            _pred_list.append(int(pred_label))

        # print(val_pred_list)
        # print(val_label_list)
        val_conf_mat = conf_matrix(_pred_list, _label_list, 8, True, [i for i in range(8)])
        cal_recall_precision(val_conf_mat, True, [i for i in range(8)])


def test():
    pass


if __name__ == '__main__':
    os.environ["CUDA_kVISIBLE_DEVICES"] = "7"
    train()
