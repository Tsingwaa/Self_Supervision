"""
@Author: Tsingwaa Tsang
@Date: 2020-02-16 22:25:05
@LastEditors: Tsingwaa Tsang
@LastEditTime: 2020-02-16 22:25:05
@Description: Null
"""

import os
from copy import deepcopy

import numpy as np
from torch.nn import CrossEntropyLoss
from torch.optim import SGD, lr_scheduler
from torchvision.models import resnet18
from tqdm import tqdm

from data_loader.my_dataloader import get_dataloader
from model.model import Net8FC

os.environ["CUDA_kVISIBLE_DEVICES"] = "0"


def train():
    # 定义变量
    # all_classes = 32
    aug_classes = 5
    num_classes = 8

    epochs = 3

    # get dataloader
    mean_std_path = './data/mean_std.json'
    data_root = './data/'
    loader = get_dataloader(mean_std_path, data_root)  # loader dict:'train','valid', 'test'

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
    scheduler = lr_scheduler.StepLR(opt, step_size=40, gamma=0.1)

    model.train()

    # 作为其余7个多余的损坏函数的拟合目标
    # pesdo_target = torch.tensor([0.01 for i in range(4)]).cuda()

    for epoch in range(1, epochs + 1):
        pred_list = []
        label_list = []
        total_loss = 0.0

        for batch_idx, (data, label, tgt) in tqdm(enumerate(loader['train'])):
            # 收集rot_label序列
            label_list.extend(tgt)

            data, tgt = data.cuda(), tgt.cuda()
            # print(label, tgt)

            tgt_batch_list = []
            for i in range(8):
                tgt_batch_list.append(tgt.clone())

                for j, tgt_elem in enumerate(tgt):
                    if label[j] != i:
                        tgt_batch_list[i][j] = 4  # 其他分类器若输入非本类图片，则target修改为4，即非本类

            opt.zero_grad()
            output = model(data, label, "train")  # 得到含有8个分类器输出的列表
            # 收集预测标签序列，与目的标签一起进行评估
            # pred_list.extend(torch.argmax(output, dim=0).tolist())

            for k in range(batch_idx):  # 对每张图片的八个输出求8个loss，loss最小的分类器输出为该类
                loss_k = []
                for idx in range(8):
                    loss_k.append(criterion_list[idx](output[idx][k], tgt_batch_list[idx][k]).item())

                pred_list.append(np.argmin(loss_k))

            batch_loss = criterion_list[0](output[0], tgt_batch_list[0])
            for idx in range(1, 8):
                batch_loss += criterion_list[idx](output[idx], tgt_batch_list[idx])

            total_loss += batch_loss.item()

            # my_metric2()

            batch_loss.backward()

            opt.step()

        print()
        # 每几个epoch结束，进行测试调节优化，
        if epoch % 5 == 0:
            valid(model, loader['valid'])

        scheduler.step(epoch)


def valid(model, val_loader):
    for index, (data, label, rot_label) in enumerate(val_loader):
        pass


if __name__ == '__main__':
    train()
