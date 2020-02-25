"""
@Author: Tsingwaa Tsang
@Date: 2020-02-16 22:25:05
@LastEditors: Tsingwaa Tsang
@LastEditTime: 2020-02-16 22:25:05
@Description: Null
"""

import os
from copy import deepcopy

from torch.nn import CrossEntropyLoss
from torch.optim import SGD, lr_scheduler
from torchvision.models import resnet18
from tqdm import tqdm

from data_loader.my_dataloader import train_dataloader
from model.metric import *
from model.model import Net8FC

os.environ["CUDA_kVISIBLE_DEVICES"] = "0"


def main():
    # 定义变量
    # all_classes = 32
    aug_classes = 5
    num_classes = 8

    epochs = 1

    copy_resnet18 = deepcopy(resnet18(pretrained=True))
    # model = Net1FC(copy_resnet18, all_classes).cuda()
    model = Net8FC(copy_resnet18, num_classes, aug_classes).cuda()

    # 初始化损失函数
    criterion0 = CrossEntropyLoss()
    criterion1 = CrossEntropyLoss()
    criterion2 = CrossEntropyLoss()
    criterion3 = CrossEntropyLoss()
    criterion4 = CrossEntropyLoss()
    criterion5 = CrossEntropyLoss()
    criterion6 = CrossEntropyLoss()
    criterion7 = CrossEntropyLoss()

    criterion_list = {
        0: criterion0, 1: criterion1, 2: criterion2, 3: criterion3,
        4: criterion4, 5: criterion5, 6: criterion6, 7: criterion7
    }

    # 设置优化器
    opt = SGD(model.parameters(), lr=1e-2, momentum=0.9)
    scheduler = lr_scheduler.StepLR(opt, step_size=40, gamma=0.1)

    model.train()
    total_loss = 0.0
    pred_list = []
    target_list = []

    # 作为其余7个多余的损坏函数的拟合目标
    pesdo_target = torch.tensor([0.01 for i in range(4)]).cuda()

    for epoch in range(1, epochs + 1):
        for batch_idx, (data, label, target) in tqdm(enumerate(train_dataloader)):
            # 收集目的标签序列
            target_list.extend(target)

            data, target = data.cuda(), target.cuda()

            opt.zero_grad()
            output = model(data)
            # 收集预测标签序列，与目的标签一起进行评估
            pred_list.extend(torch.argmax(output, dim=0).tolist())

            # 对待通过的通道进行评判
            loss = criterion_list[label](output, target)
            for idx in range(8):
                if idx != label:
                    for j in enumerate(target):
                    loss += criterion_list[i](output, pesdo_target)

            total_loss += loss.item()

            my_metric2()

            loss.backward()

            opt.step()

        # 每几个epoch结束，进行测试调节优化，
        if epoch % 5 == 0:
            pass

        scheduler.step(epoch)
