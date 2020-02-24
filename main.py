"""
@Author: Tsingwaa Tsang
@Date: 2020-02-16 22:25:05
@LastEditors: Tsingwaa Tsang
@LastEditTime: 2020-02-16 22:25:05
@Description: Null
"""


import os
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import SGD, lr_scheduler
from torchvision.models import resnet18
from copy import deepcopy
from tqdm import tqdm

from data_loader.my_dataloader import train_dataloader, test_dataloader
from model.model import Net1FC
from model.metric import *

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def main():
    # 定义变量
    all_classes = 32
    EPOCHS = 1

    copy_resnet18 = deepcopy(resnet18(pretrained=True))
    model = Net1FC(copy_resnet18, all_classes).cuda()

    # 初始化损失函数
    criterion = CrossEntropyLoss()

    # 设置优化器
    opt = SGD(model.parameters(), lr=1e-2, momentum=0.9)
    scheduler = lr_scheduler.StepLR(opt, step_size=40, gamma=0.1)

    model.train()
    total_loss = 0.0
    pred_list = []
    tgt_list = []

    for epoch in range(1, EPOCHS+1):
        for batch_idx, (data, tgt) in tqdm(enumerate(train_dataloader)):
            tgt_list.extend(tgt)

            data, tgt = data.cuda(), tgt.cuda()

            opt.zero_grad()
            output = model(data)
            pred_list.extend(torch.argmax(output, dim=0).tolist())

            loss = criterion(output, tgt)
            total_loss += loss.item()
            loss.backward()

            opt.step()

        # 每几个epoch结束，进行测试调节优化，
        if epoch % 5 == 0:

        scheduler.step()
