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

from data_loader.my_dataloader import get_dataloader
from model.metric import *
from model.model import Net8FC

os.environ["CUDA_kVISIBLE_DEVICES"] = "0"


def main():
    # 定义变量
    # all_classes = 32
    aug_classes = 5
    num_classes = 8

    epochs = 1

    # get dataloader
    mean_std_path = './data/mean_std.json'
    data_root = './data/'
    loader = get_dataloader(mean_std_path, data_root)

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
    # pesdo_target = torch.tensor([0.01 for i in range(4)]).cuda()

    for epoch in range(1, epochs + 1):
        for batch_idx, (data, label, target) in tqdm(enumerate(loader['train'])):
            # 收集目的标签序列
            target_list.extend(target)

            data, target = data.cuda(), target.cuda()
            target0 = target1 = target2 = target3 = target4 = target5 = target6 = target7 = target
            target_batch_dict = {
                0: target0, 1: target1, 2: target2, 3: target3,
                4: target4, 5: target5, 6: target6, 7: target7
            }
            for i in range(8):
                for j, target_elem in enumerate(target):
                    if target_elem != i:
                        target_batch_dict[i][j] = 4  # 设置为其他未知类rot_label=4

            print(target_batch_dict[0])

            opt.zero_grad()
            output = model(data)
            # 收集预测标签序列，与目的标签一起进行评估
            pred_list.extend(torch.argmax(output, dim=0).tolist())

            # 对待通过的通道进行评判
            loss = criterion_list[0](output[0], target_batch_dict[0])
            for idx in range(start=1, stop=8):
                loss += criterion_list[idx](output[idx], target_batch_dict[idx])

            total_loss += loss.item()

            # my_metric2()

            loss.backward()

            opt.step()

        # 每几个epoch结束，进行测试调节优化，
        if epoch % 5 == 0:
            pass

        scheduler.step(epoch)


if __name__ == '__main__':
    main()
