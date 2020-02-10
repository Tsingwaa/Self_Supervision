"""
* @Author: Tsingwaa Tsang
* @Date: 2020-02-06 15:09:19
* @LastEditors: Tsingwaa Tsang
* @LastEditTime: 2020-02-11 00:17:42
* @Description: Null
"""


import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel
from torchvision.models import resnet18
from copy import deepcopy


class MnistModel(BaseModel):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


my_resnet = deepcopy(resnet18(True))


class Net(nn.Module):
    def __init__(self, model, num_classes, aug_classes):
        """
        * @Description: 
        * @Args: 
        * @Return: 
        """
        super(Net, self).__init__()
        # 先去除最后一层fc层
        self.resnet_layer = nn.Sequential(*list(model.children())[:-1])
        # 再定义四类的fc层
        self.fc1 = nn.Linear(model.fc.in_features, aug_classes)
        self.fc2 = nn.Linear(model.fc.in_features, aug_classes)
        self.fc3 = nn.Linear(model.fc.in_features, aug_classes)
        self.fc4 = nn.Linear(model.fc.in_features, aug_classes)
        self.fc5 = nn.Linear(model.fc.in_features, aug_classes)
        self.fc6 = nn.Linear(model.fc.in_features, aug_classes)
        self.fc7 = nn.Linear(model.fc.in_features, aug_classes)
        self.fc8 = nn.Linear(model.fc.in_features, aug_classes)

    def foward(self, x, label):
        fc_dict = {
            1: self.fc1, 2: self.fc2, 3: self.fc3, 4: self.fc4,
            5: self.fc5, 6: self.fc6, 7: self.fc7, 8: self.fc8
        }
        output_x = self.resnet_layer(x)
        output_x = fc_dict[label](output_x)

        return output_x
