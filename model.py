import torch
import torch.nn as nn
import math
import torch.nn.functional as F

from torchvision import models
from torch.autograd import Variable



class resnet101(nn.Module):
    def __init__(self, age_clss):
        super(resnet101, self).__init__()
        self.resNet = models.resnet101(pretrained=True)
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(1000),
            nn.Dropout(0.5),
            nn.Linear(1000, age_clss)
        )
        # self.fc1 = nn.Linear(2048, 512)
        # self.bn1 = nn.BatchNorm1d(512)
        # self.fc2 = nn.Linear(512, 256)
        # self.bn2 = nn.BatchNorm1d(256)
        # self.age_pred = nn.Linear(256, age_clss)

    def forward(self, x):
        x = self.resNet.forward(x)
        x = self.classifier(x)
        x = F.log_softmax(x)
        # x = self.resNet.conv1(x)
        # x = self.resNet.bn1(x)
        # x = self.resNet.relu(x)
        # x = self.resNet.maxpool(x)
        #
        # x = self.resNet.layer1(x)
        # x = self.resNet.layer2(x)
        # x = self.resNet.layer3(x)
        # x = self.resNet.layer4(x)
        # x = self.resNet.avgpool(x)
        # x = x.view(x.size(0), -1)
        # x = F.relu(self.fc1(x))
        # # print(x.size())
        # x = F.relu(self.bn1(self.fc1(x)))
        # x = F.relu(self.bn2(self.fc2(x)))
        # age = F.softmax(self.age_pred(x), dim=1)
        # # gender = F.softmax(self.gender_pred(x), dim=1)
        return  x


class DenseNet_Model(nn.Module):
    ################
    ## densenet121##
    ################
    def __init__(self, age_clss):
        super(DenseNet_Model, self).__init__()
        self.denseNet = models.densenet121(pretrained=True)
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)

        self.age_pred = nn.Linear(256, age_clss)
        self.gender_pred = nn.Linear(256, 2)

    def forward(self, x):
        features = self.denseNet.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)

        out = F.relu(self.bn1(self.fc1(out)))
        out = F.relu(self.bn2(self.fc2(out)))
        age = F.softmax(self.age_pred(out), dim=1)
        gender = F.softmax(self.gender_pred(out), dim=1)
        return age, gender


class VGG16_net(nn.Module):
    ###########
    ## VGG16 ##
    ###########
    def __init__(self, age_clss):
        super(VGG16_net, self).__init__()
        self.vgg16 = models.vgg16(pretrained=True)
        self.classifier = nn.Sequential(
                          nn.ReLU(True),
                          nn.Linear(512 * 7 * 7, 100)
                          )
    #     self.fc1 = nn.Linear(4096, 512)
    #     self.bn1 = nn.BatchNorm1d(512)
    #     self.fc2 = nn.Linear(512, 256)
    #     self.bn2 = nn.BatchNorm1d(256)
    #
    #     self.classifier = nn.Sequential(
    #         nn.Linear(512 * 7 * 7, 4096),
    #         nn.ReLU(True),
    #         nn.Dropout(),
    #         nn.Linear(4096, 4096),
    #         nn.ReLU(True),
    #         nn.Dropout(),
    #     )
    #
    #     self.age_pred = nn.Linear(256, age_clss)
    #     self.gender_pred = nn.Linear(256, 2)
    #
    def forward(self, x):
        x = self.vgg16.features(x)
        x = self.vgg16.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        x = F.softmax(x)
    #     x = F.relu(self.bn1(self.fc1(x)))
    #     x = F.relu(self.bn2(self.fc2(x)))
    #     age = F.softmax(self.age_pred(x),dim=1)
    #     gender = F.softmax(self.gender_pred(x), dim=1)
    #
        return x
