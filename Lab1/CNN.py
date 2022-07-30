import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

class CNNLayer(nn.Module):
    def __init__(self, in_feature_maps, out_feature_maps, downsample=True):
        super(CNNLayer, self).__init__()

        self.stride = 2 if downsample == True else 1
        self.conv0 = n, stride = self.stride, padding = 1)
        self.bn0 = nn.BatchNorm2d(out_feature_maps)n.Conv2d(in_feature_maps, out_feature_maps, 3
        self.conv1 = nn.Conv2d(out_feature_maps, out_feature_maps, 3, stride = 1, padding = 1)
        self.bn1 = nn.BatchNorm2d(out_feature_maps)
        self.Maxpool2 = nn.MaxPool2d(2,2)
        self.dropout = nn.Dropout(0.2)
    def forward(self, input):
        x = F.relu(self.dropout(self.bn0(self.conv0(input))))
        x = F.relu(self.dropout(self.bn1(self.conv1(x))))
        x = self.Maxpool2(x)
        return x

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv0 = nn.Conv2d(3, 64, 3, stride=1, padding=1)  # (64*64*3) -> (64*64*64)
        self.bn0 = nn.BatchNorm2d(64)

        self.maxpool = nn.MaxPool2d(3, 2, 1)
        self.cnnlayer1 = CNNLayer(64, 128, False)
        self.cnnlayer2 = CNNLayer(128, 256, False)
        self.cnnlayer3 = CNNLayer(256, 512, False)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(512, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.dropout1 = nn.Dropout(0.2)
        self.fc = nn.Linear(256, 200)

        self.dropout = nn.Dropout(0.15)

    def forward(self, input):
        x = F.relu(self.bn0(self.dropout(self.conv0(input))))
        x = self.maxpool(x)
        x = self.cnnlayer1(x)
        x = self.cnnlayer2(x)
        x = self.cnnlayer3(x)
        x = self.avgpool(x)
        x = x.view(x.shape[0], -1)
        x = self.dropout1(x)
        x = F.relu(self.dropout1(self.bn1(self.fc1(x))))
        x = self.fc(x)

        return x
