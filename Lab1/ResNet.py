import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

class ResNetLayer(nn.Module):
    def __init__(self, in_feature_maps, out_feature_maps, downsample = True):
        super(ResNetLayer, self).__init__()

        self.stride = 2 if downsample == True else 1
        self.conv0 = nn.Conv2d(in_feature_maps, out_feature_maps, 3, stride = self.stride, padding = 1)
        self.bn0 = nn.BatchNorm2d(out_feature_maps)
        self.conv1 = nn.Conv2d(out_feature_maps, out_feature_maps, 3, stride = 1, padding = 1)
        self.bn1 = nn.BatchNorm2d(out_feature_maps)

        self.skipconn_cnn = nn.Conv2d(in_feature_maps, out_feature_maps, kernel_size=1, stride=self.stride, padding = 0)
        self.skipconn_bn = nn.BatchNorm2d(out_feature_maps)
        self.dropout = nn.Dropout(0.2)
    def forward(self, input):
        identity = input
        x = F.relu(self.bn0(self.dropout(self.conv0(input))))
        x = self.bn1(self.conv1(x))
        x += self.skipconn_bn(self.skipconn_cnn(identity))
        x = F.relu(self.dropout(x))
        return x


class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        
        self.conv0 = nn.Conv2d(3, 64, 5, stride = 1, padding = 2) #(64*64*3) -> (64*64*64)
        self.bn0 = nn.BatchNorm2d(64)

        self.maxpool = nn.MaxPool2d(3, 2, 1)
        self.resnetlayer1 = ResNetLayer(64, 64, False)
        self.resnetlayer2 = ResNetLayer(64, 128)
        self.resnetlayer3 = ResNetLayer(128, 128, False)
        self.resnetlayer4 = ResNetLayer(128, 256)
        self.resnetlayer5 = ResNetLayer(256, 256, False)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout1 = nn.Dropout(0.2)
        self.fc = nn.Linear(256, 200)

        self.dropout = nn.Dropout(0.15)

    def forward(self, input):
        x = F.relu(self.bn0(self.dropout(self.conv0(input))))
        x = self.maxpool(x)
        x = self.resnetlayer1(x)
        x = self.resnetlayer2(x)
        x = self.resnetlayer3(x)
        x = self.resnetlayer4(x)
        x = self.resnetlayer5(x)
        x = self.avgpool(x)
        x = x.view(x.shape[0], -1)
        x = self.dropout1(x)
        x = self.fc(x)

        return x