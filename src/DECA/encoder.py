import math
import torch
import torch.nn as nn
import torchvision.models as models


class BaseEncoder(nn.Module):
    def __init__(self, outsize, last_op=None):
        super().__init__()
        self.feature_size = 2048
        self.outsize = outsize
        # self.encoder = resnet.load_ResNet50Model()  # out: 2048
        self._create_encoder()
        ### regressor
        self.layers = nn.Sequential(
            nn.Linear(self.feature_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, self.outsize)
        )
        self.last_op = last_op

    def forward_features(self, inputs):
        return self.encoder(inputs)

    def forward_features_to_output(self, features):
        parameters = self.layers(features)
        if self.last_op:
            parameters = self.last_op(parameters)
        return parameters

    def forward(self, inputs, output_features=False):
        features = self.forward_features(inputs)
        parameters = self.forward_features_to_output(features)
        if not output_features:
            return parameters
        return parameters, features

    def _create_encoder(self):
        raise NotImplementedError()

    def reset_last_layer(self):
        # initialize the last layer to zero to help the network 
        # predict the initial pose a bit more stable
        torch.nn.init.constant_(self.layers[-1].weight, 0)
        torch.nn.init.constant_(self.layers[-1].bias, 0)


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
       # self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x1 = self.layer4(x)

        x2 = self.avgpool(x1)
        x2 = x2.view(x2.size(0), -1)
        return x2


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
    
    
class ResnetEncoder(BaseEncoder):
    def __init__(self, outsize, last_op=None):
        super(ResnetEncoder, self).__init__(outsize, last_op)
   
    def copy_parameter_from_resnet(self, model, resnet_dict):
        cur_state_dict = model.state_dict()
        for name, param in list(resnet_dict.items())[0:None]:
            if name not in cur_state_dict:
                continue
            if isinstance(param, nn.Parameter):
                param = param.data
            try:
                cur_state_dict[name].copy_(param)
            except:
                print(name, ' is inconsistent!')
                continue
    
    def load_ResNet50Model(self):
        model = ResNet(Bottleneck, [3, 4, 6, 3])
        self.copy_parameter_from_resnet(model, models.resnet50(pretrained = True).state_dict())
        return model

    def _create_encoder(self):
        self.encoder = self.load_ResNet50Model()  # out: 2048