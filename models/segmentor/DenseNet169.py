import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from collections import OrderedDict
import re
import pdb

__all__ = ['DenseNet', 'densenet121', 'densenet169', 'densenet201', 'densenet161']

model_urls = {
    'densenet121': 'https://download.pytorch.org/models/densenet121-a639ec97.pth',
    'densenet169': 'https://download.pytorch.org/models/densenet169-b2777c0a.pth',
    'densenet201': 'https://download.pytorch.org/models/densenet201-c1103571.pth',
    'densenet161': 'https://download.pytorch.org/models/densenet161-8d451a50.pth',
}


def densenet121(pretrained, **kwargs):
    r"""Densenet-121 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = DenseNet(num_init_features=64, growth_rate=32, block_config=(6, 12, 24, 16),
                     **kwargs)
    if pretrained:
        print("----------------------------- densenet121 --------------")
        # '.'s are no longer allowed in module names, but pervious _DenseLayer
        # has keys 'norm.1', 'relu.1', 'conv.1', 'norm.2', 'relu.2', 'conv.2'.
        # They are also in the checkpoints in model_urls. This pattern is used
        # to find such keys.
        pattern = re.compile(
            r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
        state_dict = model_zoo.load_url(model_urls['densenet121'])
        for key in list(state_dict.keys()):
            res = pattern.match(key)
            if res:
                new_key = res.group(1) + res.group(2)
                state_dict[new_key] = state_dict[key]
                del state_dict[key]

        new_params = model.state_dict().copy()
        for name, param in new_params.items():
            print(" ------------------------------- name : ", name)
            if name in state_dict and param.size() == state_dict[name].size():
                new_params[name].copy_(state_dict[name])
                print('copy {}'.format(name))

        model.load_state_dict(new_params)

    # model.classifier = None
    features = model.features
    features.block0 = nn.Sequential(features.conv0, features.norm0, features.relu0)
    features.pool0 = nn.Sequential(features.pool0)

    features.denseblock1 = nn.Sequential(*list(features.denseblock1))
    features.transition1 = nn.Sequential(*list(features.transition1)[:-1])

    features.denseblock2 = nn.Sequential(*list(features.denseblock2))
    features.transition2 = nn.Sequential(*list(features.transition2)[:-1])

    features.denseblock3 = nn.Sequential(*list(features.denseblock3))
    features.transition3 = nn.Sequential(*list(features.transition3)[:-1])

    model.features = features
    return model


def densenet161(pretrained=False, **kwargs):
    r"""Densenet-161 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = DenseNet(num_init_features=96, growth_rate=48, block_config=(6, 12, 36, 24),
                     **kwargs)
    if pretrained:
        print("----------------------------- densenet161 --------------")

        # '.'s are no longer allowed in module names, but pervious _DenseLayer
        # has keys 'norm.1', 'relu.1', 'conv.1', 'norm.2', 'relu.2', 'conv.2'.
        # They are also in the checkpoints in model_urls. This pattern is used
        # to find such keys.
        pattern = re.compile(
            r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
        state_dict = model_zoo.load_url(model_urls['densenet161'])
        for key in list(state_dict.keys()):
            res = pattern.match(key)
            if res:
                new_key = res.group(1) + res.group(2)
                state_dict[new_key] = state_dict[key]
                del state_dict[key]

        new_params = model.state_dict().copy()
        for name, param in new_params.items():
            print(" ------------------------------- name : ", name)
            if name in state_dict and param.size() == state_dict[name].size():
                new_params[name].copy_(state_dict[name])
                print('copy {}'.format(name))

        model.load_state_dict(new_params)
    # print("--------------------- model:", model)
    features = model.features
    features.block0 = nn.Sequential(features.conv0, features.norm0, features.relu0)
    features.pool0 = nn.Sequential(features.pool0)

    features.denseblock1 = nn.Sequential(*list(features.denseblock1))
    features.transition1 = nn.Sequential(*list(features.transition1)[:-1])

    features.denseblock2 = nn.Sequential(*list(features.denseblock2))
    features.transition2 = nn.Sequential(*list(features.transition2)[:-1])

    features.denseblock3 = nn.Sequential(*list(features.denseblock3))
    features.transition3 = nn.Sequential(*list(features.transition3)[:-1])
    model.features = features
    return model

def densenet169(pretrained=False, **kwargs):
    r"""Densenet-169 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = DenseNet(num_init_features=64, growth_rate=32, block_config=(6, 12, 32, 32),
                     **kwargs)
    if pretrained:
        print("----------------------------- densenet169 --------------")

        # '.'s are no longer allowed in module names, but pervious _DenseLayer
        # has keys 'norm.1', 'relu.1', 'conv.1', 'norm.2', 'relu.2', 'conv.2'.
        # They are also in the checkpoints in model_urls. This pattern is used
        # to find such keys.
        pattern = re.compile(
            r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
        state_dict = model_zoo.load_url(model_urls['densenet169'])
        for key in list(state_dict.keys()):
            res = pattern.match(key)
            if res:
                new_key = res.group(1) + res.group(2)
                state_dict[new_key] = state_dict[key]
                del state_dict[key]

        new_params = model.state_dict().copy()
        for name, param in new_params.items():
            # print(" ------------------------------- name : ", name)
            if name in state_dict and param.size() == state_dict[name].size():
                new_params[name].copy_(state_dict[name])
                # print('copy {}'.format(name))

        model.load_state_dict(new_params)
    model.classifier = None
    features = model.features
    features.block0 = nn.Sequential(features.conv0, features.norm0, features.relu0)
    features.pool0 = nn.Sequential(features.pool0)

    features.denseblock1 = nn.Sequential(*list(features.denseblock1))
    features.transition1 = nn.Sequential(*list(features.transition1)[:-1])

    features.denseblock2 = nn.Sequential(*list(features.denseblock2))
    features.transition2 = nn.Sequential(*list(features.transition2)[:-1])

    features.denseblock3 = nn.Sequential(*list(features.denseblock3))
    features.transition3 = nn.Sequential(*list(features.transition3)[:-1])

    model.features = features
    return model


def densenet201(pretrained=False, **kwargs):
    r"""Densenet-201 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = DenseNet(num_init_features=64, growth_rate=32, block_config=(6, 12, 48, 32),
                     **kwargs)
    if pretrained:
        print("----------------------------- densenet201 --------------")

        # '.'s are no longer allowed in module names, but pervious _DenseLayer
        # has keys 'norm.1', 'relu.1', 'conv.1', 'norm.2', 'relu.2', 'conv.2'.
        # They are also in the checkpoints in model_urls. This pattern is used
        # to find such keys.
        pattern = re.compile(
            r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
        state_dict = model_zoo.load_url(model_urls['densenet201'])
        for key in list(state_dict.keys()):
            res = pattern.match(key)
            if res:
                new_key = res.group(1) + res.group(2)
                state_dict[new_key] = state_dict[key]
                del state_dict[key]

        new_params = model.state_dict().copy()
        for name, param in new_params.items():
            print(" ------------------------------- name : ", name)
            # print("param.size() : ", param.size())
            # print("state_dict[name].size() : ", state_dict[name].size())
            if name in state_dict and param.size() == state_dict[name].size():
                new_params[name].copy_(state_dict[name])
                print('copy {}'.format(name))

        model.load_state_dict(new_params)
    model.classifier = None
    features = model.features
    features.block0 = nn.Sequential(features.conv0, features.norm0, features.relu0)
    features.pool0 = nn.Sequential(features.pool0)

    features.denseblock1 = nn.Sequential(*list(features.denseblock1))
    features.transition1 = nn.Sequential(*list(features.transition1)[:-1])

    features.denseblock2 = nn.Sequential(*list(features.denseblock2))
    features.transition2 = nn.Sequential(*list(features.transition2)[:-1])

    features.denseblock3 = nn.Sequential(*list(features.denseblock3))
    features.transition3 = nn.Sequential(*list(features.transition3)[:-1])

    return model



class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size *
                                           growth_rate, kernel_size=1, stride=1, bias=False)),
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                                           kernel_size=3, stride=1, padding=1, bias=False)),
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


class DenseNet(nn.Module):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_
    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
    """

    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=4, drop_rate=0, num_classes=9):

        super(DenseNet, self).__init__()

        # First convolution
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        # Final batch norm
        # self.features.add_module('norm5', nn.BatchNorm2d(num_features))

        # ----------------------------  laeyr121 -----------------------------
        # self.deconv0 = nn.Conv2d(3, num_classes, kernel_size=3, stride=1, padding=1, groups=1, bias=False)
        # self.deconv1 = nn.Sequential(nn.Conv2d(64, num_classes, kernel_size=1),
        #                              nn.UpsamplingBilinear2d(scale_factor=2))
        # self.deconv2 = nn.Sequential(nn.Conv2d(64, num_classes, kernel_size=1),
        #                              nn.UpsamplingBilinear2d(scale_factor=4))
        # self.deconv3 = nn.Sequential(nn.Conv2d(128, num_classes, kernel_size=1),
        #                              nn.UpsamplingBilinear2d(scale_factor=4))
        # self.deconv4 = nn.Sequential(nn.Conv2d(256, num_classes, kernel_size=1),
        #                              nn.UpsamplingBilinear2d(scale_factor=8))
        # self.deconv5 = nn.Sequential(nn.Conv2d(512, num_classes, kernel_size=1),
        #                              nn.UpsamplingBilinear2d(scale_factor=16))


        # ---------------------------- upper 169 laeyr -----------------------------
        self.deconv0 = nn.Conv2d(3, num_classes, kernel_size=3, stride=1, padding=1, groups=1, bias=False)
        self.deconv1 = nn.Sequential(nn.Conv2d(64, num_classes, kernel_size=1),
                                             nn.UpsamplingBilinear2d(scale_factor=2))
        self.deconv2 = nn.Sequential(nn.Conv2d(64, num_classes, kernel_size=1),
                                             nn.UpsamplingBilinear2d(scale_factor=4))
        self.deconv3 = nn.Sequential(nn.Conv2d(128, num_classes, kernel_size=1),
                                             nn.UpsamplingBilinear2d(scale_factor=4))
        self.deconv4 = nn.Sequential(nn.Conv2d(256, num_classes, kernel_size=1),
                                             nn.UpsamplingBilinear2d(scale_factor=8))
        self.deconv5 = nn.Sequential(nn.Conv2d(640, num_classes, kernel_size=1),
                                             nn.UpsamplingBilinear2d(scale_factor=16))
        # self.deconv0 = nn.ConvTranspose2d(3, num_classes, kernel_size=3, stride=1, padding=1, groups=1, bias=False)
        # self.deconv1 = nn.ConvTranspose2d(64, num_classes, kernel_size=6, stride=4, padding=1, groups=1, bias=False)
        # self.deconv2 = nn.ConvTranspose2d(128, num_classes, kernel_size=6, stride=4, padding=1, groups=1, bias=False)
        # self.deconv3 = nn.ConvTranspose2d(256, num_classes, kernel_size=10, stride=8, padding=1, groups=1, bias=False)
        # self.deconv4 = nn.ConvTranspose2d(640, num_classes, kernel_size=18, stride=16, padding=1, groups=1, bias=False)
        # self.deconv5 = nn.ConvTranspose2d(1664, num_classes, kernel_size=32, stride=35, padding=1, groups=1, bias=False)

        # ---------------------------- upper 201 laeyr -----------------------------
        # self.deconv0 = nn.ConvTranspose2d(3, num_classes, kernel_size=3, stride=1, padding=1, groups=1, bias=False)
        # self.deconv1 = nn.ConvTranspose2d(64, num_classes, kernel_size=6, stride=4, padding=1, groups=1, bias=False)
        # self.deconv2 = nn.ConvTranspose2d(128, num_classes, kernel_size=6, stride=4, padding=1, groups=1, bias=False)
        # self.deconv3 = nn.ConvTranspose2d(256, num_classes, kernel_size=10, stride=8, padding=1, groups=1, bias=False)
        # self.deconv4 = nn.ConvTranspose2d(896, num_classes, kernel_size=18, stride=16, padding=1, groups=1, bias=False)
        # self.deconv5 = nn.ConvTranspose2d(1920, num_classes, kernel_size=32, stride=35, padding=1, groups=1, bias=False)

        # ----------------------- classifier -----------------------
        # self.bn_class = nn.BatchNorm2d(num_classes * 5)
        self.conv_class = nn.Conv2d(num_classes * 6, num_classes, kernel_size=1, padding=0)
        # --------

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x_conct0 = x
        x_conct1 = x = self.features.block0(x)  # 1/2
        x_conct2 = x = self.features.pool0(x)  # 1/4
        x = self.features.denseblock1(x)
        x_conct3 = x = self.features.transition1(x)
        x = F.avg_pool2d(x, kernel_size=2, stride=2)  # 1/8
        x = self.features.denseblock2(x)
        x_conct4 =x = self.features.transition2(x)
        x = F.avg_pool2d(x, kernel_size=2, stride=2)  # 1/16
        x = self.features.denseblock3(x)
        x_conct5 = x = self.features.transition3(x)


        # print("--------------------- x 0 :", x_conct0.shape)
        # print("--------------------- x 1 :", x_conct1.shape)
        # print("--------------------- x 2 :", x_conct2.shape)
        # print("--------------------- x 3 :", x_conct3.shape)
        # print("--------------------- x 4 :", x_conct4.shape)
        # print("--------------------- x 5 :", x_conct5.shape)

        x_deconv0 = self.deconv0(x_conct0)
        x_deconv1 = self.deconv1(x_conct1)
        x_deconv2 = self.deconv2(x_conct2)
        x_deconv3 = self.deconv3(x_conct3)
        x_deconv4 = self.deconv4(x_conct4)
        x_deconv5 = self.deconv5(x_conct5)


        # print("--------------------- x 0 :", x_deconv0.shape)
        # print("--------------------- x 1 :", x_deconv1.shape)
        # print("--------------------- x 2 :", x_deconv2.shape)
        # print("--------------------- x 3 :", x_deconv3.shape)
        # print("--------------------- x 4 :", x_deconv4.shape)
        # print("--------------------- x 5 :", x_deconv5.shape)

        x = torch.cat([x_deconv0, x_deconv1, x_deconv2, x_deconv3, x_deconv4, x_deconv5], 1)
        # x = torch.cat([x_deconv1, x_deconv2, x_deconv3, x_deconv4,x_deconv5], 1)
        # ----------------------- classifier -----------------------
        # x = self.conv_class(F.relu(self.bn_class(x)))
        x = self.conv_class(x)
        # ------
        return x
