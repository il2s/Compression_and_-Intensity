import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm3d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv3d(num_input_features, bn_size *
                        growth_rate, kernel_size=1, stride=1, bias=False)),
        self.add_module('norm2', nn.BatchNorm3d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv3d(bn_size * growth_rate, growth_rate,
                        kernel_size=3, stride=1, padding=1, bias=False)),
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout3d(new_features, p=self.drop_rate, training=self.training)
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
        self.add_module('norm', nn.BatchNorm3d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv3d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))

        self.add_module('pool_norm', nn.BatchNorm3d(num_output_features))
        self.add_module('pool_relu', nn.ReLU(inplace=True))
        self.add_module('pool', nn.Conv3d(num_output_features, num_output_features, kernel_size=2, stride=2))
        #self.add_module('pool', nn.MaxPool3d( kernel_size=2, stride=2))

class Spatial(nn.Module):

    def __init__(self, channels):
        super(Spatial, self).__init__()
        self.bn = nn.BatchNorm3d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.sse1 = nn.Conv3d(channels, 1, kernel_size = 1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x2 = self.bn(x)
        x2 = self.relu(x2)
        x2 = self.sse1(x2)
        x2 = self.sigmoid(x2)

        return x2


class SCSEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SCSEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)

        self.channel_excitation = nn.Sequential(nn.Conv3d(channel, channel//reduction, kernel_size=1, padding=0),
                                                #nn.BatchNorm3d(channel//reduction),
                                                nn.ReLU(inplace=True),
                                                nn.Conv3d(channel // reduction, channel, kernel_size=1, padding=0),
                                                nn.Softmax(dim=1))

        self.spatial_se = nn.Sequential(nn.Conv3d(channel, 1, kernel_size=1, stride=1, padding=0),
                                        nn.Softmax(dim=1))

    def forward(self, x):
        chn_se = self.avg_pool(x)
        chn_se = self.channel_excitation(chn_se)
        chn_se = torch.mul(x, chn_se)

        spa_se = self.spatial_se(x)
        spa_se = torch.mul(x, spa_se)

        out_se = chn_se + spa_se
        return out_se


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

    def __init__(self, growth_rate=32, block_config=(4, 4, 4, 16),
                 num_init_features=64, bn_size=4, drop_rate=0, num_classes=9):

        super(DenseNet, self).__init__()

        # First three convolutions
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv3d(3, num_init_features, kernel_size=3, stride=1, padding=1)),
            ('norm0', nn.BatchNorm3d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('conv1', nn.Conv3d(num_init_features, num_init_features, kernel_size=3, stride=1, padding=1, bias=False)),
            ('norm1', nn.BatchNorm3d(num_init_features)),
            ('relu1', nn.ReLU(inplace=True)),
            ('conv2', nn.Conv3d(num_init_features, num_init_features, kernel_size=3, stride=1, padding=1,bias=False)),
            ('norm2', nn.BatchNorm3d(num_init_features)),
            ('relu2', nn.ReLU(inplace=True)),
        ]))
        self.conv_pool_first = nn.Conv3d(num_init_features, num_init_features, kernel_size=2, stride=2, padding=0,
                                          bias=False)
        #self.conv_pool_first = nn.MaxPool3d(kernel_size=2, stride=2)

        # Each denseblock
        num_features = num_init_features
        num_features_list=[]
        self.dense_blocks = nn.ModuleList([])
        self.scse_blocks = nn.ModuleList([])
        self.transit_blocks = nn.ModuleList([])
        self.upsampling_blocks = nn.ModuleList([])
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)

            self.dense_blocks.append(block)
            num_features = num_features + num_layers * growth_rate
            #se = SEModule(num_features, reduction=8)
            scse = SCSEBlock(num_features, reduction=16)
            self.scse_blocks.append(scse)

            up_block = nn.ConvTranspose3d(num_features, num_classes, kernel_size=2 ** (i + 1) + 2,
                                          stride=2 ** (i + 1),
                                          padding=1, groups=1, bias=False)
            self.upsampling_blocks.append(up_block)
            num_features_list.append(num_features)

            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
                self.transit_blocks.append(trans)
                num_features = num_features // 2

        # ----------------------- classifier -----------------------
        self.bn_class_3_3_first = nn.BatchNorm3d(num_init_features)
        self.conv_class_3_3_first = nn.Conv3d(num_init_features, num_classes, kernel_size=3, padding=1)

        self.bn_class_3_3 = nn.BatchNorm3d(num_classes * 5)
        self.conv_class_3_3 = nn.Conv3d(num_classes * 5, num_classes * 5, kernel_size=3, padding=1)

        self.bn_class = nn.BatchNorm3d(num_classes * 5)
        self.conv_class = nn.Conv3d(num_classes * 5, num_classes, kernel_size=1, padding=0)


        if self.training:
            self.bn_class_aux1 = nn.BatchNorm3d(num_classes)
            self.conv_class_aux1 = nn.Conv3d(num_classes, num_classes, kernel_size=1, padding=0)

            self.bn_class_aux2 = nn.BatchNorm3d(num_classes)
            self.conv_class_aux2 = nn.Conv3d(num_classes, num_classes, kernel_size=1, padding=0)

            self.bn_class_aux3 = nn.BatchNorm3d(num_classes)
            self.conv_class_aux3 = nn.Conv3d(num_classes, num_classes, kernel_size=1, padding=0)

            self.bn_class_aux4 = nn.BatchNorm3d(num_classes)
            self.conv_class_aux4 = nn.Conv3d(num_classes, num_classes, kernel_size=1, padding=0)

        # ----------------------------------------------------------
        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight)
                #nn.init.xavier_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        #print(self.training)

    def forward(self, x):
        first_three_features = self.features(x)
        #first_three_features_bn=self.features_bn(first_three_features)
        out = self.conv_pool_first(first_three_features)
        #out = self.scse_first(out)

        # Block 1
        out = self.dense_blocks[0](out)
        out = self.scse_blocks[0](out)
        up_block1 = self.upsampling_blocks[0](out)
        out = self.transit_blocks[0](out)

        # Block 2
        out = self.dense_blocks[1](out)
        out = self.scse_blocks[1](out)
        up_block2 = self.upsampling_blocks[1](out)
        out = self.transit_blocks[1](out)

        # Block 3
        out = self.dense_blocks[2](out)
        out = self.scse_blocks[2](out)
        up_block3 = self.upsampling_blocks[2](out)
        out = self.transit_blocks[2](out)

        # Block 4
        out = self.dense_blocks[3](out)
        out = self.scse_blocks[3](out)
        up_block4 = self.upsampling_blocks[3](out)

        first_three_features_3_3 = self.conv_class_3_3_first(F.relu(self.bn_class_3_3_first(first_three_features)))  # For more

        # Concatination
        out =  torch.cat([up_block1, up_block2, up_block3, up_block4, first_three_features_3_3], 1)

        out = self.conv_class_3_3(F.relu(self.conv_class_3_3(out))) # For more
        # ----------------------- classifier -----------------------
        out = self.conv_class(F.relu(self.bn_class(out)))
        # ----------------------------------------------------------

        if self.training:
            out_class_aux1 = self.conv_class_aux1(F.relu(self.bn_class_aux1(up_block1)))
            out_class_aux2 = self.conv_class_aux2(F.relu(self.bn_class_aux2(up_block2)))
            out_class_aux3 = self.conv_class_aux3(F.relu(self.bn_class_aux3(up_block3)))
            out_class_aux4 = self.conv_class_aux4(F.relu(self.bn_class_aux4(up_block4)))
            return out, out_class_aux1, out_class_aux2, out_class_aux3, out_class_aux4

        return out
