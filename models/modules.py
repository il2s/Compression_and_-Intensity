import torch
import torch.nn as nn
import torch.nn.functional as F
class Spatial(nn.Module):
    def __init__(self, channel):
        super(Spatial, self).__init__()
        self.spatial = nn.Sequential(nn.Conv2d(channel, channel, kernel_size=3, padding=1),
                                     nn.ReLU(inplace=True),
                                     nn.Conv2d(channel, channel, kernel_size=3, padding=1),
                                     nn.ReLU(inplace=True),
                                     nn.AdaptiveAvgPool3d((1, 240, 240)),
        )


    def forward(self, x):
        x = self.spatial(x)
        return x

class oneSpatial(nn.Module):
    def __init__(self, channel, out_channel):
        super(oneSpatial, self).__init__()
        self.spatial = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, out_channel, kernel_size=1, padding=0),
            nn.Sigmoid()
        )
    def forward(self, x):
        x= self.spatial(x)
        return x



class CSEBlock(nn.Module):
    def __init__(self, channel):
        super(CSEBlock, self).__init__()
        self.reduction = 16
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.channel_excitation = nn.Sequential(nn.Conv2d(channel, channel//self.reduction, kernel_size=1, padding=0),
                                                nn.ReLU(inplace=True),
                                                nn.Conv2d(channel // self.reduction, channel, kernel_size=1, padding=0),
                                                nn.Sigmoid())



    def forward(self, x):
        chn_se = self.avg_pool(x)
        chn_se = self.channel_excitation(chn_se)
        chn_se = torch.mul(x, chn_se)

        return chn_se

class SCSEBlock(nn.Module):
    def __init__(self, channel):
        super(SCSEBlock, self).__init__()
        self.reduction = 16
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.channel_excitation = nn.Sequential(nn.Conv2d(channel, channel//self.reduction, kernel_size=1, padding=0),
                                                nn.ReLU(inplace=True),
                                                nn.Conv2d(channel // self.reduction, channel, kernel_size=1, padding=0),
                                                nn.Sigmoid())

        self.spatial_se = nn.Sequential(nn.Conv2d(channel, 1, kernel_size=1, stride=1, padding=0),
                                        nn.Sigmoid())

    def forward(self, x):
        chn_se = self.avg_pool(x)
        chn_se = self.channel_excitation(chn_se)
        chn_se = torch.mul(x, chn_se)

        spa_se = self.spatial_se(x)
        spa_se = torch.mul(x, spa_se)

        out_se = chn_se + spa_se
        return out_se


class CBAMBlock(nn.Module):
    def __init__(self, channel):
        super(CBAMBlock, self).__init__()
        self.reduction = 16

        self.channel_excitation = nn.Sequential(nn.Conv2d(channel, channel // self.reduction, kernel_size=1, padding=0),
                                                nn.ReLU(inplace=True),
                                                nn.Conv2d(channel // self.reduction, channel, kernel_size=1, padding=0),
                                                )

        self.spatial_excitation = nn.Sequential(nn.Conv2d(2, 1, kernel_size=7, padding=3),
                                                nn.Sigmoid())

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_c = nn.AdaptiveAvgPool2d(1)(x)
        avg_c = self.channel_excitation(avg_c)

        max_c = nn.AdaptiveMaxPool2d(1)(x)
        max_c = self.channel_excitation(max_c)

        out_c = avg_c + max_c
        out_c = self.sigmoid(out_c)

        out_c = torch.mul(x, out_c)

        avg_s = nn.AdaptiveAvgPool3d((1, len(x[0][0][0]), len(x[0][0][0])))(out_c)
        max_s = nn.AdaptiveMaxPool3d((1, len(x[0][0][0]), len(x[0][0][0])))(out_c)

        out_s = torch.cat([avg_s, max_s], 1)
        out_s = self.spatial_excitation(out_s)

        out = torch.mul(out_c, out_s )
        out = x + out
        return out