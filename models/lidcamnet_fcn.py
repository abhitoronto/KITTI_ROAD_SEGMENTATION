"""
    This module contains LidCamNet FCN model implementation from https://arxiv.org/pdf/1809.07941.pdf
"""

import torch
import torch.nn as nn
# import torch.nn.functional as F

def ConvBlock(in_ch, out_ch, ks=3, stride=1, padding=1, dilation=1, bn_enable=True):
    block = [nn.Conv2d(in_ch, out_ch, kernel_size=ks, stride=stride, padding=padding, dilation=dilation)]
    if bn_enable:
        block.append(nn.BatchNorm2d(out_ch))
    block.append(nn.ELU(inplace=True))
    return nn.Sequential(*block)


def DeconvBlock(in_ch, out_ch, ks=4, stride=2, bn_enable=True, conv_enable=True):
    block = [nn.ConvTranspose2d(in_ch, out_ch, kernel_size=ks, stride=stride, padding=1)]
    if bn_enable:
        block.append(nn.BatchNorm2d(out_ch))
    block.append(nn.ELU(inplace=True))
    if conv_enable:
        block.append(ConvBlock(out_ch, out_ch, bn_enable=bn_enable))
    return nn.Sequential(*block)


class LidCamNet(nn.Module):
    def __init__(self, num_classes=1, input_channels=32, bn_enable=False, init_type="He"):
        super(LidCamNet, self).__init__()

        self.num_classes = num_classes
        self.input_channels = input_channels
        self.bn_enable = bn_enable
        self.init_type = init_type

        self.encoder = nn.Sequential(
            ConvBlock(3, self.input_channels, ks=4, stride=2, bn_enable=self.bn_enable),                                #1/2
            ConvBlock(self.input_channels, self.input_channels, bn_enable=self.bn_enable),
            ConvBlock(self.input_channels, self.input_channels * 2, ks=4, stride=2, bn_enable=self.bn_enable),          #1/4
            ConvBlock(self.input_channels * 2, self.input_channels * 2, bn_enable=self.bn_enable),
            ConvBlock(self.input_channels * 2, self.input_channels * 4, ks=4, stride=2, bn_enable=self.bn_enable),      #1/8
        )

        self.center = nn.Sequential(
            ConvBlock(self.input_channels * 4, self.input_channels * 4, ks=3, bn_enable=self.bn_enable),                                    #6
            nn.Dropout2d(0.25),
            ConvBlock(self.input_channels * 4, self.input_channels * 4, ks=3, bn_enable=self.bn_enable),                                    #7
            nn.Dropout2d(0.25),
            ConvBlock(self.input_channels * 4, self.input_channels * 4, ks=3, dilation=(1,2), padding=(1,2), bn_enable=self.bn_enable),     #8
            nn.Dropout2d(0.25),
            ConvBlock(self.input_channels * 4, self.input_channels * 4, ks=3, dilation=(2,4), padding=(2,4), bn_enable=self.bn_enable),     #9
            nn.Dropout2d(0.25),
            ConvBlock(self.input_channels * 4, self.input_channels * 4, ks=3, dilation=(4,8), padding=(4,8), bn_enable=self.bn_enable),     #10
            nn.Dropout2d(0.25), 
            ConvBlock(self.input_channels * 4, self.input_channels * 4, ks=3, dilation=(8,16), padding=(8,16), bn_enable=self.bn_enable),   #11
            nn.Dropout2d(0.25),
            ConvBlock(self.input_channels * 4, self.input_channels * 4, ks=3, dilation=(16,32), padding=(16,32), bn_enable=self.bn_enable), #12
            nn.Dropout2d(0.25),
            ConvBlock(self.input_channels * 4, self.input_channels * 4, ks=3, bn_enable=self.bn_enable),                                    #13
            nn.Dropout2d(0.25),
            ConvBlock(self.input_channels * 4, self.input_channels * 4, ks=1, padding=0, bn_enable=self.bn_enable),                         #14
            nn.Dropout2d(0.25),
        )

        self.decoder = nn.Sequential(
            DeconvBlock(self.input_channels * 4, self.input_channels * 2, bn_enable=self.bn_enable),
            DeconvBlock(self.input_channels * 2, self.input_channels, bn_enable=self.bn_enable),
            DeconvBlock(self.input_channels, 8, bn_enable=self.bn_enable, conv_enable=False)
        )

        self.final = nn.Conv2d(8, self.num_classes, kernel_size=3, padding=1)

        #Initialization
        if self.init_type == "He":
            for m in self.modules():
                if isinstance(m, (nn.Conv2d)):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        m.bias.data.zero_() 
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
        elif self.init_type == "Xavier":
            raise NotImplementedError("This type of initialization is not implemented.")

    def forward(self, x):
        x = self.encoder(x)
        x = self.center(x)
        x = self.decoder(x)
        x = self.final(x)

        return x


class LidCamNetEarlyFusion(nn.Module):
    def __init__(self, num_classes=1, input_channels=32, bn_enable=False, init_type="He"):
        super(LidCamNetEarlyFusion, self).__init__()

        self.num_classes = num_classes
        self.input_channels = input_channels
        self.bn_enable = bn_enable
        self.init_type = init_type

        self.encoder = nn.Sequential(
            ConvBlock(3*2, self.input_channels, ks=4, stride=2, bn_enable=self.bn_enable),  # 1/2
            ConvBlock(self.input_channels, self.input_channels, bn_enable=self.bn_enable),
            ConvBlock(self.input_channels, self.input_channels * 2, ks=4, stride=2, bn_enable=self.bn_enable),  # 1/4
            ConvBlock(self.input_channels * 2, self.input_channels * 2, bn_enable=self.bn_enable),
            ConvBlock(self.input_channels * 2, self.input_channels * 4, ks=4, stride=2, bn_enable=self.bn_enable),
            # 1/8
        )

        self.center = nn.Sequential(
            ConvBlock(self.input_channels * 4, self.input_channels * 4, ks=3, bn_enable=self.bn_enable),  # 6
            nn.Dropout2d(0.25),
            ConvBlock(self.input_channels * 4, self.input_channels * 4, ks=3, bn_enable=self.bn_enable),  # 7
            nn.Dropout2d(0.25),
            ConvBlock(self.input_channels * 4, self.input_channels * 4, ks=3, dilation=(1, 2), padding=(1, 2),
                      bn_enable=self.bn_enable),  # 8
            nn.Dropout2d(0.25),
            ConvBlock(self.input_channels * 4, self.input_channels * 4, ks=3, dilation=(2, 4), padding=(2, 4),
                      bn_enable=self.bn_enable),  # 9
            nn.Dropout2d(0.25),
            ConvBlock(self.input_channels * 4, self.input_channels * 4, ks=3, dilation=(4, 8), padding=(4, 8),
                      bn_enable=self.bn_enable),  # 10
            nn.Dropout2d(0.25),
            ConvBlock(self.input_channels * 4, self.input_channels * 4, ks=3, dilation=(8, 16), padding=(8, 16),
                      bn_enable=self.bn_enable),  # 11
            nn.Dropout2d(0.25),
            ConvBlock(self.input_channels * 4, self.input_channels * 4, ks=3, dilation=(16, 32), padding=(16, 32),
                      bn_enable=self.bn_enable),  # 12
            nn.Dropout2d(0.25),
            ConvBlock(self.input_channels * 4, self.input_channels * 4, ks=3, bn_enable=self.bn_enable),  # 13
            nn.Dropout2d(0.25),
            ConvBlock(self.input_channels * 4, self.input_channels * 4, ks=1, padding=0, bn_enable=self.bn_enable),
            # 14
            nn.Dropout2d(0.25),
        )

        self.decoder = nn.Sequential(
            DeconvBlock(self.input_channels * 4, self.input_channels * 2, bn_enable=self.bn_enable),
            DeconvBlock(self.input_channels * 2, self.input_channels, bn_enable=self.bn_enable),
            DeconvBlock(self.input_channels, 8, bn_enable=self.bn_enable, conv_enable=False)
        )

        self.final = nn.Conv2d(8, self.num_classes, kernel_size=3, padding=1)

        # Initialization
        if self.init_type == "He":
            for m in self.modules():
                if isinstance(m, (nn.Conv2d)):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        m.bias.data.zero_()
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
        elif self.init_type == "Xavier":
            raise NotImplementedError("This type of initialization is not implemented.")

    def forward(self, x1, x2):
        # TODO: Change this back
        x = torch.cat((x1, x2), 1)

        x = self.encoder(x)
        x = self.center(x)
        x = self.decoder(x)
        x = self.final(x)

        return x


class LidCamNetLateFusion(nn.Module):
    def __init__(self, num_classes=1, input_channels=32, bn_enable=False, init_type="He"):
        super(LidCamNetLateFusion, self).__init__()

        self.num_classes = num_classes
        self.input_channels = input_channels
        self.bn_enable = bn_enable
        self.init_type = init_type

        self.encoder = nn.Sequential(
            ConvBlock(3, self.input_channels, ks=4, stride=2, bn_enable=self.bn_enable),  # 1/2
            ConvBlock(self.input_channels, self.input_channels, bn_enable=self.bn_enable),
            ConvBlock(self.input_channels, self.input_channels * 2, ks=4, stride=2, bn_enable=self.bn_enable),  # 1/4
            ConvBlock(self.input_channels * 2, self.input_channels * 2, bn_enable=self.bn_enable),
            ConvBlock(self.input_channels * 2, self.input_channels * 4, ks=4, stride=2, bn_enable=self.bn_enable),
            # 1/8
        )

        self.center = nn.Sequential(
            ConvBlock(self.input_channels * 4, self.input_channels * 4, ks=3, bn_enable=self.bn_enable),  # 6
            nn.Dropout2d(0.25),
            ConvBlock(self.input_channels * 4, self.input_channels * 4, ks=3, bn_enable=self.bn_enable),  # 7
            nn.Dropout2d(0.25),
            ConvBlock(self.input_channels * 4, self.input_channels * 4, ks=3, dilation=(1, 2), padding=(1, 2),
                      bn_enable=self.bn_enable),  # 8
            nn.Dropout2d(0.25),
            ConvBlock(self.input_channels * 4, self.input_channels * 4, ks=3, dilation=(2, 4), padding=(2, 4),
                      bn_enable=self.bn_enable),  # 9
            nn.Dropout2d(0.25),
            ConvBlock(self.input_channels * 4, self.input_channels * 4, ks=3, dilation=(4, 8), padding=(4, 8),
                      bn_enable=self.bn_enable),  # 10
            nn.Dropout2d(0.25),
            ConvBlock(self.input_channels * 4, self.input_channels * 4, ks=3, dilation=(8, 16), padding=(8, 16),
                      bn_enable=self.bn_enable),  # 11
            nn.Dropout2d(0.25),
            ConvBlock(self.input_channels * 4, self.input_channels * 4, ks=3, dilation=(16, 32), padding=(16, 32),
                      bn_enable=self.bn_enable),  # 12
            nn.Dropout2d(0.25),
            ConvBlock(self.input_channels * 4, self.input_channels * 4, ks=3, bn_enable=self.bn_enable),  # 13
            nn.Dropout2d(0.25),
            ConvBlock(self.input_channels * 4, self.input_channels * 4, ks=1, padding=0, bn_enable=self.bn_enable),
            # 14
            nn.Dropout2d(0.25),
        )

        self.decoder = nn.Sequential(
            DeconvBlock(self.input_channels * 4, self.input_channels * 2, bn_enable=self.bn_enable),
            DeconvBlock(self.input_channels * 2, self.input_channels, bn_enable=self.bn_enable),
            DeconvBlock(self.input_channels, 8, bn_enable=self.bn_enable, conv_enable=False)
        )

        self.final = nn.Conv2d(8*2, self.num_classes, kernel_size=3, padding=1)

        # Initialization
        if self.init_type == "He":
            for m in self.modules():
                if isinstance(m, (nn.Conv2d)):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        m.bias.data.zero_()
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
        elif self.init_type == "Xavier":
            raise NotImplementedError("This type of initialization is not implemented.")

    def forward(self, x1, x2):
        x1 = self.encoder(x1)
        x1 = self.center(x1)
        x1 = self.decoder(x1)

        x2 = self.encoder(x2)
        x2 = self.center(x2)
        x2 = self.decoder(x2)

        x = torch.cat((x1, x2), 1)
        x = self.final(x)

        return x