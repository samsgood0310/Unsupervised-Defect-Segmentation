"""Autoencoder with ssim loss

author: Haixin wang
e-mail: haixinwa@gmail.com
"""
import torch
import torch.nn as nn


class INConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1,
                 relu=True, ins_n=True, bias=False):
        super(INConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.ins_n = nn.InstanceNorm2d(out_planes, affine=True) if ins_n else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.ins_n is not None:
            x = self.ins_n(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class INDeConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, out_padding=0, dilation=1, groups=1,
                 relu=True, ins_n=True, bias=False):
        super(INDeConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.ConvTranspose2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                                       output_padding=out_padding, dilation=dilation, groups=groups, bias=bias)
        self.ins_n = nn.InstanceNorm2d(out_planes, affine=True) if ins_n else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.ins_n is not None:
            x = self.ins_n(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class Encoder(nn.Module):
    def __init__(self, img_channel):
        super(Encoder, self).__init__()
        self.conv1 = INConv(in_planes=img_channel, out_planes=32, kernel_size=4, stride=2, padding=1, relu=False)
        self.activation1 = nn.LeakyReLU(inplace=True, negative_slope=0.2)
        self.conv2 = INConv(in_planes=32, out_planes=32, kernel_size=4, stride=2, padding=1, relu=False)
        self.activation2 = nn.LeakyReLU(inplace=True, negative_slope=0.2)
        self.conv3 = INConv(in_planes=32, out_planes=32, kernel_size=3, stride=1, padding=1, relu=False)
        self.activation3 = nn.LeakyReLU(inplace=True, negative_slope=0.2)
        self.conv4 = INConv(in_planes=32, out_planes=64, kernel_size=4, stride=2, padding=1, relu=False)
        self.activation4 = nn.LeakyReLU(inplace=True, negative_slope=0.2)
        self.conv5 = INConv(in_planes=64, out_planes=64, kernel_size=3, stride=1, padding=1, relu=False)
        self.activation5 = nn.LeakyReLU(inplace=True, negative_slope=0.2)
        self.conv6 = INConv(in_planes=64, out_planes=128, kernel_size=4, stride=2, padding=1, relu=False)
        self.activation6 = nn.LeakyReLU(inplace=True, negative_slope=0.2)
        self.conv7 = INConv(in_planes=128, out_planes=64, kernel_size=3, stride=1, padding=1, relu=False)
        self.activation7 = nn.LeakyReLU(inplace=True, negative_slope=0.2)
        self.conv8 = INConv(in_planes=64, out_planes=32, kernel_size=3, stride=1, padding=1, relu=False)
        self.activation8 = nn.LeakyReLU(inplace=True, negative_slope=0.2)
        self.conv9 = INConv(in_planes=32, out_planes=2048, kernel_size=8, stride=1, padding=0,
                            ins_n=False, relu=False)
        self.activation9 = nn.LeakyReLU(inplace=True, negative_slope=0.2)

    def forward(self, x):
        x = self.activation1(self.conv1(x))
        x = self.activation2(self.conv2(x))
        x = self.activation3(self.conv3(x))
        x = self.activation4(self.conv4(x))
        x = self.activation5(self.conv5(x))
        x = self.activation6(self.conv6(x))
        x = self.activation7(self.conv7(x))
        x = self.activation8(self.conv8(x))
        x = self.activation9(self.conv9(x))

        return x


class Decoder(nn.Module):
    def __init__(self, img_channel):
        super(Decoder, self).__init__()
        self.deconv1 = INDeConv(in_planes=32, out_planes=64, kernel_size=3, stride=1, padding=1, relu=False)
        self.activation1 = nn.LeakyReLU(inplace=True, negative_slope=0.2)
        self.deconv2 = INDeConv(in_planes=64, out_planes=128, kernel_size=3, stride=1, padding=1, relu=False)
        self.activation2 = nn.LeakyReLU(inplace=True, negative_slope=0.2)
        self.deconv3 = INDeConv(in_planes=128, out_planes=64, kernel_size=4, stride=2, padding=1, relu=False)
        self.activation3 = nn.LeakyReLU(inplace=True, negative_slope=0.2)
        self.deconv4 = INDeConv(in_planes=64, out_planes=64, kernel_size=3, stride=1, padding=1, relu=False)
        self.activation4 = nn.LeakyReLU(inplace=True, negative_slope=0.2)
        self.deconv5 = INDeConv(in_planes=64, out_planes=32, kernel_size=4, stride=2, padding=1, relu=False)
        self.activation5 = nn.LeakyReLU(inplace=True, negative_slope=0.2)
        self.deconv6 = INDeConv(in_planes=32, out_planes=32, kernel_size=3, stride=1, padding=1, relu=False)
        self.activation6 = nn.LeakyReLU(inplace=True, negative_slope=0.2)
        self.deconv7 = INDeConv(in_planes=32, out_planes=32, kernel_size=4, stride=2, padding=1, relu=False)
        self.activation7 = nn.LeakyReLU(inplace=True, negative_slope=0.2)
        self.deconv8 = INDeConv(in_planes=32, out_planes=16, kernel_size=4, stride=2, padding=1, relu=False)
        self.activation8 = nn.LeakyReLU(inplace=True, negative_slope=0.2)
        self.output = INConv(in_planes=16, out_planes=img_channel, kernel_size=1, stride=1, padding=0,
                             ins_n=False, relu=False)
        self.activation9 = nn.Sigmoid()

    def forward(self, x):
        x = self.activation1(self.deconv1(x))
        x = self.activation2(self.deconv2(x))
        x = self.activation3(self.deconv3(x))
        x = self.activation4(self.deconv4(x))
        x = self.activation5(self.deconv5(x))
        x = self.activation6(self.deconv6(x))
        x = self.activation7(self.deconv7(x))
        x = self.activation8(self.deconv8(x))
        x = self.activation9(self.output(x))

        return x


class AE_basic(nn.Module):
    def __init__(self, img_channel):
        super(AE_basic, self).__init__()
        self.encoder = Encoder(img_channel)
        self.decoder = Decoder(img_channel)

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1, 8, 8)
        x = self.decoder(x)

        return x
