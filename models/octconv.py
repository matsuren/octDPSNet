from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

ALPHA = None  # default alpha


class OctConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1,
                 alpha_in=None, alpha_out=None, type='normal', bias=False):
        super(OctConv, self).__init__()

        if alpha_in is None and ALPHA is not None:
            alpha_in = ALPHA
        if alpha_out is None and ALPHA is not None:
            alpha_out = ALPHA

        self.kernel_size = kernel_size
        self.stride = stride
        self.type = type
        hf_ch_in = int(in_channels * (1 - alpha_in))
        hf_ch_out = int(out_channels * (1 - alpha_out))
        lf_ch_in = in_channels - hf_ch_in
        lf_ch_out = out_channels - hf_ch_out

        if type == 'first':
            if stride == 2:
                self.downsample = nn.AvgPool2d(kernel_size=2, stride=stride)
            self.convh = nn.Conv2d(
                in_channels, hf_ch_out,
                kernel_size=kernel_size, stride=1, padding=padding, dilation=dilation, bias=bias,
            )
            self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)
            self.convl = nn.Conv2d(
                in_channels, lf_ch_out,
                kernel_size=kernel_size, stride=1, padding=padding, dilation=dilation, bias=bias,
            )
        elif type == 'last':
            if stride == 2:
                self.downsample = nn.AvgPool2d(kernel_size=2, stride=stride)
            self.convh = nn.Conv2d(hf_ch_in, out_channels, kernel_size=kernel_size, padding=padding, dilation=dilation,
                                   bias=bias)
            self.convl = nn.Conv2d(lf_ch_in, out_channels, kernel_size=kernel_size, padding=padding, dilation=dilation,
                                   bias=bias)
            self.upsample = partial(F.interpolate, scale_factor=2, mode="nearest")
        else:
            if stride == 2:
                self.downsample = nn.AvgPool2d(kernel_size=2, stride=stride)

            self.L2L = nn.Conv2d(
                lf_ch_in, lf_ch_out,
                kernel_size=kernel_size, stride=1, padding=padding, dilation=dilation, bias=bias
            )
            self.L2H = nn.Conv2d(
                lf_ch_in, hf_ch_out,
                kernel_size=kernel_size, stride=1, padding=padding, dilation=dilation, bias=bias
            )
            self.H2L = nn.Conv2d(
                hf_ch_in, lf_ch_out,
                kernel_size=kernel_size, stride=1, padding=padding, dilation=dilation, bias=bias
            )
            self.H2H = nn.Conv2d(
                hf_ch_in, hf_ch_out,
                kernel_size=kernel_size, stride=1, padding=padding, dilation=dilation, bias=bias
            )
            self.upsample = partial(F.interpolate, scale_factor=2, mode="nearest")
            self.avg_pool = partial(F.avg_pool2d, kernel_size=2, stride=2)

    def forward(self, x):
        if self.type == 'first':
            if self.stride == 2:
                x = self.downsample(x)

            hf = self.convh(x)
            lf = self.avg_pool(x)
            lf = self.convl(lf)

            return hf, lf
        elif self.type == 'last':
            hf, lf = x
            if self.stride == 2:
                hf = self.downsample(hf)
                return self.convh(hf) + self.convl(lf)
            else:
                return self.convh(hf) + self.convl(self.upsample(lf))
        else:
            hf, lf = x
            if self.stride == 2:
                hf = self.downsample(hf)
                return self.H2H(hf) + self.L2H(lf), \
                       self.L2L(F.avg_pool2d(lf, kernel_size=2, stride=2)) + self.H2L(self.avg_pool(hf))
            else:
                return self.H2H(hf) + self.upsample(self.L2H(lf)), self.L2L(lf) + self.H2L(self.avg_pool(hf))


class OctConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=False,
                 alpha_in=None, alpha_out=None, type='normal'):
        super(OctConv3d, self).__init__()

        if alpha_in is None and ALPHA is not None:
            alpha_in = ALPHA
        if alpha_out is None and ALPHA is not None:
            alpha_out = ALPHA

        self.kernel_size = kernel_size
        self.stride = stride
        self.type = type
        hf_ch_in = int(in_channels * (1 - alpha_in))
        hf_ch_out = int(out_channels * (1 - alpha_out))
        lf_ch_in = in_channels - hf_ch_in
        lf_ch_out = out_channels - hf_ch_out

        if type == 'first':
            if stride == 2:
                self.downsample = nn.AvgPool3d(kernel_size=2, stride=stride)
            self.convh = nn.Conv3d(
                in_channels, hf_ch_out,
                kernel_size=kernel_size, stride=1, padding=padding, dilation=dilation, bias=bias,
            )
            self.avg_pool = nn.AvgPool3d(kernel_size=2, stride=2)
            self.convl = nn.Conv3d(
                in_channels, lf_ch_out,
                kernel_size=kernel_size, stride=1, padding=padding, dilation=dilation, bias=bias,
            )
        elif type == 'last':
            if stride == 2:
                self.downsample = nn.AvgPool3d(kernel_size=2, stride=stride)
            self.convh = nn.Conv3d(hf_ch_in, out_channels, kernel_size=kernel_size, padding=padding, dilation=dilation,
                                   bias=bias)
            self.convl = nn.Conv3d(lf_ch_in, out_channels, kernel_size=kernel_size, padding=padding, dilation=dilation,
                                   bias=bias)
            self.upsample = partial(F.interpolate, scale_factor=2, mode="nearest")
        else:
            if stride == 2:
                self.downsample = nn.AvgPool3d(kernel_size=2, stride=stride)

            self.L2L = nn.Conv3d(
                lf_ch_in, lf_ch_out,
                kernel_size=kernel_size, stride=1, padding=padding, dilation=dilation, bias=bias
            )
            self.L2H = nn.Conv3d(
                lf_ch_in, hf_ch_out,
                kernel_size=kernel_size, stride=1, padding=padding, dilation=dilation, bias=bias
            )
            self.H2L = nn.Conv3d(
                hf_ch_in, lf_ch_out,
                kernel_size=kernel_size, stride=1, padding=padding, dilation=dilation, bias=bias
            )
            self.H2H = nn.Conv3d(
                hf_ch_in, hf_ch_out,
                kernel_size=kernel_size, stride=1, padding=padding, dilation=dilation, bias=bias
            )
            self.upsample = partial(F.interpolate, scale_factor=2, mode="nearest")
            self.avg_pool = partial(F.avg_pool3d, kernel_size=2, stride=2)

    def forward(self, x):
        if self.type == 'first':
            if self.stride == 2:
                x = self.downsample(x)

            hf = self.convh(x)
            lf = self.avg_pool(x)
            lf = self.convl(lf)

            return hf, lf
        elif self.type == 'last':
            hf, lf = x
            if self.stride == 2:
                hf = self.downsample(hf)
                return self.convh(hf) + self.convl(lf)
            else:
                return self.convh(hf) + self.convl(self.upsample(lf))
        else:
            hf, lf = x
            if self.stride == 2:
                hf = self.downsample(hf)
                return self.H2H(hf) + self.L2H(lf), \
                       self.L2L(F.avg_pool3d(lf, kernel_size=2, stride=2)) + self.H2L(self.avg_pool(hf))
            else:
                return self.H2H(hf) + self.upsample(self.L2H(lf)), self.L2L(lf) + self.H2L(self.avg_pool(hf))


class disparityregression(nn.Module):
    def __init__(self, maxdisp, min_val=0):
        super(disparityregression, self).__init__()
        self.disp = torch.Tensor(np.reshape(np.array(range(min_val, maxdisp + min_val)), [1, maxdisp, 1, 1]))

    def forward(self, x):
        self.disp = self.disp.to(x.device)
        self.disp.requires_grad_(False)
        disp = self.disp.repeat(x.size()[0], 1, x.size()[2], x.size()[3])
        out = torch.sum(x * disp, 1)
        return out


def convbn(in_planes, out_planes, kernel_size, stride, pad, dilation):
    return nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                                   padding=dilation if dilation > 1 else pad, dilation=dilation, bias=False),
                         nn.BatchNorm2d(out_planes))


def convbn_3d(in_planes, out_planes, kernel_size, stride, pad):
    return nn.Sequential(
        nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, padding=pad, stride=stride, bias=False),
        nn.BatchNorm3d(out_planes))


def oct_convbn_3d(in_planes, out_planes, kernel_size, stride, pad):
    return nn.Sequential(OctConv3d(in_planes, out_planes, kernel_size=kernel_size, padding=pad, stride=stride),
                         _BatchNorm3d(out_planes))


def norm_conv3x3(in_planes, out_planes, stride=1, type=None):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def norm_conv1x1(in_planes, out_planes, stride=1, type=None):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def oct_conv3x3(in_planes, out_planes, stride=1, type='normal'):
    """3x3 convolution with padding"""
    return OctConv(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, type=type)


def oct_conv1x1(in_planes, out_planes, stride=1, type='normal'):
    """1x1 convolution"""
    return OctConv(in_planes, out_planes, kernel_size=1, stride=stride, type=type)


class _BatchNorm2d(nn.Module):
    def __init__(self, num_features, alpha_in=None, alpha_out=None, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super(_BatchNorm2d, self).__init__()

        if alpha_in is None and ALPHA is not None:
            alpha_in = ALPHA
        if alpha_out is None and ALPHA is not None:
            alpha_out = ALPHA

        hf_ch = int(num_features * (1 - alpha_in))
        lf_ch = num_features - hf_ch
        self.bnh = nn.BatchNorm2d(hf_ch)
        self.bnl = nn.BatchNorm2d(lf_ch)

    def forward(self, x):
        hf, lf = x
        return self.bnh(hf), self.bnl(lf)


class _BatchNorm3d(nn.Module):
    def __init__(self, num_features, alpha_in=None, alpha_out=None, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super(_BatchNorm3d, self).__init__()

        if alpha_in is None and ALPHA is not None:
            alpha_in = ALPHA
        if alpha_out is None and ALPHA is not None:
            alpha_out = ALPHA

        hf_ch = int(num_features * (1 - alpha_in))
        lf_ch = num_features - hf_ch
        self.bnh = nn.BatchNorm3d(hf_ch)
        self.bnl = nn.BatchNorm3d(lf_ch)

    def forward(self, x):
        hf, lf = x
        return self.bnh(hf), self.bnl(lf)


class _LeakyReLU(nn.LeakyReLU):
    def forward(self, x):
        hf, lf = x
        hf = super(_LeakyReLU, self).forward(hf)
        lf = super(_LeakyReLU, self).forward(lf)
        return hf, lf


class _ReLU(nn.ReLU):
    def forward(self, x):
        hf, lf = x
        hf = super(_ReLU, self).forward(hf)
        lf = super(_ReLU, self).forward(lf)
        return hf, lf


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 downsample=None,
                 type="normal",
                 oct_conv_on=True):
        super(BasicBlock, self).__init__()
        conv3x3 = oct_conv3x3 if oct_conv_on else norm_conv3x3
        norm_func = _BatchNorm2d if oct_conv_on else nn.BatchNorm2d
        act_func = _ReLU if oct_conv_on else nn.ReLU

        self.conv1 = conv3x3(
            inplanes, planes, type="first" if type == "first" else "normal")
        self.bn1 = norm_func(planes)
        self.relu1 = act_func(inplace=True)
        self.conv2 = conv3x3(
            planes,
            planes,
            stride,
            type="last" if type == "last" else "normal")
        if type == "last":
            norm_func = nn.BatchNorm2d
            act_func = nn.ReLU
        self.bn2 = norm_func(planes)
        self.relu2 = act_func(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        if isinstance(out, (tuple, list)):
            assert len(out) == len(identity) and len(out) == 2
            out = (out[0] + identity[0], out[1] + identity[1])
        else:
            out += identity

        out = self.relu2(out)

        return out


class oct_SPP(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 alpha_in=None,
                 alpha_out=None):
        super(oct_SPP, self).__init__()

        if alpha_in is None and ALPHA is not None:
            alpha_in = ALPHA
        if alpha_out is None and ALPHA is not None:
            alpha_out = ALPHA

        hf_ch_in = int(in_channels * (1 - alpha_in))
        hf_ch_out = int(out_channels * (1 - alpha_out))
        lf_ch_in = in_channels - hf_ch_in
        lf_ch_out = out_channels - hf_ch_out

        self.branch1_H = nn.Sequential(
            nn.AvgPool2d((16, 16), stride=(16, 16)),
            convbn(hf_ch_in, hf_ch_out, 1, 1, 0, 1), nn.ReLU(inplace=True))
        self.branch1_L = nn.Sequential(
            nn.AvgPool2d((16, 16), stride=(16, 16)),
            convbn(lf_ch_in, lf_ch_out, 1, 1, 0, 1), nn.ReLU(inplace=True))

        self.branch2_H = nn.Sequential(
            nn.AvgPool2d((8, 8), stride=(8, 8)),
            convbn(hf_ch_in, hf_ch_out, 1, 1, 0, 1), nn.ReLU(inplace=True))
        self.branch2_L = nn.Sequential(
            nn.AvgPool2d((8, 8), stride=(8, 8)),
            convbn(lf_ch_in, lf_ch_out, 1, 1, 0, 1), nn.ReLU(inplace=True))

        self.branch3_H = nn.Sequential(
            nn.AvgPool2d((4, 4), stride=(4, 4)),
            convbn(hf_ch_in, hf_ch_out, 1, 1, 0, 1), nn.ReLU(inplace=True))
        self.branch3_L = nn.Sequential(
            nn.AvgPool2d((4, 4), stride=(4, 4)),
            convbn(lf_ch_in, lf_ch_out, 1, 1, 0, 1), nn.ReLU(inplace=True))

        self.branch4_H = nn.Sequential(
            nn.AvgPool2d((2, 2), stride=(2, 2)),
            convbn(hf_ch_in, hf_ch_out, 1, 1, 0, 1), nn.ReLU(inplace=True))
        self.branch4_L = nn.Sequential(
            nn.AvgPool2d((2, 2), stride=(2, 2)),
            convbn(lf_ch_in, lf_ch_out, 1, 1, 0, 1), nn.ReLU(inplace=True))

    def forward(self, x):
        hf, lf = x
        hf_size = (hf.size()[2], hf.size()[3])
        lf_size = (lf.size()[2], lf.size()[3])
        hf_out1 = self.branch1_H(hf)

        hf_out1 = F.interpolate(
            hf_out1, hf_size, mode='bilinear', align_corners=False)
        hf_out2 = self.branch2_H(hf)
        hf_out2 = F.interpolate(
            hf_out2, hf_size, mode='bilinear', align_corners=False)
        hf_out3 = self.branch3_H(hf)
        hf_out3 = F.interpolate(
            hf_out3, hf_size, mode='bilinear', align_corners=False)
        hf_out4 = self.branch4_H(hf)
        hf_out4 = F.interpolate(
            hf_out4, hf_size, mode='bilinear', align_corners=False)

        lf_out1 = self.branch1_L(lf)
        lf_out1 = F.interpolate(
            lf_out1, lf_size, mode='bilinear', align_corners=False)
        lf_out2 = self.branch2_L(lf)
        lf_out2 = F.interpolate(
            lf_out2, lf_size, mode='bilinear', align_corners=False)
        lf_out3 = self.branch3_L(lf)
        lf_out3 = F.interpolate(
            lf_out3, lf_size, mode='bilinear', align_corners=False)
        lf_out4 = self.branch4_L(lf)
        lf_out4 = F.interpolate(
            lf_out4, lf_size, mode='bilinear', align_corners=False)

        hf_output = torch.cat((hf, hf_out1, hf_out2, hf_out3, hf_out4), 1)
        lf_output = torch.cat((lf, lf_out1, lf_out2, lf_out3, lf_out4), 1)

        return hf_output, lf_output


class oct_feature_extraction(nn.Module):
    def __init__(self, last_type='last'):
        super(oct_feature_extraction, self).__init__()
        self.inplanes = 32
        self.firstconv = nn.Sequential(
            convbn(3, 32, 3, 2, 1, 1), nn.ReLU(inplace=True),
            oct_conv3x3(32, 32, 1, type='first'), _BatchNorm2d(32),
            _ReLU(inplace=True), oct_conv3x3(32, 32, 1), _BatchNorm2d(32),
            _ReLU(inplace=True))

        self.layer1 = self._make_layer(BasicBlock, 32, 3, 1)
        self.layer2 = self._make_layer(BasicBlock, 64, 6, stride=2)  # orig16
        self.layer3 = self._make_layer(BasicBlock, 128, 3)
        self.layer4 = self._make_layer(BasicBlock, 128, 3)

        self.SPP = oct_SPP(128, 32)

        self.lastconv = nn.Sequential(
            oct_conv3x3(320, 128, 1), _BatchNorm2d(128), _ReLU(inplace=True),
            oct_conv1x1(128, 32, 1, type=last_type))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, type="normal"):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion or type == 'first':
            norm_func = nn.BatchNorm2d if type == "last" else _BatchNorm2d
            downsample = nn.Sequential(
                oct_conv1x1(
                    self.inplanes, planes * block.expansion, stride,
                    type=type),
                norm_func(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(self.inplanes, planes, stride, downsample, type=type))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(self.inplanes, planes, oct_conv_on=type != "last"))

        return nn.Sequential(*layers)

    def forward(self, x):
        output = self.firstconv(x)
        output = self.layer1(output)
        output_raw = self.layer2(output)
        output = self.layer3(output_raw)
        output_skip = self.layer4(output)

        output_spp = self.SPP(output_skip)

        output_feature_H = torch.cat(
            (output_raw[0], output_spp[0]), 1)
        output_feature_L = torch.cat(
            (output_raw[1], output_spp[1]), 1)

        output_feature = self.lastconv((output_feature_H, output_feature_L))

        return output_feature
