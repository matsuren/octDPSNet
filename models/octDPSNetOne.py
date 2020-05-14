from __future__ import print_function
import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
import math
from models.submodule import *
from models.octconv import disparityregression
# from models.octconv import _LeakyReLU, _ReLU
from models import octconv
from inverse_warp import inverse_warp, get_homography, homography_transform
from functools import partial
from torch.utils import model_zoo

from models.oct_semodule import SpatialSELayerOct

def norm_conv3x3(in_planes, out_planes, stride=1, type=None):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 downsample=None):
        super(BasicBlock, self).__init__()
        conv3x3 = norm_conv3x3
        norm_func = nn.BatchNorm2d
        act_func = nn.ReLU

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


class feature_extraction(nn.Module):
    def __init__(self, last_type='last'):
        super(feature_extraction, self).__init__()
        self.inplanes = 32
        self.firstconv = nn.Sequential(
            convbn(3, 32, 3, 2, 1, 1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, 1, 1),  nn.AvgPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            convbn(32, 32, 3, 1, 1, 1), nn.ReLU(inplace=True))

        self.layer1 = self._make_layer(BasicBlock, 32, 3, 1)
        self.layer2 = self._make_layer(BasicBlock, 64, 6, stride=2)  # orig16
        self.layer3 = self._make_layer(BasicBlock, 128, 3)
        self.layer4 = self._make_layer(BasicBlock, 128, 3)

        self.SPP = SPP(128, 32)

        self.lastconv = nn.Sequential(
            convbn(320, 128, 3, 1, 1, 1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 32, 1))

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
            norm_func = nn.BatchNorm2d
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, 1, stride),
                norm_func(planes * block.expansion)
            )

        layers = []
        layers.append(
            block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        output = self.firstconv(x)
        output = self.layer1(output)
        layer1_output = output
        output_raw = self.layer2(output)
        output = self.layer3(output_raw)
        output_skip = self.layer4(output)

        output_spp = self.SPP(output_skip)

        output_feature_L = torch.cat(
            (output_raw, output_spp), 1)

        output_feature = self.lastconv(output_feature_L)

        return output_feature, layer1_output




class SPP(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels):
        super(SPP, self).__init__()

        lf_ch_in = in_channels
        lf_ch_out = out_channels

        self.branch1_L = nn.Sequential(
            nn.AvgPool2d((16, 16), stride=(16, 16)),
            convbn(lf_ch_in, lf_ch_out, 1, 1, 0, 1), nn.ReLU(inplace=True))

        self.branch2_L = nn.Sequential(
            nn.AvgPool2d((8, 8), stride=(8, 8)),
            convbn(lf_ch_in, lf_ch_out, 1, 1, 0, 1), nn.ReLU(inplace=True))

        self.branch3_L = nn.Sequential(
            nn.AvgPool2d((4, 4), stride=(4, 4)),
            convbn(lf_ch_in, lf_ch_out, 1, 1, 0, 1), nn.ReLU(inplace=True))

        self.branch4_L = nn.Sequential(
            nn.AvgPool2d((2, 2), stride=(2, 2)),
            convbn(lf_ch_in, lf_ch_out, 1, 1, 0, 1), nn.ReLU(inplace=True))

    def forward(self, x):
        lf = x
        lf_size = (lf.size()[2], lf.size()[3])
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

        lf_output = torch.cat((lf, lf_out1, lf_out2, lf_out3, lf_out4), 1)

        return lf_output


def addHL(x1, x2):
    return x1 + x2



def convbn_3d(in_planes, out_planes, kernel_size, stride, pad):
    return nn.Sequential(nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, padding=pad, stride=stride),
                         nn.BatchNorm3d(out_planes))

class costRegularization(nn.Module):
    def __init__(self):
        super(costRegularization, self).__init__()
        self.dres0 = nn.Sequential(convbn_3d(64, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(32, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True))

        self.dres1 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(32, 32, 3, 1, 1))

        self.dres2 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(32, 32, 3, 1, 1))

        self.dres3 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(32, 32, 3, 1, 1))

        self.dres4 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(32, 32, 3, 1, 1))

        # output cost 1ch, cost_L 1ch
        self.classify = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1))

    def forward(self, cost):
        cost = cost.contiguous()
        cost0 = self.dres0(cost)
        cost0 = addHL(self.dres1(cost0), cost0)
        cost0 = addHL(self.dres2(cost0), cost0)
        cost0 = addHL(self.dres3(cost0), cost0)
        cost0 = addHL(self.dres4(cost0), cost0)
        cost0 = self.classify(cost0)
        return cost0



class Conv_L(nn.Module):
    def __init__(self):
        super(Conv_L, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1 + 32, 32, kernel_size=3, stride=1, dilation=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, dilation=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, dilation=4, padding=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, kernel_size=3, stride=1, dilation=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=3, stride=1, dilation=1, padding=1)
        )
        self.upsample = partial(F.interpolate, scale_factor=2, mode="nearest")
        self.avg_pool = partial(F.avg_pool2d, kernel_size=2, stride=2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.avg_pool(x)
        x = self.conv(x)
        x = self.upsample(x)
        x = self.relu(x)
        return x

class Conv_H(nn.Module):
    def __init__(self):
        super(Conv_H, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1 + 32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, dilation=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, dilation=4, padding=4),
            nn.ReLU(inplace=True),
            # OctConv(32, 32, kernel_size=3, stride=1, dilation=8, padding=8),
            # nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, kernel_size=3, stride=1, dilation=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1)
        )
        self.upsample = partial(F.interpolate, scale_factor=2, mode="nearest")
        self.avg_pool = partial(F.avg_pool2d, kernel_size=2, stride=2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.avg_pool(x)
        x = self.conv(x)
        x = self.upsample(x)
        x = self.relu(x)
        return x


class octDPSNetOne(nn.Module):
    def __init__(self, nlabel, mindepth):
        super(octDPSNetOne, self).__init__()
        print('octDPSNetOne')
        print('octconv alpha:', octconv.ALPHA)
        assert octconv.ALPHA == 1
        self.device = None
        self.nlabel = nlabel
        self.mindepth = mindepth
        self.grid_sample = partial(torch.nn.functional.grid_sample, padding_mode='zeros', align_corners=False)
        #         self.feature_extraction = feature_extraction()
        self.feature_extraction = feature_extraction()
        self.cost_regularization = costRegularization()

        self.semodule = SpatialSELayerOct(self.nlabel)

        self.upsample = partial(F.interpolate, scale_factor=2, mode="trilinear", align_corners=False)
        # TODO Leakyrelu? Numver of parameter?
        self.convs_L = Conv_L()

        # self.convs_H_first = OctConv(32, 32, kernel_size=3, stride=1, padding=1, type='last')
        self.convs_H = Conv_H()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, ref, targets, pose, intrinsics, intrinsics_inv):
        if self.device is None:
            self.device = ref.device

        intrinsics8 = intrinsics.clone()
        intrinsics_inv8 = intrinsics_inv.clone()
        intrinsics8[:, :2, :] = intrinsics8[:, :2, :] / 8
        intrinsics_inv8[:, :2, :2] = intrinsics_inv8[:, :2, :2] * 8

        refimg_fea_L, layer_1_output = self.feature_extraction(ref)

        B, C_H, H, W = refimg_fea_L.size()
        H = H*2
        W = W*2
        C_L = refimg_fea_L.size(1)

        # pixel coordinate
        i_range = torch.arange(0, H).view(1, H, 1).expand(1, H, W).type_as(refimg_fea_L)  # [1, H, W]
        j_range = torch.arange(0, W).view(1, 1, W).expand(1, H, W).type_as(refimg_fea_L)  # [1, H, W]
        ones = torch.ones(1, H, W).type_as(refimg_fea_L)
        pixel_coords = torch.stack((j_range, i_range, ones), dim=1)  # [1, 3, H, W]
        pixel_coords_L = pixel_coords[:, :, :H//2, :W//2].expand(B, 3, H//2, W//2).contiguous().view(B, 3, -1)
        # disp2depth = torch.full((B, H, W), self.mindepth * self.nlabel, device=self.device)
        # disp2depth_L = torch.full((B, H//2, W//2), self.mindepth * self.nlabel, device=self.device)

        for j, target in enumerate(targets):
            targetimg_fea_L, _ = self.feature_extraction(target)
            cost_L = torch.zeros((B,  C_L * 2, self.nlabel // 2, H//2, W//2), device=self.device)
            for i in range(self.nlabel // 2):
                float_index = 1 + 0.5 + 2 * i
                #     print(float_index)
                # depth = torch.div(disp2depth_L, float_index)
                depth = self.mindepth*self.nlabel/float_index

                # targetimg_fea_t = inverse_warp(targetimg_fea_L, depth, pose[:, j], intrinsics8, intrinsics_inv8)
                current_pixel_coords = pixel_coords_L.to(self.device)
                homography = get_homography(depth, pose[:, j], intrinsics8, intrinsics_inv8)
                # transform
                pixel_coords = homography_transform(homography, current_pixel_coords, B, H//2, W//2)
                targetimg_fea_t = self.grid_sample(targetimg_fea_L, pixel_coords)

                cost_L[:, :C_L, i - 1, :, :] = refimg_fea_L
                cost_L[:, C_L:, i - 1, :, :] = targetimg_fea_t

            cost0 = self.cost_regularization(cost_L)
            if j == 0:
                costs = cost0
            else:
                costs = addHL(costs, cost0)

        costs_L = costs/ len(targets)

        # depth refinement
        costss_L = torch.zeros((B, 1, self.nlabel // 2, H // 2, W // 2), device=self.device)
        for i in range(self.nlabel // 2):
            costt_L = costs_L[:, :, i, :, :]
            costss_L[:, :, i, :, :] = self.convs_L(torch.cat([refimg_fea_L, costt_L], 1)) + costt_L

        # TODO
        # combine low-res high-res volume
        costss_H = torch.zeros((B, 1, self.nlabel, H, W), device=self.device)
        # refimg_fea = self.convs_H_first((refimg_fea_H, refimg_fea_L))

        # Squeeze and Excitation
        costss_L = self.upsample(costss_L)
        # costs_se_H, costs_se_L = self.semodule(costs, costss_L)

        for i in range(self.nlabel):
            costt_L = costss_L[:, :, i, :, :]
            # # part1
            # cost_tmp = self.convs_H(torch.cat([refimg_fea, costt_H + costt_L],1))
            # costss_H[:, :, i, :, :] = cost_tmp + costt_L
            # part2
            costss_H[:, :, i, :, :] = self.convs_H(torch.cat([layer_1_output, costt_L], 1)) + costt_L
            # # part3
            # costss_H[:, :, i, :, :] = self.convs_H(torch.cat([refimg_fea, costt_H, costt_L], 1))
            # # part4
            # costss_H[:, :, i, :, :] = self.convs_H(torch.cat([refimg_fea, costt_H, costt_L],1)) + costt_H

        vol_size = [self.nlabel, ref.size()[2], ref.size()[3]]
        costss_Hup = F.interpolate(costss_H, vol_size, mode='trilinear', align_corners=False)
        pred0 = F.softmax(torch.squeeze(costss_Hup, 1), dim=1)
        pred0 = disparityregression(self.nlabel, 1)(pred0).unsqueeze(1)
        depth0 = self.mindepth * self.nlabel / pred0

        if self.training:
            #####################################
            # Visualize effect of SE module
            costs_se_up = F.interpolate((costss_L), vol_size, mode='trilinear', align_corners=False)
            pred_se = F.softmax(torch.squeeze(costs_se_up, 1), dim=1)
            pred_se = disparityregression(self.nlabel, 1)(pred_se).unsqueeze(1)
            depth_se = self.mindepth * self.nlabel / pred_se

            #######################################
            # Loss is not calculated from depth L
            # # therefore inverse depth upsampling is used here to save memory
            pred_L = F.softmax(torch.squeeze(costss_L, 1), dim=1)
            pred_L = disparityregression(self.nlabel, 1)(pred_L).unsqueeze(1)
            pred_L = F.interpolate(pred_L, [ref.size()[2], ref.size()[3]], mode='bilinear', align_corners=False)
            depth_L = self.mindepth * self.nlabel / pred_L

            # costss_Lup = F.interpolate(costss_L, vol_size, mode='trilinear', align_corners=False)
            # pred_L = F.softmax(torch.squeeze(costss_Lup,1), dim=1)
            # pred_L = disparityregression(self.nlabel, 1)(pred_L).unsqueeze(1)
            # depth_L = self.mindepth*self.nlabel / pred_L
            return depth_L, depth0, depth_se
        else:
            return depth0


