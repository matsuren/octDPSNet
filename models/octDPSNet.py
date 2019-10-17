from __future__ import print_function
import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
import math
from models.submodule import *
from models.octconv import OctConv, OctConv3d, oct_feature_extraction, oct_convbn_3d
from models.octconv import disparityregression
from models.octconv import _LeakyReLU, _ReLU
from models import octconv
from inverse_warp import inverse_warp
from functools import partial
from torch.utils import model_zoo

from models.oct_semodule import SpatialSELayerOct


def addHL(x1, x2):
    x1_H, x1_L = x1
    x2_H, x2_L = x2
    return x1_H+x2_H, x1_L+x2_L


def oct_convtext(in_planes, out_planes, kernel_size=3, stride=1, dilation=1, type='normal'):
    if type == 'last':
        return nn.Sequential(
            OctConv(in_planes, out_planes, kernel_size=kernel_size, stride=stride, dilation=dilation, padding=((kernel_size - 1) * dilation) // 2, type=type),
            nn.LeakyReLU(0.1,inplace=True)
        )
    else:
        return nn.Sequential(
            OctConv(in_planes, out_planes, kernel_size=kernel_size, stride=stride, dilation=dilation, padding=((kernel_size - 1) * dilation) // 2, type=type),
            _LeakyReLU(0.1,inplace=True)
        )


class costRegularization(nn.Module):
    def __init__(self):
        super(costRegularization, self).__init__()
        self.dres0 = nn.Sequential(oct_convbn_3d(64, 32, 3, 1, 1),
                                   _ReLU(inplace=True),
                                   oct_convbn_3d(32, 32, 3, 1, 1),
                                   _ReLU(inplace=True))

        self.dres1 = nn.Sequential(oct_convbn_3d(32, 32, 3, 1, 1),
                                   _ReLU(inplace=True),
                                   oct_convbn_3d(32, 32, 3, 1, 1))

        self.dres2 = nn.Sequential(oct_convbn_3d(32, 32, 3, 1, 1),
                                   _ReLU(inplace=True),
                                   oct_convbn_3d(32, 32, 3, 1, 1))

        self.dres3 = nn.Sequential(oct_convbn_3d(32, 32, 3, 1, 1),
                                   _ReLU(inplace=True),
                                   oct_convbn_3d(32, 32, 3, 1, 1))

        self.dres4 = nn.Sequential(oct_convbn_3d(32, 32, 3, 1, 1),
                                   _ReLU(inplace=True),
                                   oct_convbn_3d(32, 32, 3, 1, 1))

        # output cost 1ch, cost_L 1ch
        self.classify = nn.Sequential(oct_convbn_3d(32, 32, 3, 1, 1),
                                      _ReLU(inplace=True),
                                      OctConv3d(32, 2, kernel_size=3, padding=1, stride=1, alpha_out=0.5))

    def forward(self, cost, cost_L):
        cost = cost.contiguous()
        cost_L = cost_L.contiguous()
        cost0 = self.dres0((cost, cost_L))
        cost0 = addHL(self.dres1(cost0), cost0)
        cost0 = addHL(self.dres2(cost0), cost0)
        cost0 = addHL(self.dres3(cost0), cost0)
        cost0 = addHL(self.dres4(cost0), cost0)
        cost0 = self.classify(cost0)

        return cost0


class octDPSNet(nn.Module):
    def __init__(self, nlabel, mindepth):
        super(octDPSNet, self).__init__()
        print('octDPSNet')
        print('octconv alpha:', octconv.ALPHA)
        self.device = None
        self.nlabel = nlabel
        self.mindepth = mindepth

#         self.feature_extraction = feature_extraction()
        self.feature_extraction = oct_feature_extraction(last_type='normal')
        self.cost_regularization = costRegularization()

        self.semodule = SpatialSELayerOct(self.nlabel)

        self.upsample = partial(F.interpolate, scale_factor=2, mode="trilinear", align_corners=False)
        # TODO Leakyrelu? Numver of parameter?
        self.convs_L = nn.Sequential(
            OctConv(1+int(32*octconv.ALPHA), 32, kernel_size=3, stride=1, dilation=1,padding=1, type='first'),
            _ReLU(inplace=True),
            OctConv(32, 32, kernel_size=3, stride=1, dilation=2,padding=2),
            _ReLU(inplace=True),
            OctConv(32, 32, kernel_size=3, stride=1, dilation=4, padding=4),
            _ReLU(inplace=True),
            OctConv(32, 16, kernel_size=3, stride=1, dilation=1, padding=1),
            _ReLU(inplace=True),
            OctConv(16, 1, kernel_size=3, stride=1, dilation=1, padding=1, type='last'),
            nn.ReLU(inplace=True)
        )

        self.convs_H_first = OctConv(32, 32, kernel_size=3, stride=1, padding=1, type='last')
        self.convs_H = nn.Sequential(
            OctConv(1+32, 32, kernel_size=3, stride=1, padding=1, type='first'),
            _ReLU(inplace=True),
            OctConv(32, 32, kernel_size=3, stride=1, dilation=2, padding=2),
            _ReLU(inplace=True),
            OctConv(32, 32, kernel_size=3, stride=1, dilation=4, padding=4),
            _ReLU(inplace=True),
            # OctConv(32, 32, kernel_size=3, stride=1, dilation=8, padding=8),
            # _ReLU(inplace=True),
            OctConv(32, 16, kernel_size=3, stride=1, dilation=1, padding=1),
            _ReLU(inplace=True),
            OctConv(16, 1, kernel_size=3, stride=1, padding=1, type='last'),
            nn.ReLU(inplace=True)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1]*m.kernel_size[2] * m.out_channels
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

        intrinsics4 = intrinsics.clone()
        intrinsics_inv4 = intrinsics_inv.clone()
        intrinsics4[:,:2,:] = intrinsics4[:,:2,:] / 4
        intrinsics_inv4[:,:2,:2] = intrinsics_inv4[:,:2,:2] * 4

        intrinsics8 = intrinsics.clone()
        intrinsics_inv8 = intrinsics_inv.clone()
        intrinsics8[:,:2,:] = intrinsics8[:,:2,:] / 8
        intrinsics_inv8[:,:2,:2] = intrinsics_inv8[:,:2,:2] * 8

        refimg_fea_H, refimg_fea_L = self.feature_extraction(ref)

        disp2depth = torch.ones(refimg_fea_H.size(0), refimg_fea_H.size(2), refimg_fea_H.size(3)).to(self.device) * self.mindepth * self.nlabel
        disp2depth_L = torch.ones(refimg_fea_L.size(0), refimg_fea_L.size(2), refimg_fea_L.size(3)).to(self.device) * self.mindepth * self.nlabel

        for j, target in enumerate(targets):
            cost = torch.FloatTensor(refimg_fea_H.size()[0], refimg_fea_H.size()[1]*2, self.nlabel,  refimg_fea_H.size()[2],  refimg_fea_H.size()[3]).zero_().to(self.device)
            targetimg_fea_H, targetimg_fea_L  = self.feature_extraction(target)

            for i in range(1, self.nlabel+1):
                depth = torch.div(disp2depth, i)
                targetimg_fea_t = inverse_warp(targetimg_fea_H, depth, pose[:,j], intrinsics4, intrinsics_inv4)
                cost[:, :refimg_fea_H.size()[1], i-1, :,:] = refimg_fea_H
                cost[:, refimg_fea_H.size()[1]:, i-1, :,:] = targetimg_fea_t

            cost_L = torch.FloatTensor(refimg_fea_L.size()[0], refimg_fea_L.size()[1]*2, self.nlabel//2,  refimg_fea_L.size()[2],  refimg_fea_L.size()[3]).zero_().to(self.device)
            for i in range(self.nlabel // 2):
                float_index = 1 + 0.5 + 2 * i
                #     print(float_index)
                depth = torch.div(disp2depth_L, float_index)
                targetimg_fea_t = inverse_warp(targetimg_fea_L, depth, pose[:,j], intrinsics8, intrinsics_inv8)
                cost_L[:, :refimg_fea_L.size()[1], i-1, :,:] = refimg_fea_L
                cost_L[:, refimg_fea_L.size()[1]:, i-1, :,:] = targetimg_fea_t

            cost0 = self.cost_regularization(cost, cost_L)

            if j == 0:
                costs = cost0
            else:
                costs = addHL(costs, cost0)

        costs, costs_L = costs[0]/len(targets), costs[1]/len(targets)

        # depth refinement
        costss_L = torch.FloatTensor(refimg_fea_L.size()[0], 1, self.nlabel//2, refimg_fea_L.size()[2],
                                     refimg_fea_L.size()[3]).zero_().to(self.device)
        for i in range(self.nlabel//2):
            costt_L = costs_L[:, :, i, :, :]
            costss_L[:, :, i, :, :] = self.convs_L(torch.cat([refimg_fea_L, costt_L],1)) + costt_L

        # TODO
        # combine low-res high-res volume
        costss_H = torch.FloatTensor(refimg_fea_H.size()[0], 1, self.nlabel, refimg_fea_H.size()[2],
                                     refimg_fea_H.size()[3]).zero_().to(self.device)
        refimg_fea = self.convs_H_first((refimg_fea_H, refimg_fea_L))

        # Squeeze and Excitation
        costss_L = self.upsample(costss_L)
        costs_se_H, costs_se_L = self.semodule(costs, costss_L)

        for i in range(self.nlabel):
            costt_H = costs_se_H[:, :, i, :, :]
            costt_L = costs_se_L[:, :, i, :, :]
            # # part1
            # cost_tmp = self.convs_H(torch.cat([refimg_fea, costt_H + costt_L],1))
            # costss_H[:, :, i, :, :] = cost_tmp + costt_L
            # part2
            costss_H[:, :, i, :, :] = self.convs_H(torch.cat([refimg_fea, costt_H + costt_L], 1)) + costt_L + costt_H
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
            costs_se_up = F.interpolate((costs_se_H + costs_se_L), vol_size, mode='trilinear', align_corners=False)
            pred_se = F.softmax(torch.squeeze(costs_se_up, 1), dim=1)
            pred_se = disparityregression(self.nlabel, 1)(pred_se).unsqueeze(1)
            depth_se = self.mindepth*self.nlabel / pred_se

            #######################################
            # Loss is not calculated from depth L
            # # therefore inverse depth upsampling is used here to save memory
            pred_L = F.softmax(torch.squeeze(costss_L,1), dim=1)
            pred_L = disparityregression(self.nlabel, 1)(pred_L).unsqueeze(1)
            pred_L = F.interpolate(pred_L, [ref.size()[2], ref.size()[3]], mode='bilinear', align_corners=False)
            depth_L = self.mindepth*self.nlabel / pred_L

            # costss_Lup = F.interpolate(costss_L, vol_size, mode='trilinear', align_corners=False)
            # pred_L = F.softmax(torch.squeeze(costss_Lup,1), dim=1)
            # pred_L = disparityregression(self.nlabel, 1)(pred_L).unsqueeze(1)
            # depth_L = self.mindepth*self.nlabel / pred_L
            return depth_L, depth0, depth_se
        else:
            return depth0


def octdpsnet(nlabel, mindepth, alpha, pretrained=False):
    a_to_url = {
        0.25: 'octdps_a25n64-a6d5f6e8.pth',
        0.50: 'octdps_a50n64-e6b0d50e.pth',
        0.75: 'octdps_a75n64-2c47a5f9.pth',
        0.875: 'octdps_a875n64-de7a76ad.pth',
        0.9375: 'octdps_a9375n64-398ec6ee.pth'}
    model = octDPSNet(nlabel, mindepth)
    if pretrained:
        if alpha in a_to_url:
            url = 'http://www.robot.t.u-tokyo.ac.jp/~komatsu/data/{}'.format(a_to_url[alpha])
            print(a_to_url[alpha])
            weights = model_zoo.load_url(url, map_location='cpu')
            model.load_state_dict(weights['state_dict'])
    return model
