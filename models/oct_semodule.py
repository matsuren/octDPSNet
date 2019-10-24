import torch
import torch.nn as nn


class SpatialSELayerOct(nn.Module):

    def __init__(self, num_channels, reduction_ratio=8):
        """

        :param num_channels: No of input channels
        :param reduction_ratio: Reduction ratio
        """
        super(SpatialSELayerOct, self).__init__()
        reduction_ch = num_channels // reduction_ratio
        self.conv_H = nn.Sequential(
            nn.Conv2d(num_channels, reduction_ch, 1),
            nn.ReLU(inplace=True))
        self.conv_L = nn.Sequential(
            nn.Conv2d(num_channels, reduction_ch, 1),
            nn.ReLU(inplace=True))

        self.conv_HL = nn.Conv2d(2 * reduction_ch, 1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_H, input_L):
        assert input_H.size() == input_L.size(), 'input_H and input_L should have the same size'

        # spatial squeeze
        batch_size, _, _, width, height = input_H.size()

        out_H = self.conv_H(torch.squeeze(input_H, dim=1))
        out_L = self.conv_L(torch.squeeze(input_L, dim=1))

        out = self.conv_HL(torch.cat((out_H, out_L), dim=1))
        squeeze_tensor = self.sigmoid(out)

        # spatial excitation
        squeeze_tensor = squeeze_tensor.view(batch_size, 1, 1, width, height)
        output_H = torch.mul(input_H, squeeze_tensor)
        output_L = torch.mul(input_L, 1 - squeeze_tensor)
        return output_H, output_L


class SpatialSELayerOct2(nn.Module):

    def __init__(self, num_channels):
        """

        :param num_channels: No of input channels
        """
        super(SpatialSELayerOct2, self).__init__()
        self.conv = nn.Conv2d(num_channels * 2, 1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_H, input_L):
        # spatial squeeze
        assert input_H.size() == input_L.size(), 'input_H and input_L should have the same size'

        # spatial squeeze
        batch_size, _, _, width, height = input_H.size()
        input_tensor = torch.cat(
            (torch.squeeze(input_H, dim=1), torch.squeeze(input_L, dim=1)),
            dim=1
        )
        out = self.conv(input_tensor)
        squeeze_tensor = self.sigmoid(out)

        # spatial excitation
        squeeze_tensor = squeeze_tensor.view(batch_size, 1, 1, width, height)
        output_H = torch.mul(input_H, squeeze_tensor)
        output_L = torch.mul(input_L, 1 - squeeze_tensor)
        return output_H, output_L
