import torch
import torch.nn as nn
import torchvision.models as models


class MCANet(nn.Module):
    def __init__(self):
        super(MCANet, self).__init__()

        # feature extraction
        self.conv_feature_extract = nn.Conv2d(
            in_channels=6, out_channels=64, kernel_size=3, padding=1
        )
        self.mcab21 = MCAB()
        self.mcab22 = MCAB()
        self.mcab23 = MCAB()

        # reconstruction
        self.conv3 = nn.Conv2d(
            in_channels=64 * 3, out_channels=64, kernel_size=3, padding=1
        )
        self.res4 = ResBlock()
        self.res5 = ResBlock()
        self.res6 = ResBlock()
        self.res7 = ResBlock()
        self.res8 = ResBlock()
        self.res9 = ResBlock()
        self.conv10 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=3, padding=1
        )
        self.conv11 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=3, padding=1
        )
        self.conv12 = nn.Conv2d(
            in_channels=64, out_channels=3, kernel_size=3, padding=1
        )

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        # feature extraction
        h, w = x.shape[2],x.shape[3]
        x = x.view(-1, 6, h, w)
        f123 = self.relu(self.conv_feature_extract(x))
        f123 = f123.view(-1, 192, h, w)

        f1p = self.mcab21(f123[:, 0:64, ...])
        f2p = self.mcab22(f123[:, 64:128, ...])
        f3p = self.mcab23(f123[:, 128:192, ...])

        f123p = torch.cat((f1p, f2p, f3p), dim=1)

        # reconstruction
        out = self.relu(f123p)
        out = self.relu(self.conv3(out))
        out = self.res4(out)
        out = self.res5(out)
        out = self.res6(out)
        out = self.res7(out)
        out = self.res8(out)
        out = self.res9(out)
        out = self.relu(self.conv10(out))
        out = self.relu(self.conv11(out))
        out = self.sigmoid(self.conv12(out))

        return out


class CAB(nn.Module):
    def __init__(self):
        super(CAB, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1)
        self.relu = nn.ReLU()

    def forward(self, z1, z2):
        A1 = torch.cat((z1, z2), 1).mean(dim=(2, 3), keepdim=True)
        A2 = self.conv2(self.relu(self.conv1(A1)))
        return A2 * z2


class MCAB(nn.Module):
    def __init__(self):
        super(MCAB, self).__init__()

        self.dilconv1 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=3, padding=1, dilation=1
        )
        self.dilconv2 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=3, padding=3, dilation=3
        )
        self.dilconv3 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=3, padding=5, dilation=5
        )
        self.dilconv4 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=3, padding=7, dilation=7
        )
        self.conv = nn.Conv2d(in_channels=64 * 4, out_channels=64, kernel_size=1)
        self.cab1 = CAB()
        self.cab2 = CAB()
        self.cab3 = CAB()
        self.relu = nn.ReLU()

    def forward(self, f_i):
        y1 = self.relu(self.dilconv1(f_i))
        y2 = self.relu(self.dilconv2(y1))
        y2p = self.cab1(y1, y2)
        y3 = self.relu(self.dilconv3(y2p))
        y3p = self.cab2(y2p, y3)
        y4 = self.relu(self.dilconv4(y3p))
        y4p = self.cab3(y3p, y4)

        return f_i + self.conv(torch.cat((y1, y2p, y3p, y4p), dim=1))


class ResBlock(nn.Module):
    def __init__(self):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=3, padding=1
        )
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=3, padding=1
        )

    def forward(self, x):
        return self.relu(x + self.conv2(self.relu(self.conv1(x))))
