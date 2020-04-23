import torch
import torch.nn as nn
import torch.nn.functional as fn


class _netG(nn.Module):
    def __init__(self, nz, nc, ngf):
        super(_netG, self).__init__()

        self.deconv1 = nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False)
        self.deconv1_bn = nn.BatchNorm2d(ngf * 8)

        self.deconv2 = nn.ConvTranspose2d(ngf * 8, ngf * 8, 4, 2, 1, bias=False)
        self.deconv2_bn = nn.BatchNorm2d(ngf * 8)

        self.deconv3 = nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False)
        self.deconv3_bn = nn.BatchNorm2d(ngf * 4)

        self.deconv4 = nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False)
        self.deconv4_bn = nn.BatchNorm2d(ngf * 2)

        self.deconv5 = nn.ConvTranspose2d(ngf * 2, nc, 3, 1, 1)

    def forward(self, input):
        oG_l1 = fn.relu(self.deconv1_bn(self.deconv1(input)))
        oG_l2 = fn.relu(self.deconv2_bn(self.deconv2(oG_l1)))
        oG_l3 = fn.relu(self.deconv3_bn(self.deconv3(oG_l2)))
        oG_l4 = fn.relu(self.deconv4_bn(self.deconv4(oG_l3)))
        oG_out = torch.tanh(self.deconv5(oG_l4))
        return oG_out


class _netE(nn.Module):
    def __init__(self, nc, nez, ndf):
        super(_netE, self).__init__()

        self.conv1 = nn.Conv2d(nc, ndf, 3, 1, 1, bias=False)
        self.conv2 = nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False)
        self.conv3 = nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False)
        self.conv4 = nn.Conv2d(ndf * 4, ndf * 4, 4, 2, 1, bias=False)
        self.conv5 = nn.Conv2d(ndf * 4, nez, 4, 1, 0)

    def forward(self, input):
        oE_l1 = fn.leaky_relu(self.conv1(input), 0.2)
        oE_l2 = fn.leaky_relu(self.conv2(oE_l1), 0.2)
        oE_l3 = fn.leaky_relu(self.conv3(oE_l2), 0.2)
        oE_l4 = fn.leaky_relu(self.conv4(oE_l3), 0.2)
        oE_out = self.conv5(oE_l4)
        return oE_out


class _netI(nn.Module):
    def __init__(self, nc, nz, nif):
        super(_netI, self).__init__()

        self.conv1 = nn.Conv2d(nc, nif, 3, 1, 1, bias=False)
        self.conv2 = nn.Conv2d(nif, nif * 2, 4, 2, 1, bias=False)
        self.conv3 = nn.Conv2d(nif * 2, nif * 4, 4, 2, 1, bias=False)
        self.conv4 = nn.Conv2d(nif * 4, nif * 8, 4, 2, 1, bias=False)
        self.conv51 = nn.Conv2d(nif * 8, nz, 4, 1, 0)  # for mu
        self.conv52 = nn.Conv2d(nif * 8, nz, 4, 1, 0)  # for log_sigma

    def forward(self, input):
        oI_l1 = fn.leaky_relu(self.conv1(input), 0.2)
        oI_l2 = fn.leaky_relu(self.conv2(oI_l1), 0.2)
        oI_l3 = fn.leaky_relu(self.conv3(oI_l2), 0.2)
        oI_l4 = fn.leaky_relu(self.conv4(oI_l3), 0.2)

        oI_mu = self.conv51(oI_l4)
        oI_log_sigma = self.conv52(oI_l4)
        return oI_mu, oI_log_sigma
