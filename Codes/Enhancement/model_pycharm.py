import torch
import torch.nn as nn


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=(1, 1)),
            nn.ReLU(inplace=True)
        ) # 1216x448x32
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) #608x224x32

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=(1, 1)),
            nn.ReLU(inplace=True) #608x224x64
        )
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) #304x112x64

        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=(1, 1)),
            nn.ReLU(inplace=True)
        ) #304x112x128
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2) #152x56x128

        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=(1, 1)),
            nn.ReLU(inplace=True)
        ) #152x56x256
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2) #76X28X256

        self.conv5 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512)
        ) #76X28X512
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2) #38x14x512

        self.conv6 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512)
        ) #38x14x512
        self.pool6 = nn.MaxPool2d(kernel_size=2, stride=2) #19x7X512

        self.conv7 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(2, 1), padding=0),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512)
        ) #9x5x512

        self.conv8 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(2, 1), padding=0),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512)
        ) #4x3x512

        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(1024)
        ) #2x1x1024

        self.upconv1 = nn.Sequential(
            nn.Upsample(scale_factor=(2, 3), mode='nearest'),
            nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=(1, 1)),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.iconv1 = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512)
        )

        self.upconv2 = nn.Sequential(
            nn.Upsample(size=(9, 5), mode='nearest'),
            # nn.ZeroPad2d((1, 0, 1, 1)),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=(1, 1)),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.iconv2 = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512)
        )

        self.upconv3 = nn.Sequential(
            nn.Upsample(size=(19, 7), mode='nearest'),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=(1, 1)),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.iconv3 = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512)
        )

        self.upconv4 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)
        self.iconv4 = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512)
        )

        self.upconv5 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)
        self.iconv5 = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512),
            nn.Conv2d(512,512, kernel_size=3, stride=1, padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512)
        )

        self.upconv6 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.iconv6 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256)
        )

        self.upconv7 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.iconv7 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128)
        )

        self.upconv8 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.iconv8 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64)
        )

        self.upconv9 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.iconv9 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32)
        )

        self.outconv = nn.Conv2d(32, 3, kernel_size=1, stride=1)

    def forward(self, x):
        conv1 = self.conv1(x)
        pool1 = self.pool1(conv1)

        conv2 = self.conv2(pool1)
        pool2 = self.pool2(conv2)

        conv3 = self.conv3(pool2)
        pool3 = self.pool3(conv3)

        conv4 = self.conv4(pool3)
        pool4 = self.pool4(conv4)

        conv5 = self.conv5(pool4)
        pool5 = self.pool5(conv5)

        conv6 = self.conv6(pool5)
        pool6 = self.pool6(conv6)

        conv7 = self.conv7(pool6)
        conv8 = self.conv8(conv7)

        bottleneck = self.bottleneck(conv8)

        upconv1 = self.upconv1(bottleneck)
        concat1 = torch.cat([upconv1, conv8], dim=1)
        iconv1 = self.iconv1(concat1)

        upconv2 = self.upconv2(iconv1)
        concat2 = torch.cat([upconv2, conv7], dim=1)
        iconv2 = self.iconv2(concat2)

        upconv3 = self.upconv3(iconv2)
        concat3 = torch.cat([upconv3, pool6], dim=1)
        iconv3 = self.iconv3(concat3)

        upconv4 = self.upconv4(iconv3)
        concat4 = torch.cat([upconv4, conv6], dim=1)
        iconv4 = self.iconv4(concat4)

        upconv5 = self.upconv5(iconv4)
        concat5 = torch.cat([upconv5, conv5], dim=1)
        iconv5 = self.iconv5(concat5)

        upconv6 = self.upconv6(iconv5)
        concat6 = torch.cat([upconv6, conv4], dim=1)
        iconv6 = self.iconv6(concat6)

        upconv7 = self.upconv7(iconv6)
        concat7 = torch.cat([upconv7, conv3], dim=1)
        iconv7 = self.iconv7(concat7)

        upconv8 = self.upconv8(iconv7)
        concat8 = torch.cat([upconv8, conv2], dim=1)
        iconv8 = self.iconv8(concat8)

        upconv9 = self.upconv9(iconv8)
        concat9 = torch.cat([upconv9, conv1], dim=1)
        iconv9 = self.iconv9(concat9)

        output = self.outconv(iconv9)

        return output
