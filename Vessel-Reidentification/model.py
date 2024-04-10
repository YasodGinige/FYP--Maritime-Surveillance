import torch
import torch.nn as nn
import torchvision.transforms as T
from resnet import resnet34, resnet50, CNN1, CNN2
import math
from torchinfo import summary


class Generator_Block(nn.Module):
    def __init__(self):
        super(Generator_Block, self).__init__()
        nc, nz, ngf = 1, 256, 64
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            # state size. nz x 24 x 24
            nn.ConvTranspose2d(nz, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 48 x 48
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 96 x 96
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Sigmoid()
            # state size. (nc) x 192 x 192
        )

        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, input):
        x = self.main(input)
        return x


class PartAtt_Generator(nn.Module):
    def __init__(self):
        super(PartAtt_Generator, self).__init__()
        self.extractor = resnet34()
        self.generator_front = Generator_Block()
        self.generator_rear = Generator_Block()
        self.generator_side = Generator_Block()

    def forward(self, x):
        x = self.extractor(x, 3)
        front = self.generator_front(x)
        rear = self.generator_rear(x)
        side = self.generator_side(x)
        return torch.cat([front, rear, side], 1)


class Foreground_Generator(nn.Module):
    def __init__(self):
        super(Foreground_Generator, self).__init__()
        nz, ngf = 3, 16
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=nz, out_channels=ngf, kernel_size=3, stride=1, padding="same"),
            nn.ReLU(),
            nn.Conv2d(in_channels=ngf, out_channels=ngf // 2, kernel_size=9, stride=1, padding="same"),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(True))

        self.block2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=15, stride=1, padding='same'),
            nn.ReLU())

        self.block3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=19, stride=1, padding='same'),
            nn.ReLU())

        self.block3_b = nn.Sequential(
            nn.MaxPool2d(kernel_size=2))

        ###############################################

        self.block4 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=19, stride=1, padding='same'),
            nn.ReLU(),
            nn.UpsamplingBilinear2d(scale_factor=2))

        self.block5 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=15, stride=1, padding='same'),
            nn.ReLU(),
            nn.UpsamplingBilinear2d(scale_factor=2))

        self.block6 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=9, stride=1, padding='same'),
            nn.ReLU(),
            nn.UpsamplingBilinear2d(scale_factor=2))

        self.block7 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=5, stride=1, padding='same'),
            nn.Tanh(),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(in_channels=16, out_channels=1, kernel_size=3, stride=1, padding='same'),
            nn.Sigmoid())

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        res1 = self.block1(x)
        res2 = self.block2(res1)
        res3 = self.block3(res2)
        x = self.block3_b(res3)
        out = self.block4(x)
        out += res3
        out2 = self.block5(out)
        out2 += res2
        out3 = self.block6(out2)
        out3 += res1
        res = self.block7(out3)
        res = res.view(-1, 192, 192)
        return res


class Second_Stage_Extractor(nn.Module):
    def __init__(self):
        super(Second_Stage_Extractor, self).__init__()
        self.stage1_extractor = CNN1().eval()  # we don't train the stage 1 extractor
        self.stage2_extractor_global = CNN2(num_features=256)
        self.stage2_extractor_front = CNN2(num_features=128)
        self.stage2_extractor_rear = CNN2(num_features=128)
        self.stage2_extractor_side = CNN2(num_features=128)

    def forward(self, image, front_mask, rear_mask, side_mask):
        # masks should be 24x24
        # front_mask, rear_mask, side_mask = image_masks

        global_stage_1 = self.stage1_extractor(image)
        front_image = torch.mul(global_stage_1, front_mask)
        rear_image = torch.mul(global_stage_1, rear_mask)
        side_image = torch.mul(global_stage_1, side_mask)

        global_features = self.stage2_extractor_global(global_stage_1)
        front_features = self.stage2_extractor_front(front_image)
        rear_features = self.stage2_extractor_rear(rear_image)
        side_features = self.stage2_extractor_side(side_image)
        return torch.cat((global_features, front_features, rear_features, side_features), dim=1)


class BoatIDClassifier(nn.Module):
    def __init__(self, input_size=2560, num_of_classes=100):
        super(BoatIDClassifier, self).__init__()
        self.input_size = input_size
        self.num_classes = num_of_classes
        self.batchNorm = nn.BatchNorm1d(self.input_size)
        self.fc = nn.Linear(self.input_size, self.num_classes)

    def forward(self, features):
        # features should be of size 2560 = 1024 + 512 + 512 + 512
        out = self.batchNorm(features)
        out = self.fc(out)
        # No softmax since softmax is applied in the cross entropy loss
        return out


class Dino_VIT16(nn.Module):
    def __init__(self):
        super(Dino_VIT16, self).__init__()
        self.vit16 = torch.hub.load('facebookresearch/dino:main', 'dino_vits16')
        self.front_linear = torch.nn.Linear(384, 600, bias=True)
        self.rear_linear = torch.nn.Linear(384, 600, bias=True)
        self.side_linear = torch.nn.Linear(384, 600, bias=True)
        self.global_linear = torch.nn.Linear(384, 600, bias=True)
        self.vit16.eval()

    def forward(self, img):
        initial_features = self.vit16(img)
        front_features = self.front_linear(initial_features)
        rear_features = self.rear_linear(initial_features)
        side_features = self.side_linear(initial_features)
        global_features = self.global_linear(initial_features)
        return torch.cat((global_features, front_features, rear_features, side_features), dim=1)
# if __name__ == '__main__':
#     vits16 =
#     batch_size = 16
#     summary(vits16, input_size=(batch_size, 3, 196, 196))
