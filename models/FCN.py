import torch
import torch.nn as nn
import torchvision.models

Vgg16 = torchvision.models.vgg16(pretrained=True)
max_pool = [4, 9, 16, 23, 30]
for param in Vgg16.features.parameters():
    param.requires_grad = False
for i in max_pool:
    Vgg16.features[i].return_indices = True

class ConvBlock(nn.Module):
    def __init__(self, start, pool):
        super(ConvBlock, self).__init__()
        self.convs = Vgg16.features[start:pool]
        self.max_pool = Vgg16.features[pool]

    def forward(self, input):
        output = self.convs(input)
        unpooled_shape = output.size()
        output, indices = self.max_pool(output)
        return output, indices, unpooled_shape

class ConvBlocks(nn.Module):
    def __init__(self):
        super(ConvBlocks, self).__init__()
        self.convBlock_1 = ConvBlock(start=0, pool=max_pool[0])
        self.convBlock_2 = ConvBlock(start=max_pool[0]+1, pool=max_pool[1])
        self.convBlock_3 = ConvBlock(start=max_pool[1]+1, pool=max_pool[2])
        self.convBlock_4 = ConvBlock(start=max_pool[2]+1, pool=max_pool[3])
        self.convBlock_5 = ConvBlock(start=max_pool[3]+1, pool=max_pool[4])
    def forward(self, input):
        out_1, indice_1, unpool_shape_1 = self.convBlock_1(input)
        out_2, indice_2, unpool_shape_2 = self.convBlock_2(out_1)
        out_3, indice_3, unpool_shape_3 = self.convBlock_3(out_2)
        out_4, indice_4, unpool_shape_4 = self.convBlock_4(out_3)
        out_5, indice_5, unpool_shape_5 = self.convBlock_5(out_4)
        output = [
            out_1,
            out_2,
            out_3,
            out_4,
            out_5
        ]
        indices = [indice_1,
                   indice_2,
                   indice_3,
                   indice_4,
                   indice_5]
        unpool_shapes = [
            unpool_shape_1,
            unpool_shape_2,
            unpool_shape_3,
            unpool_shape_4,
            unpool_shape_5
        ]
        return output, indices, unpool_shapes

class FCN(nn.Module):
    def __init__(self, network=ConvBlocks(), n_class=21):
        super(FCN, self).__init__()
        self.n_class = n_class
        self.featureExtractor = network
        self.relu = nn.ReLU(inplace=True)
        self.deconv1 = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn1 = nn.BatchNorm2d(512)
        self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.deconv5 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn5 = nn.BatchNorm2d(32)
        self.classifier = nn.Conv2d(32, n_class, kernel_size=1)

    def forward(self, input):
        output, indices, unpool_shapes = self.featureExtractor(input)
        x1 = output[0]
        x2 = output[1]
        x3 = output[2]
        x4 = output[3]
        x5 = output[4]
        score = self.bn1(self.relu(self.deconv1(x5)))
        score = torch.add(score, x4)
        score = self.bn2(self.relu(self.deconv2(score)))
        score = torch.add(score, x3)
        score = self.bn3(self.relu(self.deconv3(score)))
        score = torch.add(score, x2)
        score = self.bn4(self.relu(self.deconv4(score)))
        score = torch.add(score, x1)
        score = self.bn5(self.relu(self.deconv5(score)))
        score = self.classifier(score)
        return score, indices, unpool_shapes

if __name__ == '__main__':
    print()