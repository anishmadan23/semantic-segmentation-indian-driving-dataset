import torch
from torch import nn
from torchvision import models
from utils import *

class SegNet(nn.Module):
    def __init__(self, num_classes, pretrained=True):

        self.num_classes = num_classes
        super(SegNet, self).__init__()
        vgg = models.vgg16_bn(pretrained=True)

        features = list(vgg.features.children())
        self.enc1 = nn.Sequential(*features[0:7])
        self.enc2 = nn.Sequential(*features[7:14])
        self.enc3 = nn.Sequential(*features[14:24])
        self.enc4 = nn.Sequential(*features[24:34])
        self.enc5 = nn.Sequential(*features[34:])


        self.dec5 = seg_dec_type2(512, 512)
        self.dec4 = seg_dec_type2(512, 256)
        self.dec3 = seg_dec_type2(256, 128)
        self.dec2 = seg_dec_type1(128, 64)
        self.dec1 = seg_dec_type1(64, num_classes)



    def forward(self, inputs):

        down1, indices_1, unpool_shape1 = self.down1(inputs)
        down2, indices_2, unpool_shape2 = self.down2(down1)
        down3, indices_3, unpool_shape3 = self.down3(down2)
        down4, indices_4, unpool_shape4 = self.down4(down3)
        down5, indices_5, unpool_shape5 = self.down5(down4)

        up5 = self.up5(down5, indices_5, unpool_shape5)
        up4 = self.up4(up5, indices_4, unpool_shape4)
        up3 = self.up3(up4, indices_3, unpool_shape3)
        up2 = self.up2(up3, indices_2, unpool_shape2)
        up1 = self.up1(up2, indices_1, unpool_shape1)

class segnet(nn.Module):
    def __init__(self, num_classes, in_channels=3, is_unpooling=True):
        super(segnet, self).__init__()

        self.num_classes = num_classes
        self.in_channels = in_channels
        self.is_unpooling = is_unpooling

        self.enc1 = seg_enc_type1(self.in_channels, 64)
        self.enc2 = seg_enc_type1(64, 128)
        self.enc3 = seg_enc_type2(128, 256)
        self.enc4 = seg_enc_type2(256, 512)
        self.enc5 = seg_enc_type2(512, 512)

        self.up5 = seg_dec_type2(512, 512)
        self.up4 = seg_dec_type2(512, 256)
        self.up3 = seg_dec_type2(256, 128)
        self.up2 = seg_dec_type1(128, 64)
        self.up1 = seg_dec_type1(64, self.num_classes)

    def forward(self, inputs):

        enc1, indices_1, unpool_shape1 = self.enc1(inputs)
        enc2, indices_2, unpool_shape2 = self.enc2(enc1)
        enc3, indices_3, unpool_shape3 = self.enc3(enc2)
        enc4, indices_4, unpool_shape4 = self.enc4(enc3)
        enc5, indices_5, unpool_shape5 = self.enc5(enc4)

        up5 = self.up5(enc5, indices_5, unpool_shape5)
        up4 = self.up4(up5, indices_4, unpool_shape4)
        up3 = self.up3(up4, indices_3, unpool_shape3)
        up2 = self.up2(up3, indices_2, unpool_shape2)
        up1 = self.up1(up2, indices_1, unpool_shape1)

        return up1

    def init_vgg16_params(self, vgg16):            #initialise params for encoding block with pretrained vgg
        blocks = [self.enc1, self.enc2, self.enc3, self.enc4, self.enc5]

        ranges = [[0, 4], [5, 9], [10, 16], [17, 23], [24, 29]]
        features = list(vgg16.features.children())

        vgg_layers = []
        for _layer in features:
            if isinstance(_layer, nn.Conv2d):
                vgg_layers.append(_layer)

        merged_layers = []
        for idx, conv_block in enumerate(blocks):
            if idx < 2:
                units = [conv_block.conv1.cbr_unit, conv_block.conv2.cbr_unit]
            else:
                units = [
                    conv_block.conv1.cbr_unit,
                    conv_block.conv2.cbr_unit,
                    conv_block.conv3.cbr_unit,
                ]
            for _unit in units:
                for _layer in _unit:
                    if isinstance(_layer, nn.Conv2d):
                        merged_layers.append(_layer)

        assert len(vgg_layers) == len(merged_layers)         #sanity check

        for l1, l2 in zip(vgg_layers, merged_layers):
            if isinstance(l1, nn.Conv2d) and isinstance(l2, nn.Conv2d):
                assert l1.weight.size() == l2.weight.size()
                assert l1.bias.size() == l2.bias.size()
                l2.weight.data = l1.weight.data
                l2.bias.data = l1.bias.data