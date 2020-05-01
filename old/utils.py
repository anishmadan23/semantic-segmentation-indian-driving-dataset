import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class conv2DBatchNormRelu(nn.Module):
    def __init__(
        self,
        in_channels,
        num_filters,
        k_size,
        stride,
        padding,
        bias=True,
        dilation=1,
        is_batchnorm=True,
    ):
        super(conv2DBatchNormRelu, self).__init__()

        conv_mod = nn.Conv2d(int(in_channels),
                             int(num_filters),
                             kernel_size=k_size,
                             padding=padding,
                             stride=stride,
                             bias=bias,
                             dilation=dilation,)

        if is_batchnorm:
            self.cbr_unit = nn.Sequential(conv_mod, 
                                          nn.BatchNorm2d(int(num_filters)), 
                                          nn.ReLU(inplace=True))
        else:
            self.cbr_unit = nn.Sequential(conv_mod, nn.ReLU(inplace=True))

    def forward(self, inputs):
        outputs = self.cbr_unit(inputs)
        return outputs


class deconv2DBatchNormRelu(nn.Module):
    def __init__(self, in_channels, num_filters, k_size, stride, padding, bias=True):
        super(deconv2DBatchNormRelu, self).__init__()

        self.dcbr_unit = nn.Sequential(
            nn.ConvTranspose2d(
                int(in_channels),
                int(num_filters),
                kernel_size=k_size,
                padding=padding,
                stride=stride,
                bias=bias,
            ),
            nn.BatchNorm2d(int(num_filters)),
            nn.ReLU(inplace=True),
        )

    def forward(self, inputs):
        outputs = self.dcbr_unit(inputs)
        return outputs

class seg_enc_type1(nn.Module):
    def __init__(self, in_size, out_size):
        super(seg_enc_type1, self).__init__()
        self.conv1 = conv2DBatchNormRelu(in_size, out_size, 3, 1, 1)
        self.conv2 = conv2DBatchNormRelu(out_size, out_size, 3, 1, 1)
        self.maxpool_with_argmax = nn.MaxPool2d(2, 2, return_indices=True)

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        unpooled_shape = outputs.size()
        outputs, indices = self.maxpool_with_argmax(outputs)
        return outputs, indices, unpooled_shape


class seg_enc_type2(nn.Module):
    def __init__(self, in_size, out_size):
        super(seg_enc_type2, self).__init__()
        self.conv1 = conv2DBatchNormRelu(in_size, out_size, 3, 1, 1)
        self.conv2 = conv2DBatchNormRelu(out_size, out_size, 3, 1, 1)
        self.conv3 = conv2DBatchNormRelu(out_size, out_size, 3, 1, 1)
        self.maxpool_with_argmax = nn.MaxPool2d(2, 2, return_indices=True)

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        outputs = self.conv3(outputs)
        unpooled_shape = outputs.size()
        outputs, indices = self.maxpool_with_argmax(outputs)
        return outputs, indices, unpooled_shape

class seg_dec_type1(nn.Module):
    def __init__(self, in_size, out_size):
        super(seg_dec_type1, self).__init__()
        self.unpool = nn.MaxUnpool2d(2, 2)
        self.conv1 = conv2DBatchNormRelu(in_size, in_size, 3, 1, 1)
        self.conv2 = conv2DBatchNormRelu(in_size, out_size, 3, 1, 1)

    def forward(self, inputs, indices, output_shape):
        outputs = self.unpool(input=inputs, indices=indices, output_size=output_shape)
        outputs = self.conv1(outputs)
        outputs = self.conv2(outputs)
        return outputs


class seg_dec_type2(nn.Module):
    def __init__(self, in_size, out_size):
        super(seg_dec_type2, self).__init__()
        self.unpool = nn.MaxUnpool2d(2, 2)
        self.conv1 = conv2DBatchNormRelu(in_size, in_size, 3, 1, 1)
        self.conv2 = conv2DBatchNormRelu(in_size, in_size, 3, 1, 1)
        self.conv3 = conv2DBatchNormRelu(in_size, out_size, 3, 1, 1)

    def forward(self, inputs, indices, output_shape):
        outputs = self.unpool(input=inputs, indices=indices, output_size=output_shape)
        outputs = self.conv1(outputs)
        outputs = self.conv2(outputs)
        outputs = self.conv3(outputs)
        return outputs


from skimage import img_as_float

def checkAccuracy(pred, truth, batch_size):
    pred = pred.cpu().numpy()
    # print(np.unique(pred))
    truth = truth.cpu().numpy()
    acc = np.count_nonzero(pred==truth) / (128*128*batch_size)
    return acc

def checkiou(pred, truth, batch_size):
    intersection = pred & truth
    union = pred | truth
    iou = torch.mean((torch.sum(intersection).float()/torch.sum(union).float()).float()) 
    return iou


def transform_batch(transform, batch_images):
    t = []
    for img in batch_images:
        print(img.shape, img.type)

        img = torch.from_numpy(img_as_float(img.numpy()))
        print(img.max(), img.min())
        img_t = transform(img.repeat(1,3,1,1).permute(1,2,0).numpy())
        print(img_t.shape, img_t.max(), img_t.min())
        exit()
        t.append(img_t)
        # torch.cat((t,img_t), 0)

    # t = np.asarray(t) 
    return torch.stack(t)
