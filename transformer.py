# Modified from https://github.com/pytorch/examples/blob/master/fast_neural_style/neural_style/
# The main changes are conditional instance normalization logic 
import torch

class TransformerNet(torch.nn.Module):
    def __init__(self, style_num):
        super(TransformerNet, self).__init__()
        self.conv1 = ConvLayer(3, 32, kernel_size = 9, stride = 1) 
        self.in1 = ConditionalInstanceNorm2d(style_num, 32)

        self.conv2 = ConvLayer(32, 64, kernel_size = 3, stride = 2)
        self.in2 = ConditionalInstanceNorm2d(style_num, 64)

        self.conv3 = ConvLayer(64, 128, kernel_size = 3, stride = 2)
        self.in3 = ConditionalInstanceNorm2d(style_num, 128)

        self.res1 = ResidualBlock(128)
        self.res2 = ResidualBlock(128)
        self.res3 = ResidualBlock(128)
        self.res4 = ResidualBlock(128)
        self.res5 = ResidualBlock(128)
        self.deconv1 = UpsampleConvLayer(128, 64, kernel_size = 3, stride = 1, upsample = 2)
        self.in4 = ConditionalInstanceNorm2d(style_num, 64)
        self.deconv2 = UpsampleConvLayer(64, 32, kernel_size = 3, stride = 1, upsample = 2)
        self.in5 = ConditionalInstanceNorm2d(style_num, 32)
        self.deconv3 = ConvLayer(32, 3, kernel_size = 9, stride = 1)
        self.relu = torch.nn.ReLU()

    def forward(self, X, style_control):
 
        y = self.relu(self.in1(self.conv1(X), style_control))
        y = self.relu(self.in2(self.conv2(y), style_control))
        y = self.relu(self.in3(self.conv3(y), style_control))
        y = self.res1(y)
        y = self.res2(y)
        y = self.res3(y)
        y = self.res4(y)
        y = self.res5(y)
        y = self.relu(self.in4(self.deconv1(y), style_control))
        y = self.relu(self.in5(self.deconv2(y), style_control))
        y = self.deconv3(y) 
        
        return y

class ConditionalInstanceNorm2d(torch.nn.Module):
    """
    Conditional Instance Normalization
    introduced in https://arxiv.org/abs/1610.07629
    """
    def __init__(self, style_num, in_channels):
        super(ConditionalInstanceNorm2d, self).__init__()
        self.condInstanceNorm = torch.nn.ModuleList()
        # we have instance norm for each style
        for i in range(style_num):
            self.condInstanceNorm.append(torch.nn.InstanceNorm2d(in_channels, affine=True))
     
    def forward(self, x, style_control):
        out = []
        for i in range(x.shape[0]):
            total_weight = 0
            out_xi = x[i].clone().fill_(0)
            for j in range(len(style_control[i%len(style_control)])):
                total_weight += style_control[i%len(style_control)][j]
                out_xi += style_control[i%len(style_control)][j] * self.condInstanceNorm[j](x[i].unsqueeze(0)).squeeze_(0)
            out.append(out_xi / total_weight)
        out = torch.stack(out)

        return out

class ConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvLayer, self).__init__()
        reflection_padding = kernel_size // 2 # same dimension after padding
        self.reflection_pad = torch.nn.ReflectionPad2d(reflection_padding)
        self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride) # remember this dimension

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out


class ResidualBlock(torch.nn.Module):
    """ResidualBlock
    introduced in: https://arxiv.org/abs/1512.03385
    recommended architecture: http://torch.ch/blog/2016/02/04/resnets.html
    """

    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.in1 = torch.nn.InstanceNorm2d(channels, affine=True)
        self.conv2 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.in2 = torch.nn.InstanceNorm2d(channels, affine=True)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.relu(self.in1(self.conv1(x)))
        out = self.in2(self.conv2(out))
        out = out + residual # need relu right after
        return out


class UpsampleConvLayer(torch.nn.Module):
    """UpsampleConvLayer
    Upsamples the input and then does a convolution. This method gives better results
    compared to ConvTranspose2d.
    ref: http://distill.pub/2016/deconv-checkerboard/
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, upsample=None):
        super(UpsampleConvLayer, self).__init__()
        self.upsample = upsample
        if upsample:
            self.upsample_layer = torch.nn.Upsample(mode='nearest', scale_factor=upsample)
        reflection_padding = kernel_size // 2
        self.reflection_pad = torch.nn.ReflectionPad2d(reflection_padding)
        self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        x_in = x
        if self.upsample:
            x_in = self.upsample_layer(x_in)
        out = self.reflection_pad(x_in)
        out = self.conv2d(out)
        return out