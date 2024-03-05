import torch
import torch.nn as nn
import torch.nn.functional as F
import functools
from torch.nn.utils.parametrizations import spectral_norm

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        def conv_block(in_channels, out_channels, dropout_rate=0.3):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                #nn.Dropout(dropout_rate)
        )

        def upconv_block(in_channels, out_channels):
            return nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
                nn.BatchNorm2d(out_channels)
        )

        # Downsampling
        self.conv1 = conv_block(1, 128)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = conv_block(128, 256)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = conv_block(256, 512)
        self.pool3 = nn.MaxPool2d(2, 2)

        # Bridge
        self.conv4 = conv_block(512, 1024)

        # Upsampling
        self.upconv5 = upconv_block(1024, 512)
        self.conv5 = conv_block(1024, 512)

        self.upconv6 = upconv_block(512, 256)
        self.conv6 = conv_block(512, 256)

        self.upconv7 = upconv_block(256, 128)
        self.conv7 = conv_block(256, 128)

        self.output = nn.Sequential(
            nn.Conv2d(128, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, img):
        x = img
        x1 = self.conv1(x)
        x2 = self.pool1(x1)
        x3 = self.conv2(x2)
        x4 = self.pool2(x3)
        x5 = self.conv3(x4)
        x6 = self.pool3(x5)

        x7 = self.conv4(x6)

        x8 = self.upconv5(x7)
        x9 = torch.cat([x8, x5], dim=1)
        x10 = self.conv5(x9)

        x11 = self.upconv6(x10)
        x12 = torch.cat([x11, x3], dim=1)
        x13 = self.conv6(x12)

        x14 = self.upconv7(x13)
        x15 = torch.cat([x14, x1], dim=1)
        x16 = self.conv7(x15)

        out = self.output(x16)

        return out
    
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        def discriminator_block(in_channels, out_channels, normalization=True):
            layers = [nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)]
            if normalization:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return nn.Sequential(*layers)

        # Specify the input and output channels
        input_channels = 1 # for grayscale images

        # Discriminator architecture
        self.conv1 = discriminator_block(input_channels, 64, normalization=False)
        self.conv2 = discriminator_block(64, 128)
        self.conv3 = discriminator_block(128, 256)
        self.conv4 = discriminator_block(256, 512)

        # The following padding is used to adjust the shape of the output
        self.pad = nn.ZeroPad2d((1, 0, 1, 0))
        self.final_conv = nn.Conv2d(512, 1, kernel_size=4, padding=1)

    def forward(self, img):
        # Forward pass through the discriminator layers
        x = self.conv1(img)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pad(x)
        x = self.final_conv(x)
        return torch.sigmoid(x)


class ResnetBlock(nn.Module):
    def __init__(self, n_channels, norm_layer, use_dropout, use_bias):
        super(ResnetBlock, self).__init__()
        
        conv_block = [
            nn.ReflectionPad2d(1),
            nn.Conv2d(n_channels, n_channels, kernel_size=3, padding=0, bias=use_bias),
            norm_layer(n_channels),
            nn.ReLU()
        ]
        
        if use_dropout:
            conv_block.append(nn.Dropout(0.5))
        
        conv_block += [
            nn.ReflectionPad2d(1),
            nn.Conv2d(n_channels, n_channels, kernel_size=3, padding=0, bias=use_bias),
            norm_layer(n_channels)
        ]

        self.conv_block = nn.Sequential(*conv_block)
        
    def forward(self, x):
        out = x + self.conv_block(x) # Skip connection
        return out


class ResnetGenerator(nn.Module):
    def __init__(self, norm, n_blocks=6, use_dropout=False):
        super(ResnetGenerator, self).__init__()

        norm_layer = get_norm_layer(norm)

        use_bias = norm_layer == nn.InstanceNorm2d

        # Conv to inner channel dimensions, 1->64
        self.conv1 = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(1, 64, kernel_size=7, padding=0, bias=use_bias),
            norm_layer(64),
            nn.ReLU()
        )

        # Downsampling
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=use_bias),
            norm_layer(128),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=use_bias),
            norm_layer(256),
            nn.ReLU()
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=use_bias),
            norm_layer(512),
            nn.ReLU()
        )

        # Resnet blocks
        resnet_blocks = []
        for i in range(n_blocks):
            resnet_blocks.append(ResnetBlock(512, norm_layer, use_dropout, use_bias))
        self.resnet_blocks = nn.Sequential(*resnet_blocks)

        # Upsampling
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1, bias=use_bias),
            norm_layer(256),
            nn.ReLU()
        )
        
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1, bias=use_bias),
            norm_layer(128),
            nn.ReLU()
        )
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1, bias=use_bias),
            norm_layer(64),
            nn.ReLU()
        )

        # Conv to output dimensions
        #self.output = nn.Sequential(
        #    nn.ReflectionPad2d(3),
        #    nn.Conv2d(64, 1, kernel_size=7, padding=0),
        #    nn.Tanh() # Used in original paper over sigmoid for some reason
        #)
        self.output1 = nn.ReflectionPad2d(3)
        self.output2 = nn.Conv2d(64, 1, kernel_size=7, padding=0)
        self.output3 = nn.Sigmoid()

    def forward(self, img):
        x = img

        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.resnet_blocks(x4)
        x6 = self.deconv1(x5)
        x7 = self.deconv2(x6)
        x8 = self.deconv3(x7)
        out1 = self.output1(x8)
        out1_old = out1.clone()
        out2 = self.output2(out1)
        out2_old = out2.clone()
        out = self.output3(out2)
        #out = self.output(x6)

        return out
    
class PatchGANDiscriminator(nn.Module):
    def __init__(self, norm):
        super(PatchGANDiscriminator, self).__init__()

        def build_discriminator_block(in_channels, out_channels, norm_layer, use_bias):
            if norm_layer == None:
                return nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
                    nn.LeakyReLU(0.2)
                )
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=use_bias),
                norm_layer(out_channels),
                nn.LeakyReLU(0.2)
            )

        norm_layer = get_norm_layer(norm)
        
        use_bias = norm_layer == nn.InstanceNorm2d

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2)
        )

        self.conv2 = build_discriminator_block(64, 128, norm_layer, use_bias)

        self.conv3 = build_discriminator_block(128, 256, norm_layer, use_bias)

        self.conv4 = build_discriminator_block(256, 512, norm_layer, use_bias)

        self.output = nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1)


    def forward(self, img):
        # Forward pass through the discriminator layers
        x = self.conv1(img)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.output(x)
        return torch.sigmoid(x) 

class SNPatchGANDiscriminator(nn.Module):
    def __init__(self, norm, npi=1):
        super(SNPatchGANDiscriminator, self).__init__()

        def build_discriminator_block(in_channels, out_channels, norm_layer, use_bias,npi=1):
            layers = [
                spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=use_bias), n_power_iterations=npi),
            ]
            if norm_layer is not None:
                layers.append(norm_layer(out_channels))
            layers.append(nn.LeakyReLU(0.2))
            return nn.Sequential(*layers)

        norm_layer = get_norm_layer(norm)
        use_bias = norm_layer == nn.InstanceNorm2d

        self.conv1 = spectral_norm(nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1))
        self.conv1.add_module('leaky_relu', nn.LeakyReLU(0.2))

        self.conv2 = build_discriminator_block(64, 128, norm_layer, use_bias)

        self.conv3 = build_discriminator_block(128, 256, norm_layer, use_bias)

        self.conv4 = build_discriminator_block(256, 512, norm_layer, use_bias)

        self.output = spectral_norm(nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1))

    def forward(self, img):
        # Forward pass through the discriminator layers
        x = self.conv1(img)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.output(x)
        return torch.sigmoid(x)

def get_norm_layer(norm):
    if norm == "instance":
        return nn.InstanceNorm2d
    elif norm == "batch":
        return nn.BatchNorm2d
    elif norm == "None":
        return None
    else:
        raise NotImplementedError("Normalization layer is not found:" + norm)
    


class UNet128(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64):
        super(UNet128, self).__init__()
        
        # Downsample
        self.e1 = nn.Conv2d(input_nc, ngf, kernel_size=4, stride=2, padding=1)
        self.e2 = nn.Sequential(nn.LeakyReLU(0.2, inplace=True),
                                nn.Conv2d(ngf, ngf * 2, kernel_size=4, stride=2, padding=1),
                                nn.BatchNorm2d(ngf * 2))
        self.e3 = nn.Sequential(nn.LeakyReLU(0.2, inplace=True),
                                nn.Conv2d(ngf * 2, ngf * 4, kernel_size=4, stride=2, padding=1),
                                nn.BatchNorm2d(ngf * 4))
        self.e4 = nn.Sequential(nn.LeakyReLU(0.2, inplace=True),
                                nn.Conv2d(ngf * 4, ngf * 8, kernel_size=4, stride=2, padding=1),
                                nn.BatchNorm2d(ngf * 8))
        self.e5 = nn.Sequential(nn.LeakyReLU(0.2, inplace=True),
                                nn.Conv2d(ngf * 8, ngf * 8, kernel_size=4, stride=2, padding=1),
                                nn.BatchNorm2d(ngf * 8))
        self.e6 = nn.Sequential(nn.LeakyReLU(0.2, inplace=True),
                                nn.Conv2d(ngf * 8, ngf * 8, kernel_size=4, stride=2, padding=1),
                                nn.BatchNorm2d(ngf * 8))
        self.e7 = nn.Sequential(nn.LeakyReLU(0.2, inplace=True),
                                nn.Conv2d(ngf * 8, ngf * 8, kernel_size=4, stride=2, padding=1))
        
        # Upsample
        self.d1 = nn.Sequential(nn.ReLU(inplace=True),
                                nn.ConvTranspose2d(ngf * 8, ngf * 8, kernel_size=4, stride=2, padding=1),
                                nn.BatchNorm2d(ngf * 8),
                                nn.Dropout(0.5))
        self.d2 = nn.Sequential(nn.ReLU(inplace=True),
                                nn.ConvTranspose2d(ngf * 16, ngf * 8, kernel_size=4, stride=2, padding=1),
                                nn.BatchNorm2d(ngf * 8),
                                nn.Dropout(0.5))
        self.d3 = nn.Sequential(nn.ReLU(inplace=True),
                                nn.ConvTranspose2d(ngf * 16, ngf * 8, kernel_size=4, stride=2, padding=1),
                                nn.BatchNorm2d(ngf * 8),
                                nn.Dropout(0.5))
        self.d4 = nn.Sequential(nn.ReLU(inplace=True),
                                nn.ConvTranspose2d(ngf * 16, ngf * 4, kernel_size=4, stride=2, padding=1),
                                nn.BatchNorm2d(ngf * 4))
        self.d5 = nn.Sequential(nn.ReLU(inplace=True),
                                nn.ConvTranspose2d(ngf * 8, ngf * 2, kernel_size=4, stride=2, padding=1),
                                nn.BatchNorm2d(ngf * 2))
        self.d6 = nn.Sequential(nn.ReLU(inplace=True),
                                nn.ConvTranspose2d(ngf * 4, ngf, kernel_size=4, stride=2, padding=1),
                                nn.BatchNorm2d(ngf))
        self.d7 = nn.Sequential(nn.ReLU(inplace=True),
                                nn.ConvTranspose2d(ngf * 2, output_nc, kernel_size=4, stride=2, padding=1))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Encoder
        e1 = self.e1(x)
        e2 = self.e2(e1)
        e3 = self.e3(e2)
        e4 = self.e4(e3)
        e5 = self.e5(e4)
        e6 = self.e6(e5)
        e7 = self.e7(e6)
        
        # Decoder with skip connections
        d1_ = self.d1(e7)
        d1 = torch.cat([d1_, e6], dim=1)
        d2_ = self.d2(d1)
        d2 = torch.cat([d2_, e5], dim=1)
        d3_ = self.d3(d2)
        d3 = torch.cat([d3_, e4], dim=1)
        d4_ = self.d4(d3)
        d4 = torch.cat([d4_, e3], dim=1)
        d5_ = self.d5(d4)
        d5 = torch.cat([d5_, e2], dim=1)
        d6_ = self.d6(d5)
        d6 = torch.cat([d6_, e1], dim=1)
        d7 = self.d7(d6)
        
        # Output
        o1 = self.sigmoid(d7)
        
        return o1
    

class UNet128_5(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64):
        super(UNet128_5, self).__init__()
        
        # Downsample
        self.e1 = nn.Conv2d(input_nc, ngf, kernel_size=4, stride=2, padding=1)
        self.e2 = nn.Sequential(nn.LeakyReLU(0.2, inplace=True),
                                nn.Conv2d(ngf, ngf * 2, kernel_size=4, stride=2, padding=1),
                                nn.BatchNorm2d(ngf * 2))
        self.e3 = nn.Sequential(nn.LeakyReLU(0.2, inplace=True),
                                nn.Conv2d(ngf * 2, ngf * 4, kernel_size=4, stride=2, padding=1),
                                nn.BatchNorm2d(ngf * 4))
        self.e4 = nn.Sequential(nn.LeakyReLU(0.2, inplace=True),
                                nn.Conv2d(ngf * 4, ngf * 8, kernel_size=4, stride=2, padding=1),
                                nn.BatchNorm2d(ngf * 8))
        self.e5 = nn.Sequential(nn.LeakyReLU(0.2, inplace=True),
                                nn.Conv2d(ngf * 8, ngf * 8, kernel_size=4, stride=2, padding=1),
                                nn.BatchNorm2d(ngf * 8))
        self.e6 = nn.Sequential(nn.LeakyReLU(0.2, inplace=True),
                                nn.Conv2d(ngf * 8, ngf * 8, kernel_size=4, stride=2, padding=1),
                                nn.BatchNorm2d(ngf * 8))

        # Upsample

        self.d2 = nn.Sequential(nn.ReLU(inplace=True),
                                nn.ConvTranspose2d(ngf * 8, ngf * 8, kernel_size=4, stride=2, padding=1),
                                nn.BatchNorm2d(ngf * 8),
                                nn.Dropout(0.5))
        self.d3 = nn.Sequential(nn.ReLU(inplace=True),
                                nn.ConvTranspose2d(ngf * 16, ngf * 8, kernel_size=4, stride=2, padding=1),
                                nn.BatchNorm2d(ngf * 8),
                                nn.Dropout(0.5))
        self.d4 = nn.Sequential(nn.ReLU(inplace=True),
                                nn.ConvTranspose2d(ngf * 16, ngf * 4, kernel_size=4, stride=2, padding=1),
                                nn.BatchNorm2d(ngf * 4))
        self.d5 = nn.Sequential(nn.ReLU(inplace=True),
                                nn.ConvTranspose2d(ngf * 8, ngf * 2, kernel_size=4, stride=2, padding=1),
                                nn.BatchNorm2d(ngf * 2))
        self.d6 = nn.Sequential(nn.ReLU(inplace=True),
                                nn.ConvTranspose2d(ngf * 4, ngf, kernel_size=4, stride=2, padding=1),
                                nn.BatchNorm2d(ngf))
        self.d7 = nn.Sequential(nn.ReLU(inplace=True),
                                nn.ConvTranspose2d(ngf * 2, output_nc, kernel_size=4, stride=2, padding=1))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Encoder
        e1 = self.e1(x) # 64
        e2 = self.e2(e1) # 128
        e3 = self.e3(e2) # 256
        e4 = self.e4(e3) # 512
        e5 = self.e5(e4) # 512
        e6 = self.e6(e5) # 512
        #e7 = self.e7(e6)
        
        # Decoder with skip connections
        #d1_ = self.d1(e7)
        #d1 = torch.cat([d1_, e6], dim=1)
        d2_ = self.d2(e6) # in 1024, out 512
        d2 = torch.cat([d2_, e5], dim=1) #1024
        d3_ = self.d3(d2) 
        d3 = torch.cat([d3_, e4], dim=1)
        d4_ = self.d4(d3)
        d4 = torch.cat([d4_, e3], dim=1)
        d5_ = self.d5(d4)
        d5 = torch.cat([d5_, e2], dim=1)
        d6_ = self.d6(d5)
        d6 = torch.cat([d6_, e1], dim=1)
        d7 = self.d7(d6)
        
        # Output
        o1 = self.sigmoid(d7)
        
        return o1

class UNet128_4(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=16):
        super(UNet128_4, self).__init__()
        
        # Downsample
        self.e1 = nn.Conv2d(input_nc, ngf, kernel_size=4, stride=2, padding=1)
        self.e2 = nn.Sequential(nn.LeakyReLU(0.2, inplace=True),
                                nn.Conv2d(ngf, ngf * 2, kernel_size=4, stride=2, padding=1),
                                nn.BatchNorm2d(ngf * 2))
        self.e3 = nn.Sequential(nn.LeakyReLU(0.2, inplace=True),
                                nn.Conv2d(ngf * 2, ngf * 4, kernel_size=4, stride=2, padding=1),
                                nn.BatchNorm2d(ngf * 4))
        self.e4 = nn.Sequential(nn.LeakyReLU(0.2, inplace=True),
                                nn.Conv2d(ngf * 4, ngf * 8, kernel_size=4, stride=2, padding=1),
                                nn.BatchNorm2d(ngf * 8))
        self.e5 = nn.Sequential(nn.LeakyReLU(0.2, inplace=True),
                                nn.Conv2d(ngf * 8, ngf * 16, kernel_size=4, stride=2, padding=1),
                                nn.BatchNorm2d(ngf * 16))

        # Upsample

        self.d3 = nn.Sequential(nn.ReLU(inplace=True),
                                nn.ConvTranspose2d(ngf * 16, ngf * 8, kernel_size=4, stride=2, padding=1),
                                nn.BatchNorm2d(ngf * 8),
                                nn.Dropout(0.5))
        self.d4 = nn.Sequential(nn.ReLU(inplace=True),
                                nn.ConvTranspose2d(ngf * 16, ngf * 4, kernel_size=4, stride=2, padding=1),
                                nn.BatchNorm2d(ngf * 4))
        self.d5 = nn.Sequential(nn.ReLU(inplace=True),
                                nn.ConvTranspose2d(ngf * 8, ngf * 2, kernel_size=4, stride=2, padding=1),
                                nn.BatchNorm2d(ngf * 2))
        self.d6 = nn.Sequential(nn.ReLU(inplace=True),
                                nn.ConvTranspose2d(ngf * 4, ngf, kernel_size=4, stride=2, padding=1),
                                nn.BatchNorm2d(ngf))
        self.d7 = nn.Sequential(nn.ReLU(inplace=True),
                                nn.ConvTranspose2d(ngf * 2, output_nc, kernel_size=4, stride=2, padding=1))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Encoder
        e1 = self.e1(x) # 16
        e2 = self.e2(e1) # 32
        e3 = self.e3(e2) # 64
        e4 = self.e4(e3) # 128
        e5 = self.e5(e4) # 128
        #e6 = self.e6(e5) # 512
        #e7 = self.e7(e6)
        
        # Decoder with skip connections
        #d1_ = self.d1(e7)
        #d1 = torch.cat([d1_, e6], dim=1)
        #d2_ = self.d2(e6) # in 1024, out 512
        #d2 = torch.cat([d2_, e5], dim=1) #1024
        d3_ = self.d3(e5) 
        d3 = torch.cat([d3_, e4], dim=1) #256
        d4_ = self.d4(d3)
        d4 = torch.cat([d4_, e3], dim=1) #128
        d5_ = self.d5(d4)
        d5 = torch.cat([d5_, e2], dim=1)
        d6_ = self.d6(d5)
        d6 = torch.cat([d6_, e1], dim=1)
        d7 = self.d7(d6)
        
        # Output
        o1 = self.sigmoid(d7)
        
        return o1
    
class UNet128_3(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64):
        super(UNet128_3, self).__init__()
        
        # Downsample
        self.e1 = nn.Conv2d(input_nc, ngf, kernel_size=4, stride=2, padding=1)
        self.e2 = nn.Sequential(nn.LeakyReLU(0.2, inplace=True),
                                nn.Conv2d(ngf, ngf * 2, kernel_size=4, stride=2, padding=1),
                                nn.BatchNorm2d(ngf * 2))
        self.e3 = nn.Sequential(nn.LeakyReLU(0.2, inplace=True),
                                nn.Conv2d(ngf * 2, ngf * 4, kernel_size=4, stride=2, padding=1),
                                nn.BatchNorm2d(ngf * 4))
        self.e4 = nn.Sequential(nn.LeakyReLU(0.2, inplace=True),
                                nn.Conv2d(ngf * 4, ngf * 8, kernel_size=4, stride=2, padding=1),
                                nn.BatchNorm2d(ngf * 8))
        # Upsample

        self.d4 = nn.Sequential(nn.ReLU(inplace=True),
                                nn.ConvTranspose2d(ngf * 8, ngf * 4, kernel_size=4, stride=2, padding=1),
                                nn.BatchNorm2d(ngf * 4))
        self.d5 = nn.Sequential(nn.ReLU(inplace=True),
                                nn.ConvTranspose2d(ngf * 8, ngf * 2, kernel_size=4, stride=2, padding=1),
                                nn.BatchNorm2d(ngf * 2))
        self.d6 = nn.Sequential(nn.ReLU(inplace=True),
                                nn.ConvTranspose2d(ngf * 4, ngf, kernel_size=4, stride=2, padding=1),
                                nn.BatchNorm2d(ngf))
        self.d7 = nn.Sequential(nn.ReLU(inplace=True),
                                nn.ConvTranspose2d(ngf * 2, output_nc, kernel_size=4, stride=2, padding=1))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Encoder
        e1 = self.e1(x) # 64
        e2 = self.e2(e1) # 128
        e3 = self.e3(e2) # 256
        e4 = self.e4(e3) # 512
        #e6 = self.e6(e5) # 512
        #e7 = self.e7(e6)
        
        # Decoder with skip connections
        #d1_ = self.d1(e7)
        #d1 = torch.cat([d1_, e6], dim=1)
        #d2_ = self.d2(e6) # in 1024, out 512
        #d2 = torch.cat([d2_, e5], dim=1) #1024
        d4_ = self.d4(e4)
        d4 = torch.cat([d4_, e3], dim=1)
        d5_ = self.d5(d4)
        d5 = torch.cat([d5_, e2], dim=1)
        d6_ = self.d6(d5)
        d6 = torch.cat([d6_, e1], dim=1)
        d7 = self.d7(d6)
        
        # Output
        o1 = self.sigmoid(d7)
        
        return o1
    


class UNet_size_estimator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=16):
        super(UNet_size_estimator, self).__init__()
        
        # Downsample
        self.e1 = nn.Conv2d(input_nc, ngf, kernel_size=4, stride=2, padding=1)
        self.e2 = nn.Sequential(nn.LeakyReLU(0.2, inplace=True),
                                nn.Conv2d(ngf, ngf * 2, kernel_size=4, stride=2, padding=1),
                                nn.BatchNorm2d(ngf * 2))
        self.e3 = nn.Sequential(nn.LeakyReLU(0.2, inplace=True),
                                nn.Conv2d(ngf * 2, ngf * 4, kernel_size=4, stride=2, padding=1),
                                nn.BatchNorm2d(ngf * 4))
        self.e4 = nn.Sequential(nn.LeakyReLU(0.2, inplace=True),
                                nn.Conv2d(ngf * 4, ngf * 8, kernel_size=4, stride=2, padding=1),
                                nn.BatchNorm2d(ngf * 8),
                                nn.Dropout(0.5))
        self.e5 = nn.Sequential(nn.LeakyReLU(0.2, inplace=True),
                                nn.Conv2d(ngf * 8, ngf * 8, kernel_size=4, stride=2, padding=1),
                                nn.BatchNorm2d(ngf * 8),
                                nn.Dropout(0.5))

        # Upsample

        self.d3 = nn.Sequential(nn.ReLU(inplace=True),
                                nn.ConvTranspose2d(ngf * 8, ngf * 8, kernel_size=4, stride=2, padding=1),
                                nn.BatchNorm2d(ngf * 8),
                                nn.Dropout(0.5))
        self.d4 = nn.Sequential(nn.ReLU(inplace=True),
                                nn.ConvTranspose2d(ngf * 16, ngf * 4, kernel_size=4, stride=2, padding=1),
                                nn.BatchNorm2d(ngf * 4),
                                nn.Dropout(0.5))
        self.d5 = nn.Sequential(nn.ReLU(inplace=True),
                                nn.ConvTranspose2d(ngf * 8, ngf * 2, kernel_size=4, stride=2, padding=1),
                                nn.BatchNorm2d(ngf * 2))
        self.d6 = nn.Sequential(nn.ReLU(inplace=True),
                                nn.ConvTranspose2d(ngf * 4, ngf, kernel_size=4, stride=2, padding=1),
                                nn.BatchNorm2d(ngf))
        self.d7 = nn.Sequential(nn.ReLU(inplace=True),
                                nn.ConvTranspose2d(ngf * 2, ngf, kernel_size=4, stride=2, padding=1))
        
        self.pool = nn.AdaptiveAvgPool2d((1, 1))  # Global average pooling
        self.fc = nn.Linear(ngf, 1)  # Output one scalar value


    def forward(self, x):
        # Encoder
        e1 = self.e1(x) # 64
        e2 = self.e2(e1) # 128
        e3 = self.e3(e2) # 256
        e4 = self.e4(e3) # 512
        e5 = self.e5(e4) # 512

        d3_ = self.d3(e5) 
        d3 = torch.cat([d3_, e4], dim=1) #256
        d4_ = self.d4(d3)
        d4 = torch.cat([d4_, e3], dim=1) #128
        d5_ = self.d5(d4)
        d5 = torch.cat([d5_, e2], dim=1)
        d6_ = self.d6(d5)
        d6 = torch.cat([d6_, e1], dim=1)
        d7 = self.d7(d6)
        
        # Output
        pooled = self.pool(d7)
        flattened = torch.flatten(pooled, 1)
        o1 = self.fc(flattened).squeeze(-1) 
        return o1