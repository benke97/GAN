import torch
import torch.nn as nn
import functools

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

        # Resnet blocks
        resnet_blocks = []
        for i in range(n_blocks):
            resnet_blocks.append(ResnetBlock(256, norm_layer, use_dropout, use_bias))
        self.resnet_blocks = nn.Sequential(*resnet_blocks)

        # Upsampling
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1, bias=use_bias),
            norm_layer(128),
            nn.ReLU()
        )
        self.deconv2 = nn.Sequential(
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
        x4 = self.resnet_blocks(x3)
        x5 = self.deconv1(x4)
        x6 = self.deconv2(x5)
        out1 = self.output1(x6)
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
    

def get_norm_layer(norm):
    if norm == "instance":
        return nn.InstanceNorm2d
    else:
        raise NotImplementedError("Normalization layer is not found:" + norm)