import torch
import torch.nn as nn

class ResnetBlock(nn.Module):
    def __init__(self, n_channels, norm_layer, use_dropout, use_bias):
        super(ResnetBlock, self).__init__()
        
        conv_block = [
            nn.ReflectionPad2d(1),
            nn.Conv2d(n_channels, n_channels, kernel_size=3, padding=0, bias=use_bias),
            norm_layer(n_channels),
            nn.ReLU(True)
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
    def __init__(self, norm_layer, n_blocks=6, use_dropout=False):
        super(ResnetGenerator, self).__init__()

        use_bias = norm_layer.func == nn.InstanceNorm2d

        # Conv to inner channel dimensions, 1->64
        self.conv1 = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(1, 64, kernel_size=7, padding=0, bias=use_bias),
            norm_layer(64),
            nn.ReLU(True)
        )

        # Downsampling
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=use_bias),
            norm_layer(128),
            nn.ReLU(True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=use_bias),
            norm_layer(256),
            nn.ReLU(True)
        )

        # Resnet blocks
        resnet_blocks = []
        for i in range(n_blocks):
            resnet_blocks.append(ResnetBlock(256, norm_layer, use_dropout, use_bias))
        self.resnet_blocks = nn.Sequential(*resnet_blocks)

        # Upsampling
        self.conv4 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1, bias=use_bias),
            norm_layer(128),
            nn.ReLU(True)
        )
        self.conv5 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1, bias=use_bias),
            norm_layer(64),
            nn.ReLU(True)
        )

        # Conv to output dimensions
        self.output = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, 1, kernel_size=7, padding=0),
            nn.Tanh() # Used in original paper over sigmoid for some reason
        )

    def forward(self, img):
        x = img

        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.resnet_blocks(x3)
        x5 = self.conv5(x4)
        x6 = self.conv6(x5)
        out = self.output(x6)

        return out
    
class PatchGANDiscriminator(nn.Module):
    def __init__(self, norm_layer):
        super(PatchGANDiscriminator, self).__init__()

        def build_discriminator_block(in_channels, out_channels, norm_layer, use_bias):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=use_bias),
                norm_layer(out_channels),
                nn.LeakyReLU(0.2, True)
            )

        use_bias = norm_layer.func == nn.InstanceNorm2d

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True)
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