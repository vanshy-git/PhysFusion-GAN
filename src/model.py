import torch
import torch.nn as nn

# --- U-Net Generator ---
# This is based on the Pix2Pix architecture

class UNetDown(nn.Module):
    """A U-Net encoder (downsampling) block"""
    def __init__(self, in_channels, out_channels, batch_norm=True):
        super(UNetDown, self).__init__()
        layers = [nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False)]
        if batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class UNetUp(nn.Module):
    """A U-Net decoder (upsampling) block"""
    def __init__(self, in_channels, out_channels, dropout=False):
        super(UNetUp, self).__init__()
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        if dropout:
            layers.append(nn.Dropout(0.5))
        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        x = self.model(x)
        # Concatenate with the skip connection from the encoder
        x = torch.cat((x, skip_input), 1)
        return x

class Generator(nn.Module):
    """The U-Net Generator.
    Input: (N, 4, 256, 256) -> Concatenated (LR_thermal + HR_optical)
    Output: (N, 1, 256, 256) -> Fake HR_thermal
    """
    def __init__(self, in_channels=4, out_channels=1):
        super(Generator, self).__init__()
        
        self.down1 = UNetDown(in_channels, 64, batch_norm=False) # 256 -> 128
        self.down2 = UNetDown(64, 128)      # 128 -> 64
        self.down3 = UNetDown(128, 256)     # 64 -> 32
        self.down4 = UNetDown(256, 512)     # 32 -> 16
        self.down5 = UNetDown(512, 512)     # 16 -> 8
        self.down6 = UNetDown(512, 512)     # 8 -> 4
        self.down7 = UNetDown(512, 512)     # 4 -> 2
        
        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 512, 4, 2, 1), nn.ReLU(inplace=True) # 2 -> 1
        )

        self.up1 = UNetUp(512, 512, dropout=True)     # 1 -> 2
        self.up2 = UNetUp(1024, 512, dropout=True)  # 2 -> 4
        self.up3 = UNetUp(1024, 512, dropout=True)  # 4 -> 8
        self.up4 = UNetUp(1024, 512)              # 8 -> 16
        self.up5 = UNetUp(1024, 256)              # 16 -> 32
        self.up6 = UNetUp(512, 128)               # 32 -> 64
        self.up7 = UNetUp(256, 64)                # 64 -> 128

        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(128, out_channels, 4, 2, 1), # 128 -> 256
            nn.Tanh() # Scales output to [-1, 1]
        )

    def forward(self, x):
        # x is the concatenated (lr_th, hr_opt)
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        b = self.bottleneck(d7)
        
        u1 = self.up1(b, d7)
        u2 = self.up2(u1, d6)
        u3 = self.up3(u2, d5)
        u4 = self.up4(u3, d4)
        u5 = self.up5(u4, d3)
        u6 = self.up6(u5, d2)
        u7 = self.up7(u6, d1)
        
        return self.final_layer(u7)


# --- PatchGAN Discriminator ---

class Discriminator(nn.Module):
    """The PatchGAN Discriminator.
    Input: (N, 5, 256, 256) -> Concatenated (LR_thermal + HR_optical + Real/Fake HR_thermal)
    Output: (N, 1, 30, 30) -> A patch of "realness" scores
    """
    def __init__(self, in_channels=5):
        super(Discriminator, self).__init__()

        def disc_block(in_f, out_f, batch_norm=True):
            layers = [nn.Conv2d(in_f, out_f, 4, 2, 1, bias=False)]
            if batch_norm:
                layers.append(nn.BatchNorm2d(out_f))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *disc_block(in_channels, 64, batch_norm=False), # 256 -> 128
            *disc_block(64, 128),  # 128 -> 64
            *disc_block(128, 256), # 64 -> 32
            *disc_block(256, 512), # 32 -> 16 -> 30x30 receptive field
            # Final convolution to produce a 1-channel output
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1) # 16 -> 15 -> 30x30
        )

    def forward(self, lr_in, hr_guide, th_target):
        # Concatenate all inputs along the channel dimension
        # (N, 1, H, W) + (N, 3, H, W) + (N, 1, H, W) -> (N, 5, H, W)
        x = torch.cat([lr_in, hr_guide, th_target], 1)
        return self.model(x)

if __name__ == '__main__':
    # This block allows us to test the models
    print("Testing model architectures...")
    # Create dummy tensors (Batch_size, Channels, Height, Width)
    lr_th = torch.randn(2, 1, 256, 256)
    hr_opt = torch.randn(2, 3, 256, 256)
    hr_th = torch.randn(2, 1, 256, 256)
    
    # Test Generator
    gen_input = torch.cat((lr_th, hr_opt), 1)
    print(f"Generator Input Shape: {gen_input.shape}") # Should be (2, 4, 256, 256)
    
    gen = Generator(in_channels=4, out_channels=1)
    gen_output = gen(gen_input)
    print(f"Generator Output Shape: {gen_output.shape}") # Should be (2, 1, 256, 256)
    
    # Test Discriminator
    disc = Discriminator(in_channels=5)
    disc_output = disc(lr_th, hr_opt, hr_th)
    print(f"Discriminator Input Shape: (2, 1, H, W) + (2, 3, H, W) + (2, 1, H, W)")
    print(f"Discriminator Output Shape: {disc_output.shape}") # Should be (2, 1, 30, 30)
    print("\nModel tests successful!")