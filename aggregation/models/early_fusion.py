import torch
import torch.nn as nn
import torch.nn.functional as F

from .unet_blur import down as down_blur
from .unet import down, up, inconv, outconv, double_conv

class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate=0, blur=False):
        super(Encoder, self).__init__()
        
        self.inc = inconv(in_channels, 64)
        if blur:
            self.down1 = down_blur(64, 128)
            self.down2 = down_blur(128, 256)
            self.down3 = down_blur(256, 512)
            self.down4 = down_blur(512, 512)
        else:
            self.down1 = down(64, 128)
            self.down2 = down(128, 256)
            self.down3 = down(256, 512)
            self.down4 = down(512, 512)

        self.drop = nn.Dropout(p=dropout_rate)
    
    def forward(self, x):
        
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.drop(self.down2(x2))
        x4 = self.drop(self.down3(x3))
        x5 = self.drop(self.down4(x4))

        return x5, x4, x3, x2, x1, x


class EF_UNet(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate=0, blur=False):
        super(EF_UNet, self).__init__()
        
        self.encoder = Encoder(2*in_channels, out_channels, dropout_rate, blur)
        
        self.up1 = up(1024, 256)
        self.up2 = up(512, 128)
        self.up3 = up(256, 64)
        self.up4 = up(128, 64)
        self.outc = outconv(64, out_channels)

        self.drop = nn.Dropout(p=dropout_rate)

    def forward(self, y, z):
        
        x = torch.cat((y,z), dim=1)
        
        x5, x4, x3, x2, x1, x = self.encoder(x)
        
        x = self.drop(self.up1(x5, x4))
        x = self.drop(self.up2(x, x3))
        x = self.drop(self.up3(x, x2))
        x = self.up4(x, x1)
        x = self.outc(x)
        return x             
    
class FC_Siam_Concat(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate=0, blur=False):
        super(FC_Siam_Concat, self).__init__()
        
        self.encoder = Encoder(in_channels, out_channels, dropout_rate, blur)
        
        self.up1 = up_concat(512*3, 256) ### upsample arguments 1 and 2
        self.up2 = up_concat(256*3, 128) ### upsample argument 1
        self.up3 = up_concat(128*3, 64)
        self.up4 = up_concat(64*3, 64)
        self.outc = outconv(64, out_channels)

        self.drop = nn.Dropout(p=dropout_rate)

    def forward(self, y, z):
   
        y5, y4, y3, y2, y1, y = self.encoder(y)
        z5, z4, z3, z2, z1, z = self.encoder(z)
  
        x = self.drop(self.up1(z5, y4, z4))
        x = self.drop(self.up2(x, y3, z3))
        x = self.drop(self.up3(x, y2, z2))
        x = self.up4(x, y1, z1)
        x = self.outc(x)
        return x          

class FC_Siam_Diff(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate=0, blur=False):
        super(FC_Siam_Diff, self).__init__()
        
        self.encoder = Encoder(in_channels, out_channels, dropout_rate, blur)
        
        self.up1 = up(1024, 256)
        self.up2 = up(512, 128)
        self.up3 = up(256, 64)
        self.up4 = up(128, 64)
        self.outc = outconv(64, out_channels)

        self.drop = nn.Dropout(p=dropout_rate)

    def forward(self, y, z):
        
        y5, y4, y3, y2, y1, y = self.encoder(y)
        z5, z4, z3, z2, z1, z = self.encoder(z)

        x1 = torch.abs(y1 - z1)
        x2 = torch.abs(y2 - z2)
        x3 = torch.abs(y3 - z3)
        x4 = torch.abs(y4 - z4)
        
        x = self.drop(self.up1(z5, x4))
        x = self.drop(self.up2(x, x3))
        x = self.drop(self.up3(x, x2))
        x = self.up4(x, x1)
        x = self.outc(x)
        return x      

class up_concat(nn.Module):
    ### upsample argument 1
    def __init__(self, in_ch, out_ch):
        super(up_concat, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2, x3):
        #x1 = self.up(x1)
        x1 = nn.functional.interpolate(x1, scale_factor=2, mode='nearest')
        #x2 = nn.functional.interpolate(x2, scale_factor=2, mode='nearest')
        # input is CHW
        diffY = x3.size()[2] - x1.size()[2]
        diffX = x3.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2))
        #x2 = F.pad(x2, (diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2))
        
        x = torch.cat([x3, x2, x1], dim=1)
        x = self.conv(x)
        return x
            