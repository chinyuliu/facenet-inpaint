import torch
from torch import nn

class ConvBlock(nn.Module):
    """Reusable Conv + BN + ReLU block."""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, use_bn=True, use_relu=True):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation=dilation),
        ]
        if use_bn:
            layers.append(nn.BatchNorm2d(out_channels, momentum=0.8))
        if use_relu:
            layers.append(nn.ReLU(inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)
    
class Generator(nn.Module):
    def __init__(self, dc, in_channels=3, use_sigmoid=True):
        super().__init__()
        layers = []

        # Downsampling
        layers.append(ConvBlock(in_channels, 64, kernel_size=5, stride=1, padding=2))  # 256×256 × 64
        
        layers.append(ConvBlock(64, 128, kernel_size=3, stride=2, padding=1))         # 128×128 × 128
        layers.append(ConvBlock(128, 128, kernel_size=3, stride=1, padding=1))         # 128×128 × 128
        
        layers.append(ConvBlock(128, 256, kernel_size=3, stride=2, padding=1))         # 64×64 × 256
        layers.append(ConvBlock(256, 256, kernel_size=3, stride=1, padding=1))         # 64×64 × 256
        layers.append(ConvBlock(256, 256, kernel_size=3, stride=1, padding=1))

        # Dilated Convolutions
        layers.append(ConvBlock(256, 256, kernel_size=3, stride=1, padding=2, dilation=2))  # dilation=2
        layers.append(ConvBlock(256, 256, kernel_size=3, stride=1, padding=4, dilation=4))  # dilation=4
        layers.append(ConvBlock(256, 256, kernel_size=3, stride=1, padding=8, dilation=8))  # dilation=8
        layers.append(ConvBlock(256, 256, kernel_size=3, stride=1, padding=16, dilation=16))  # dilation=16
        for _ in range(2):  # 補充後 2 層 dilation=1
            layers.append(ConvBlock(256, 256, kernel_size=3, stride=1, padding=1))
            
        # Upsampling
        layers.append(nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1))  # 128×128 × 128
        layers.append(nn.BatchNorm2d(128, momentum=0.8))
        layers.append(nn.ReLU(inplace=True))
        layers.append(ConvBlock(128, 128, kernel_size=3, stride=1, padding=1))

        layers.append(nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1))   # 256×256 × 64
        layers.append(nn.BatchNorm2d(64, momentum=0.8))
        layers.append(nn.ReLU(inplace=True))
        layers.append(ConvBlock(64, 32, kernel_size=3, stride=1, padding=1))            # 256×256 × 32
        layers.append(ConvBlock(32, 3, kernel_size=3, stride=1, padding=1, use_bn=False, use_relu=False))  # 256×256 × 3

        if use_sigmoid:
            layers.append(nn.Sigmoid())

        self.model = nn.Sequential(*layers)
        self.last_conv = self._get_last_conv()
        self.model.apply(self._init_weights)

    def _get_last_conv(self):
        for layer in reversed(self.model):
            # 如果是原始的 Conv2d 或 ConvTranspose2d，就直接返回
            if isinstance(layer, (nn.Conv2d, nn.ConvTranspose2d)):
                return layer
            # 如果是 ConvBlock，進一步檢查裡面
            elif isinstance(layer, ConvBlock):
                for sublayer in reversed(layer.block):
                    if isinstance(sublayer, (nn.Conv2d, nn.ConvTranspose2d)):
                        return sublayer
        raise ValueError("模型中未找到 Conv2d 或 ConvTranspose2d 層")

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            if m is self.last_conv:
                nn.init.xavier_normal_(m.weight)
            else:
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.model(x)

class Glo_Discriminator(nn.Module):
    def __init__(self, dc, in_channels=3, use_sigmoid=True):
        super().__init__()
        layers = []

        # Downsampling
        for i in range(6):  # 5-----------------------------------------------------
            out_channels = min(dc * (2 ** i), 512)  # 64 128 256 512
            layers.append(ConvBlock(in_channels, out_channels, kernel_size=5, stride=2, padding=2))
            in_channels = out_channels

        # Output layer
        layers.append(ConvBlock(in_channels, 1, kernel_size=1, stride=1, padding=0, use_bn=False, use_relu=False))
        if use_sigmoid:
            layers.append(nn.Sigmoid())

        self.model = nn.Sequential(*layers)
        self.last_conv = self._get_last_conv()
        # 權重初始化
        self.model.apply(self._init_weights)

    def _get_last_conv(self):
        for layer in reversed(self.model):
            # 如果是原始的 Conv2d 或 ConvTranspose2d，就直接返回
            if isinstance(layer, (nn.Conv2d, nn.ConvTranspose2d)):
                return layer
            # 如果是 ConvBlock，進一步檢查裡面
            elif isinstance(layer, ConvBlock):
                for sublayer in reversed(layer.block):
                    if isinstance(sublayer, (nn.Conv2d, nn.ConvTranspose2d)):
                        return sublayer
        raise ValueError("模型中未找到 Conv2d 或 ConvTranspose2d 層")


    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            if m is self.last_conv:
                nn.init.xavier_normal_(m.weight)
            else:
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.model(x)
    
class Loc_Discriminator(nn.Module):
    def __init__(self, dc, in_channels=3, use_sigmoid=True):
        super().__init__()
        layers = []

        # Downsampling
        for i in range(5):  # 3-----------------------------------------------------
            out_channels = min(dc * (2 ** i), 512)  # 64 128 256 512
            layers.append(ConvBlock(in_channels, out_channels, kernel_size=5, stride=2, padding=2))
            in_channels = out_channels

        # Output layer
        layers.append(ConvBlock(in_channels, 1, kernel_size=1, stride=1, padding=0, use_bn=False, use_relu=False))
        if use_sigmoid:
            layers.append(nn.Sigmoid())

        self.model = nn.Sequential(*layers)
        self.last_conv = self._get_last_conv()  # 正確指定最後一層
        self.model.apply(self._init_weights)    # 初始化參數

    def _get_last_conv(self):
        for layer in reversed(self.model):
            # 如果是原始的 Conv2d 或 ConvTranspose2d，就直接返回
            if isinstance(layer, (nn.Conv2d, nn.ConvTranspose2d)):
                return layer
            # 如果是 ConvBlock，進一步檢查裡面
            elif isinstance(layer, ConvBlock):
                for sublayer in reversed(layer.block):
                    if isinstance(sublayer, (nn.Conv2d, nn.ConvTranspose2d)):
                        return sublayer
        raise ValueError("模型中未找到 Conv2d 或 ConvTranspose2d 層")

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            if m is self.last_conv:
                nn.init.xavier_normal_(m.weight)
            else:
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.model(x)
