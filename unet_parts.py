

import mindspore.nn as nn
import mindspore.ops.operations as F
from mindspore.common.initializer import TruncatedNormal
from mindspore.nn import CentralCrop


# ── 改进5：DoubleConv 加入残差连接 ──────────────────────────────────────────
class DoubleConv(nn.Cell):

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        init_value_0 = TruncatedNormal(0.06)
        init_value_1 = TruncatedNormal(0.06)
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.SequentialCell(
            [nn.Conv2d(in_channels, mid_channels, kernel_size=3, has_bias=True,
                       weight_init=init_value_0, bias_init="zeros", pad_mode="valid"),
             nn.ReLU(),
             nn.Conv2d(mid_channels, out_channels, kernel_size=3, has_bias=True,
                       weight_init=init_value_1, bias_init="zeros", pad_mode="valid"),
             nn.ReLU()]
        )
        # 新增：1×1 残差卷积，用于匹配输入与输出的通道数
        self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1,
                                       stride=1, pad_mode='same',
                                       weight_init="normal", bias_init="zeros")

    def construct(self, x):
        residual = self.residual_conv(x)   # 残差分支：仅对齐通道，不改变空间尺寸
        out = self.double_conv(x)
        # 注意：valid padding 会使空间尺寸缩小，需对 residual 做中心裁剪对齐
        _, _, h, w = out.shape
        residual = residual[:, :, :h, :w]
        return out + residual


class Down(nn.Cell):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.SequentialCell(
            [nn.MaxPool2d(kernel_size=2, stride=2),
             DoubleConv(in_channels, out_channels)]
        )

    def construct(self, x):
        return self.maxpool_conv(x)


# ── 改进4：Up1～Up4 用双线性上采样 + 普通卷积替换转置卷积 ────────────────────

class Up1(nn.Cell):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        self.concat = F.Concat(axis=1)
        self.factor = 56.0 / 64.0
        self.center_crop = CentralCrop(central_fraction=self.factor)
        self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        # 替换：双线性插值上采样，scale_factor=2
        self.up = nn.ResizeBilinear()
        # 替换：普通 1×1 卷积调整通道数（原转置卷积同时完成升采样+减半通道）
        self.channel_conv = nn.Conv2d(in_channels, in_channels // 2, kernel_size=1,
                                      weight_init="normal", bias_init="zeros")
        self.relu = nn.ReLU()

    def construct(self, x1, x2):
        x1 = self.up(x1, scale_factor=2)   # 双线性插值 ×2
        x1 = self.channel_conv(x1)          # 通道数减半
        x1 = self.relu(x1)
        x2 = self.center_crop(x2)
        x = self.concat((x1, x2))
        return self.conv(x)


class Up2(nn.Cell):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        self.concat = F.Concat(axis=1)
        self.factor = 104.0 / 136.0
        self.center_crop = CentralCrop(central_fraction=self.factor)
        self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        self.up = nn.ResizeBilinear()
        self.channel_conv = nn.Conv2d(in_channels, in_channels // 2, kernel_size=1,
                                      weight_init="normal", bias_init="zeros")
        self.relu = nn.ReLU()

    def construct(self, x1, x2):
        x1 = self.up(x1, scale_factor=2)
        x1 = self.channel_conv(x1)
        x1 = self.relu(x1)
        x2 = self.center_crop(x2)
        x = self.concat((x1, x2))
        return self.conv(x)


class Up3(nn.Cell):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        self.concat = F.Concat(axis=1)
        self.factor = 200 / 280
        self.center_crop = CentralCrop(central_fraction=self.factor)
        self.print_fn = F.Print()
        self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        self.up = nn.ResizeBilinear()
        self.channel_conv = nn.Conv2d(in_channels, in_channels // 2, kernel_size=1,
                                      weight_init="normal", bias_init="zeros")
        self.relu = nn.ReLU()

    def construct(self, x1, x2):
        x1 = self.up(x1, scale_factor=2)
        x1 = self.channel_conv(x1)
        x1 = self.relu(x1)
        x2 = self.center_crop(x2)
        x = self.concat((x1, x2))
        return self.conv(x)


class Up4(nn.Cell):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        self.concat = F.Concat(axis=1)
        self.factor = 392 / 568
        self.center_crop = CentralCrop(central_fraction=self.factor)
        self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        self.up = nn.ResizeBilinear()
        self.channel_conv = nn.Conv2d(in_channels, in_channels // 2, kernel_size=1,
                                      weight_init="normal", bias_init="zeros")
        self.relu = nn.ReLU()

    def construct(self, x1, x2):
        x1 = self.up(x1, scale_factor=2)
        x1 = self.channel_conv(x1)
        x1 = self.relu(x1)
        x2 = self.center_crop(x2)
        x = self.concat((x1, x2))
        return self.conv(x)


class OutConv(nn.Cell):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        init_value = TruncatedNormal(0.06)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, has_bias=True,
                              weight_init=init_value, bias_init="zeros")

    def construct(self, x):
        return self.conv(x)
