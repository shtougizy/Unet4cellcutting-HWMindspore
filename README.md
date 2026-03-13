# UNet 细胞分割实验

> 📚 **人工智能实训课程设计** — 哈尔滨工业大学（威海）2022级人工智能专业  
> 基于 MindSpore 框架对 UNet 网络进行一系列改进实验，涵盖损失函数、优化器、学习率调度、网络结构与数据增广五个方向。

---

## 项目简介

本项目以细胞图像分割任务为基础，在标准 UNet 网络上逐步引入多项改进，对比各项改进前后的训练损失与分割效果，探究不同技术选型对医学图像分割模型性能的影响。

UNet 采用编码器-解码器的 U 型结构：编码器由两个 3×3 卷积 + 2×2 最大池化迭代组成，每次下采样后通道数翻倍；解码器由反卷积、跳跃连接（skip connection）拼接和两个 3×3 卷积组成，最终经 1×1 卷积输出分割结果。

---

## 改进内容

### 1. 加入 Dice 损失函数

原始训练使用交叉熵损失，新增 `DiceLoss` 类替代或联合使用：

```python
class DiceLoss(nn.Cell):
    def __init__(self, smooth=1):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def construct(self, inputs, targets):
        inputs = ops.Sigmoid()(inputs)
        inputs = ops.Reshape()(inputs, (inputs.shape[0], -1))
        targets = ops.Reshape()(targets, (targets.shape[0], -1))
        intersection = ops.ReduceSum()(inputs * targets, 1)
        dice = (2. * intersection + self.smooth) / (
            ops.ReduceSum()(inputs, 1) + ops.ReduceSum()(targets, 1) + self.smooth
        )
        return 1 - dice.mean()
```

Dice Loss 直接度量预测掩码与真实掩码的重叠程度，对前景/背景样本不均衡的医学图像分割场景更为鲁棒。

---

### 2. 替换优化器并调整初始学习率

将原有 Adam 优化器替换为 SGD（含动量），并重新调整初始学习率：

```python
optimizer = nn.SGD(net.trainable_params(), learning_rate=lr, momentum=0.9)
```

动量设为 0.9，可加速收敛并抑制梯度震荡。也可选用 `nn.AdamWeightDecay` 作为备选优化器。

---

### 3. 引入 Cosine Annealing 学习率调度器

采用余弦退火策略，使学习率在训练过程中从最大值平滑衰减至 0，避免后期学习率过大导致的震荡：

```python
scheduler = nn.CosineDecayLR(
    min_lr=float(0),
    max_lr=float(lr),
    decay_steps=epochs
)
```

**实验效果**：同时引入改进 1～3 后，训练 Loss 相比基线明显降低。

---

### 4. 用卷积 + 上采样替换解码器中的转置卷积
在 `UNet` 初始化中通过 `use_deconv=False` 控制上采样方式，`UnetUp` 模块内部使用双线性插值上采样 + 普通卷积替代转置卷积，可有效缓解转置卷积的棋盘格伪影问题：

```python
self.up_concat1 = UnetUp(filters[1], filters[0], use_deconv=False, num_classes=2)
# use_deconv=False 时，UnetUp 内部使用 Upsample + Conv 替代 ConvTranspose2d
```

---

### 5. 为卷积模块加入残差连接

在 `UnetConv2d` 中加入 1×1 残差卷积分支，使输入可直接绕过卷积块叠加到输出，缓解梯度消失并增强特征复用：

```python
class UnetConv2d(nn.Cell):
    def __init__(self, in_channel, out_channel, use_bn=True, ...):
        ...
        self.residual_conv = nn.Conv2d(
            in_channel, out_channel, kernel_size=1, stride=1,
            pad_mode='same', weight_init="normal", bias_init="zeros"
        )

    def construct(self, inputs):
        residual = self.residual_conv(inputs)
        x = self.convs(inputs)
        x += residual
        return x
```

---

### 6. 数据增广

实现 `train_data_augmentation` 函数，对训练图像和对应掩码同步施加以下变换，提升模型泛化能力：

| 增广方式 | 参数范围 |
|----------|----------|
| 水平 / 垂直翻转 | 随机概率 0.5 |
| 随机裁剪 | 裁剪比例 30% |
| 亮度抖动 | 均匀分布 ±0.2 |
| 随机平移 | 最大 ±20 像素 |
| 随机旋转 | 最大 ±20° |
| 随机缩放 | 0.8 ～ 1.2 |
| 随机剪切 | 最大 ±10° |

所有几何变换通过 OpenCV `cv2.warpAffine` 实现，图像与掩码保持完全同步变换。

---

## 环境依赖

| 依赖 | 说明 |
|------|------|
| MindSpore | 深度学习框架 |
| OpenCV (`cv2`) | 数据增广中的仿射变换 |
| NumPy | 数值计算 |
| 华为云 ModelArts | 训练平台（昇腾处理器） |

---

## 实验结果对比

| 配置 | 说明 |
|------|------|
| 基线 | 原始 UNet + Adam + 固定学习率 |
| 改进 1～3 | Dice Loss + SGD + Cosine Annealing，Loss 明显下降 |
| 改进 4 | 卷积+上采样替代转置卷积，缓解棋盘格伪影 |
| 改进 5 | 残差连接，进一步改善收敛 |
| 改进 6 | 数据增广，提升模型泛化能力 |
