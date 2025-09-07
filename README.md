# Unet4cellcutting-HWMindspore 技术文档
CREATE IN 2024.7:Optimize unet model for cell cutting,including:Adding Dice Loss;Using Convolution + Upsampling instead of Transposed Convolution;Adding Residual Connections in Convolution;Adding Data Augmentation.

UNet最初由Ronneberger等人在2015年提出，特别适用于生物医学图像的分割任务，其主要特点是结合了卷积神经网络（CNN）的强大特征提取能力和对称性结构设计，能够在准确分割图像的同时保持较高的计算效率。
UNet的结构
UNet的结构可以分为两部分：编码器和解码器。
1. 编码器（Encoder）：编码器部分由一系列卷积层和池化层组成，逐层提取图像的空间特征。每一层卷积操作后紧接着一个ReLU激活函数和一个最大池化操作。这部分的作用是逐渐减少特征图的空间维度，同时增加特征图的深度，从而捕捉图像的上下文信息。
2. 解码器（Decoder）：解码器部分与编码器结构对称，由一系列上采样层和卷积层组成。通过上采样操作逐步恢复图像的空间维度。每一个上采样操作之后紧接着一个卷积操作，以细化特征图。同时，编码器中的特征图在解码器的对应层通过跳跃连接（skip connections）直接与上采样后的特征图进行拼接（concatenate），这有助于恢复图像的细节信息。

UNet在MindSpore中的实现
在MindSpore中实现UNet模型时，主要步骤包括：
1. 定义模型结构：使用MindSpore的API定义UNet的编码器和解码器部分。编码器部分包括若干卷积层和最大池化层，解码器部分包括上采样层和卷积层。
2. 跳跃连接：在编码器和解码器之间建立跳跃连接，将编码器各层的输出特征图与解码器对应层的特征图拼接，以帮助恢复更多细节信息。
3. 损失函数和优化器：选择适当的损失函数（如交叉熵损失）和优化器（如Adam），用于训练UNet模型。
4. 训练和推理：通过MindSpore的训练框架，加载数据集，进行模型训练和评估。训练完成后，使用训练好的模型进行图像分割任务的推理。
应用场景
UNet在图像分割任务中表现出色，广泛应用于医学图像分析（如肿瘤分割、器官分割）、遥感影像处理（如土地覆盖分类）、自然图像处理等领域。其通过对图像的像素级别进行分类，实现对目标区域的精准分割。
总结来说，UNet是一个结构简单但功能强大的图像分割模型，结合了编码器和解码器的优势，通过跳跃连接有效融合低层次和高层次特征信息。在MindSpore中，UNet的实现和应用能够高效完成各种图像分割任务，满足不同领域的需求。
原始测试结果：
 

1.为Unet训练加入Dice损失
代码：

	class DiceLoss(nn.Cell):
	    def __init__(self, smooth=1):
	        super(DiceLoss, self).__init__()
	        self.smooth = smooth

    def construct(self, inputs, targets):
        inputs = ops.Sigmoid()(inputs)
        inputs = ops.Reshape()(inputs, (inputs.shape[0], -1))
        targets = ops.Reshape()(targets, (targets.shape[0], -1))
        
        intersection = ops.ReduceSum()(inputs * targets, 1)
        dice = (2. * intersection + self.smooth) / (ops.ReduceSum()(inputs, 1) + ops.ReduceSum()(targets, 1) + self.smooth)
        
        return 1 - dice.mean()
这部分代码用于计算Dice损失。Dice损失是一种常用的损失函数，特别适用于二值图像分割任务。它通过度量预测结果和真实标签之间的重叠程度来衡量模型的分割效果，损失值越小，分割效果越好。
	
	class DiceLoss(nn.Cell):
	    def __init__(self, smooth=1):
	        super(DiceLoss, self).__init__()
	        self.smooth = smooth

__init__方法：初始化类的实例。在初始化过程中，设置一个平滑因子smooth，默认值为1。平滑因子用于防止分母为零，从而稳定训练过程。

super(DiceLoss, self).__init__()：调用父类nn.Cell的初始化方法，确保父类的初始化逻辑也被执行。

	def construct(self, inputs, targets):
	inputs = ops.Sigmoid()(inputs)
	inputs = ops.Reshape()(inputs, (inputs.shape[0], -1))
	targets = ops.Reshape()(targets, (targets.shape[0], -1))

construct方法：定义前向计算逻辑，即如何计算Dice损失。
inputs = ops.Sigmoid()(inputs)：将预测的输入通过Sigmoid激活函数，压缩到[0, 1]范围内。这一步是因为在二值分割任务中，网络的输出通常是概率值。
inputs = ops.Reshape()(inputs, (inputs.shape[0], -1))：将inputs重塑为二维张量，其中第一维是批次大小，第二维是展平后的像素数。这样可以方便后续计算。
targets = ops.Reshape()(targets, (targets.shape[0], -1))：将目标标签targets同样重塑为二维张量，方便与inputs进行逐元素操作。
	
	intersection = ops.ReduceSum()(inputs * targets, 1)
	dice = (2. * intersection + self.smooth) / (ops.ReduceSum()(inputs, 1) + ops.ReduceSum()(targets, 1) + self.smooth)

intersection = ops.ReduceSum()(inputs * targets, 1)：计算预测结果和真实标签的交集。具体来说，是对逐元素相乘后的结果在每个样本的维度上求和。
dice = (2. * intersection + self.smooth) / (ops.ReduceSum()(inputs, 1) + ops.ReduceSum()(targets, 1) + self.smooth)：计算Dice系数。Dice系数的公式为：2 * 交集 + 平滑因子 / (预测总和 + 标签总和 + 平滑因子)。这个公式衡量了预测结果和真实标签的重叠程度。

	return 1 - dice.mean()

return 1 - dice.mean()：返回Dice损失值。由于Dice系数是一个相似度度量，值越大表示预测与真实标签越接近，因此Dice损失值定义为1 - Dice系数，值越小表示分割效果越好。最后返回所有样本的平均Dice损失。

总结
上述代码实现了Dice损失函数，它通过度量预测结果与真实标签之间的重叠程度，来评估图像分割模型的性能。这个实现包含了关键的前向计算逻辑，包括应用Sigmoid函数、重塑张量、计算交集以及最终的Dice损失值。在实际使用中，这个Dice损失函数可以作为训练图像分割模型的一部分，帮助模型学习更准确的分割结果。

2.使用SGD或AdamWeightDecay或=替代原有Adam优化器并调整初始学习率

代码：

optimizer=nn.SGD(net.trainable_params(),learning_rate=lr, momentum=0.9)

参数说明：
optimizer: 优化器的实例。
nn.SGD: 这是指定使用随机梯度下降（SGD）优化算法。
net.trainable_params(): 这是获取神经网络模型中所有可训练的参数。
learning_rate=lr: 这是设置学习率，其中lr是一个预先定义好的学习率值。
momentum=0.9: 这是设置动量参数为0.9，动量的引入可以加速收敛并减少梯度下降过程中可能的震荡。
通过上述代码，SGD优化器替换了原有的Adam优化器，并且设置了一个新的初始学习率和动量参数，完成了对优化器和学习率的调整。



3. 引入CosineAnnealing学习率调度器
代码：

scheduler=nn.CosineDecayLR(min_lr=float(0),max_lr=float(lr), decay_steps=epochs)

参数说明：
scheduler: 学习率调度器的实例。
	nn.CosineDecayLR: 使用余弦退火（Cosine Annealing）策略来调整学习率。
	min_lr=float(0): 最小学习率设置为 0。
	max_lr=float(lr): 最大学习率设置为 lr，其中 lr 是一个预先定义好的学习率值。
	decay_steps=epochs: 学习率衰减的步数，通常设置为训练的总 epoch 数。



前三步同时实现后的实验结果，损失率明显降低。
 




4.使用卷积+上采样替换Unet解码器中的转置卷积

	class UNet(nn.Cell):
	    def __init__(self, in_channel, n_class=2, feature_scale=2, use_deconv=False, use_bn=True):
	        super(UNet, self).__init__()
	        self.in_channel = in_channel
	        self.n_class = n_class
	        self.feature_scale = feature_scale
	        self.use_deconv = use_deconv
	        self.use_bn = use_bn

	        filters = [64, 128, 256, 512, 1024]
	        filters = [int(x / self.feature_scale) for x in filters]
	
	        # Down Sample
	        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, pad_mode="same")
	        self.conv0 = UnetConv2d(self.in_channel, filters[0], self.use_bn)
	        self.conv1 = UnetConv2d(filters[0], filters[1], self.use_bn)
	        self.conv2 = UnetConv2d(filters[1], filters[2], self.use_bn)
	        self.conv3 = UnetConv2d(filters[2], filters[3], self.use_bn)
	        self.conv4 = UnetConv2d(filters[3], filters[4], self.use_bn)
	
	        # Up Sample
	        self.up_concat1 = UnetUp(filters[1], filters[0], self.use_deconv, 2)
	        self.up_concat2 = UnetUp(filters[2], filters[1], self.use_deconv, 2)
	        self.up_concat3 = UnetUp(filters[3], filters[2], self.use_deconv, 2)
	        self.up_concat4 = UnetUp(filters[4], filters[3], self.use_deconv, 2)
	
	        # Finale Convolution
	        self.final = nn.Conv2d(filters[0], n_class, 1, weight_init="normal", bias_init="zeros")
	
	    def construct(self, inputs):
	        x0 = self.conv0(inputs)                   # channel = filters[0]
	        x1 = self.conv1(self.maxpool(x0))        # channel = filters[1]
	        x2 = self.conv2(self.maxpool(x1))        # channel = filters[2]
	        x3 = self.conv3(self.maxpool(x2))        # channel = filters[3]
	        x4 = self.conv4(self.maxpool(x3))        # channel = filters[4]
	        up4 = self.up_concat4(x4, x3)
	        up3 = self.up_concat3(up4, x2)
	        up2 = self.up_concat2(up3, x1)
	        up1 = self.up_concat1(up2, x0)
	
	        final = self.final(up1)
	        return final


代码解释：

1.	初始化方法 
	参数定义: 定义了模型的输入通道数 in_channel、类别数 n_class、特征缩放比例 feature_scale、是否使用反卷积 use_deconv 和是否使用批量归一化 use_bn。
	特征图通道数计算: 定义了 filters 列表，包含了 UNet 各层的特征图通道数，并根据 feature_scale 进行缩放。
2.	Down Sample（下采样）:
	使用了 nn.MaxPool2d 进行四次下采样，每次下采样后都通过 UnetConv2d 进行特征提取和通道数调整。
3.	Up Sample（上采样）:
	使用自定义的 UnetUp 类进行四次上采样，每次上采样将上一层的特征图与对应的下采样结果进行连接（skip connection）。
4.	Finale Convolution（最终卷积）:
	使用 nn.Conv2d 进行最终的卷积操作，将上采样后的结果映射到最终的输出类别数 n_class。
5.	构造方法 (construct):
	执行了整个 UNet 的前向计算过程。
	从输入开始，经过下采样和上采样操作，最终生成分割结果。
实验结果： 

5.为Unet中的卷积加入残差链接
class UnetConv2d(nn.Cell):
  
    def __init__(self, in_channel, out_channel, use_bn=True, num_layer=2, kernel_size=3, stride=1, padding='same'):
        super(UnetConv2d, self).__init__()
        self.num_layer = num_layer
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.in_channel = in_channel
        self.out_channel = out_channel
        convs = []
        for _ in range(num_layer):
            convs.append(conv_bn_relu(in_channel, out_channel, use_bn, kernel_size, stride, padding, "relu"))
            in_channel = out_channel
        self.convs = nn.SequentialCell(convs)
        # 添加残差卷积层以匹配通道数
        self.residual_conv=nn.Conv2d(self.in_channel,self.out_channel,kernel_size=1, stride=1, pad_mode='same',weight_init="normal", bias_init="zeros")

    def construct(self, inputs):
        residual = self.residual_conv(inputs)  # 计算残差以匹配通道数
        x = self.convs(inputs)
        x += residual  # 将卷积层输出与残差相加
        return x   

在我编辑的代码中，为了在Unet中的卷积层加入残差链接，主要进行了以下几步操作：

1. 初始化残差卷积层：
   在UnetConv2d的__init__方法中，添加了一个用于计算残差的卷积层self.residual_conv。该卷积层的作用是将输入张量的通道数转换为与卷积操作后输出的通道数相匹配。具体来说，这个残差卷积层使用了1x1卷积核，以确保残差路径上的特征图与主路径上的特征图具有相同的维度。

2. 构造方法中的残差计算：
   在construct方法中，首先通过self.residual_conv(inputs)计算残差residual。这个残差的作用是调整输入张量的通道数，使得其与卷积操作后的输出张量具有相同的通道数。

3. 将残差与卷积操作后的输出相加：
   在计算完卷积操作后的输出x之后，将其与残差residual相加，实现残差链接。这样，通过残差链接，可以使模型更容易进行梯度传播，从而加快模型的收敛速度，并且能够缓解梯度消失问题。

以下是代码实现的详细步骤：

1. 初始化残差卷积层：
   
	self.residual_conv = nn.Conv2d(self.in_channel, self.out_channel, kernel_size=1, stride=1, pad_mode='same', weight_init="normal", bias_init="zeros")

   这里的self.in_channel和self.out_channel分别是输入和输出的通道数，通过1x1卷积进行通道匹配。

2. 构造方法中的残差计算：
   
    residual = self.residual_conv(inputs)

   这一步将输入inputs通过1x1卷积层self.residual_conv得到残差residual。

3. 将残差与卷积操作后的输出相加：
	
	   x = self.convs(inputs)
	   x += residual

   先通过self.convs(inputs)得到卷积操作后的输出x，然后将x与残差residual相加，得到最终输出。

这样，通过残差链接，卷积层的输出不仅包含了卷积操作后的信息，还保留了一部分输入信息，这种结构在深层网络中可以有效缓解梯度消失问题，提升模型性能。

测试结果：
 
4.加入随机仿射变换、随机亮度等数据增广
	代码：
 
	def train_data_augmentation(img, mask):
	    # Horizontal flip
	    if np.random.random() > 0.5:
	        img = np.flipud(img)
	        mask = np.flipud(mask)
	   
	    # Vertical flip
	    if np.random.random() > 0.5:
	        img = np.fliplr(img)
	        mask = np.fliplr(mask)
	    # Random cropping
	    crop_fraction = 0.3
	    h, w = img.shape[:2]
	    top = np.random.randint(0, int(crop_fraction * h))
	    bottom = np.random.randint(int((1 - crop_fraction) * h), h)
	    left = np.random.randint(0, int(crop_fraction * w))
	    right = np.random.randint(int((1 - crop_fraction) * w), w)
	    img = img[top:bottom, left:right]
	    mask = mask[top:bottom, left:right]
	    # Adjust brightness
	    brightness = np.random.uniform(-0.2, 0.2)
	    img = np.float32(img + brightness)
	    img = np.clip(img, -1.0, 1.0)
	    
	    # Affine transformation
	    rows, cols = img.shape[:2]
	
	    # 随机平移
	    max_trans = 20  # maximum translation
	    tx = np.random.uniform(-max_trans, max_trans)
	    ty = np.random.uniform(-max_trans, max_trans)
	
	    # 随机旋转
	    max_rot = 20  # maximum rotation angle
	    angle = np.random.uniform(-max_rot, max_rot)
	
	    # 随即缩放
	    scale = np.random.uniform(0.8, 1.2)
	
	    # 随机剪切
	    max_shear = 10  # maximum shear angle
	    shear = np.random.uniform(-max_shear, max_shear)
	
	    M_translation = np.float32([[1, 0, tx], [0, 1, ty]])
	    img = cv2.warpAffine(img, M_translation, (cols, rows))
	    mask = cv2.warpAffine(mask, M_translation, (cols, rows))
	
	    center = (cols / 2, rows / 2)
	    M_rotation = cv2.getRotationMatrix2D(center, angle, scale)
	    img = cv2.warpAffine(img, M_rotation, (cols, rows))
	    mask = cv2.warpAffine(mask, M_rotation, (cols, rows))
	
	    M_shear = np.float32([[1, np.tan(np.radians(shear)), 0], [0, 1, 0]])
	    img = cv2.warpAffine(img, M_shear, (cols, rows))
	    mask = cv2.warpAffine(mask, M_shear, (cols, rows))
	
	    return img, mask
在代码中，通过实现数据增强来促进模型训练的多样性，从而提高模型的泛化能力。
以下是各个步骤的详细解释：

水平翻转和垂直翻转：
	
	if np.random.random() > 0.5:
	    img = np.flipud(img)
	    mask = np.flipud(mask)
	
	if np.random.random() > 0.5:
	    img = np.fliplr(img)
	    mask = np.fliplr(mask)
np.flipud和np.fliplr分别用于上下翻转和左右翻转。通过随机概率确定是否进行这些操作。

随机裁剪：
	
	crop_fraction = 0.3
	h, w = img.shape[:2]
	top = np.random.randint(0, int(crop_fraction * h))
	bottom = np.random.randint(int((1 - crop_fraction) * h), h)
	left = np.random.randint(0, int(crop_fraction * w))
	right = np.random.randint(int((1 - crop_fraction) * w), w)
	img = img[top:bottom, left:right]
	mask = mask[top:bottom, left:right]
随机选择裁剪的起始和结束位置，确保图像和掩码保持相同的裁剪区域。

亮度调整：
	
	brightness = np.random.uniform(-0.2, 0.2)
	img = np.float32(img + brightness)
	img = np.clip(img, -1.0, 1.0)
调整图像亮度，确保结果在合法范围内。

平移变换：
	
	max_trans = 20
	tx = np.random.uniform(-max_trans, max_trans)
	ty = np.random.uniform(-max_trans, max_trans)
	
	M_translation = np.float32([[1, 0, tx], [0, 1, ty]])
	img = cv2.warpAffine(img, M_translation, (cols, rows))
	mask = cv2.warpAffine(mask, M_translation, (cols, rows))
创建一个平移矩阵，并用cv2.warpAffine函数对图像和掩码进行平移。

旋转和缩放：
	
	max_rot = 20
	angle = np.random.uniform(-max_rot, max_rot)
	
	scale = np.random.uniform(0.8, 1.2)
	
	center = (cols / 2, rows / 2)
	M_rotation = cv2.getRotationMatrix2D(center, angle, scale)
	img = cv2.warpAffine(img, M_rotation, (cols, rows))
	mask = cv2.warpAffine(mask, M_rotation, (cols, rows))
创建旋转和缩放矩阵，并用cv2.warpAffine对图像和掩码进行变换。

剪切变换：
	
	max_shear = 10
	shear = np.random.uniform(-max_shear, max_shear)
	
	M_shear = np.float32([[1, np.tan(np.radians(shear)), 0], [0, 1, 0]])
	img = cv2.warpAffine(img, M_shear, (cols, rows))
	mask = cv2.warpAffine(mask, M_shear, (cols, rows))

创建剪切矩阵，并用cv2.warpAffine进行变换。

总结：
这段代码对UNet训练的意义在于通过数据增强（Data Augmentation）提高模型的泛化能力和鲁棒性。具体来说，数据增强的作用和意义包括以下几个方面：

1. 增加训练数据的多样性：
   通过对训练图像和对应的掩码进行多种随机变换，如翻转、裁剪、亮度调整、平移、旋转、缩放和剪切，可以生成许多不同的训练样本。这实际上扩大了训练数据集的规模，使模型在训练过程中能看到更多样化的数据，从而提高模型的泛化能力。

2. 防止过拟合：
   由于数据增强引入了各种变换，使得每次训练看到的图像都有所不同，模型难以记住特定的训练样本细节，从而减小了过拟合的风险。模型将更倾向于学习数据的本质特征，而不是记住特定样本的细节。

3. 提高模型的鲁棒性：
   随机变换如亮度调整、平移、旋转和缩放等模拟了实际应用中可能遇到的图像变换。通过在训练过程中加入这些变换，模型能够更好地适应不同的实际情况，从而在面对噪声、光照变化、图像偏移等情况时表现得更加鲁棒。

4. 更好地利用有限的数据：
   在医学图像处理等领域，标注数据往往有限。数据增强可以有效地扩展有限的数据集，使得模型在有限数据上也能获得较好的性能。

总结来说，数据增强通过模拟多种图像变换，显著提高了UNet模型的泛化能力、鲁棒性和对有限数据的利用效率，从而在实际应用中获得更好的表现。
