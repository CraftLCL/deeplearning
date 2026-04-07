# 实验4：AlexNet的网络结构及其实现代码

## 一、实验目的

1. 了解AlexNet的历史意义和在深度学习发展中的关键地位
2. 掌握AlexNet的网络结构及各层设计思想
3. 使用TensorFlow/Keras实现AlexNet
4. 理解Dropout正则化的作用

## 二、实验环境

- Python 3.8+
- TensorFlow 2.x（推荐 2.10+）
- NumPy、Matplotlib

## 三、实验原理

### 3.1 AlexNet简介

**ImageNet竞赛背景：** ImageNet大规模视觉识别挑战赛（ILSVRC）是计算机视觉领域最权威的竞赛之一。该数据集包含约**120万张**训练图像、5万张验证图像和15万张测试图像，涵盖**1000个类别**。在AlexNet出现之前，传统方法（如基于SIFT特征+SVM分类器）的Top-5错误率一直在**26%**左右徘徊，进步缓慢。

**历史性突破：** 2012年，Alex Krizhevsky、Ilya Sutskever和Geoffrey Hinton提出了AlexNet，在ILSVRC-2012竞赛中将Top-5错误率从26%骤降至**15.3%**，领先第二名（使用传统方法）超过10个百分点。这一结果震惊了整个计算机视觉界，标志着深度学习时代的正式开启。从此以后，ILSVRC的优胜方案全部基于深度神经网络。

**为什么AlexNet是革命性的？**
- 它**首次证明**了深度卷积神经网络在大规模图像分类任务中可以远远超越传统方法
- 它开创了使用**GPU进行深度学习训练**的先河（使用2块NVIDIA GTX 580 GPU并行训练）
- 它引入的多项技术（ReLU、Dropout、数据增强）至今仍是深度学习的标准组件
- 它直接推动了深度学习从学术界走向工业界的浪潮

AlexNet的关键创新：
- 使用**ReLU**激活函数（替代Sigmoid/Tanh），大幅加速训练
- 使用**Dropout**防止过拟合
- 使用**数据增强**扩充训练数据
- 利用**GPU**加速训练
- 使用**局部响应归一化**（LRN，后来被BatchNorm取代）

### 3.2 网络结构

| 层号 | 层类型 | 输出形状 | 说明 |
|------|--------|----------|------|
| 输入 | Input | 227×227×3 | RGB彩色图像 |
| C1 | Conv2D(96, 11×11, stride=4) + ReLU | 55×55×96 | 大卷积核提取低级特征 |
| S1 | MaxPooling(3×3, stride=2) | 27×27×96 | 下采样 |
| C2 | Conv2D(256, 5×5, same) + ReLU | 27×27×256 | 提取中级特征 |
| S2 | MaxPooling(3×3, stride=2) | 13×13×256 | 下采样 |
| C3 | Conv2D(384, 3×3, same) + ReLU | 13×13×384 | 提取高级特征 |
| C4 | Conv2D(384, 3×3, same) + ReLU | 13×13×384 | 进一步提取特征 |
| C5 | Conv2D(256, 3×3, same) + ReLU | 13×13×256 | 提取高级特征 |
| S5 | MaxPooling(3×3, stride=2) | 6×6×256 | 下采样 |
| F6 | Flatten + Dense(4096) + ReLU + Dropout(0.5) | 4096 | 全连接 |
| F7 | Dense(4096) + ReLU + Dropout(0.5) | 4096 | 全连接 |
| 输出 | Dense(N) + Softmax | N | N类分类 |

### 3.3 Dropout正则化

**过拟合问题：** 当模型参数量远大于训练数据量时，模型容易"记住"训练数据中的噪声和细节，而非学习通用规律。过拟合的典型表现是：**训练损失很低但测试损失很高**，训练准确率接近100%但测试准确率明显较低。AlexNet有约6000万个参数，极易过拟合，因此需要有效的正则化手段。

**Dropout的工作原理：**

- **训练阶段：** 在每次前向传播时，以概率 $p$ 随机将神经元的输出置为0（"丢弃"该神经元）。被丢弃的神经元不参与本次的前向传播和反向传播。这意味着每次训练使用的是网络的一个"子网络"，不同的训练步骤使用不同的子网络。
- **测试阶段：** 不进行任何丢弃，所有神经元都参与计算。但为了保持输出的期望值一致，需要将每个神经元的输出乘以 $(1-p)$（在Keras中，训练时会自动将保留神经元的输出除以 $(1-p)$ 进行缩放，因此测试时无需额外处理，这称为"inverted dropout"）。

**为什么 $p=0.5$ 最常用？** 当 $p=0.5$ 时，可能产生的子网络数量最多（$2^n$ 种组合，其中 $n$ 为神经元数），正则化效果最强。AlexNet正是在F6和F7两个全连接层后使用了 $Dropout(0.5)$。

**集成学习的解释：** Dropout可以理解为一种隐式的模型集成（Ensemble）。训练过程中，每个mini-batch实际上是在训练一个不同的子网络。最终测试时，相当于对所有这些子网络的预测结果取平均，从而获得更好的泛化能力。这类似于随机森林中多棵决策树投票的思想。

**Dropout的作用：** 通过随机丢弃神经元，迫使每个神经元不能依赖特定的其他神经元（防止"共适应"），从而学习到更加鲁棒、独立的特征表示。

### 3.4 ReLU激活函数

**传统激活函数的问题：** 在AlexNet之前，神经网络普遍使用Sigmoid函数 $\sigma(x) = \frac{1}{1+e^{-x}}$ 或Tanh函数 $\tanh(x)$。这些函数存在严重的**梯度消失问题**：当输入值较大或较小时，函数的梯度接近于0，导致深层网络中的参数几乎无法更新，训练极其缓慢甚至失败。

**ReLU的定义：**

$$f(x) = \max(0, x)$$

即：当输入为正时，直接输出该值；当输入为负时，输出0。

**ReLU相比Sigmoid/Tanh的优势：**

1. **无梯度消失问题：** 在正区间，ReLU的梯度恒为1，无论网络多深，梯度都能有效传播。而Sigmoid的最大梯度仅为0.25，经过多层连乘后梯度会指数衰减。
2. **计算速度快：** ReLU仅需一次比较运算（与0比较），而Sigmoid/Tanh需要指数运算，计算代价高得多。
3. **稀疏激活：** ReLU会使部分神经元输出为0，产生稀疏表示。研究表明稀疏表示更接近生物神经元的工作方式，且有助于特征解耦。
4. **收敛速度快：** 论文实验表明，使用ReLU的网络达到25%训练错误率的速度比Tanh快约**6倍**。

**ReLU的不足：** 当输入为负时，梯度为0，可能导致"神经元死亡"（Dead ReLU）。为此后续提出了Leaky ReLU、PReLU、ELU等改进版本。

### 3.5 数据增强

**为什么需要数据增强？** 深度神经网络有大量参数，需要海量数据来训练。当训练数据不足时，模型容易过拟合。数据增强通过对已有训练图像进行变换，生成"虚拟"的新训练样本，在不收集新数据的前提下有效扩充数据集。

**数据增强的核心思想：** 对图像施加不改变其语义标签的变换。例如，一张猫的图片水平翻转后仍然是猫。

**AlexNet中使用的数据增强技术：**

1. **随机裁剪（Random Crop）：** 从256x256的图像中随机裁剪出227x227的区域，每次裁剪位置不同，相当于增加了平移不变性
2. **水平翻转（Horizontal Flip）：** 以50%概率将图像左右翻转
3. **颜色抖动（Color Jitter）：** 对RGB通道进行PCA变换后加入随机扰动，增强模型对光照变化的鲁棒性

**其他常见的数据增强技术：**

- **旋转（Rotation）：** 随机小角度旋转图像
- **缩放（Scaling）：** 随机缩放图像大小
- **亮度/对比度调整：** 随机改变图像的亮度和对比度
- **Mixup/CutMix：** 将两张图像混合，标签也按比例混合（较新的方法）

### 3.6 各层参数量计算

卷积层参数量公式：$(K_h \times K_w \times C_{in} + 1) \times C_{out}$（其中+1为偏置项）

全连接层参数量公式：$(N_{in} + 1) \times N_{out}$

| 层 | 计算过程 | 参数量 |
|----|----------|--------|
| C1: Conv2D(96, 11x11) | $(11 \times 11 \times 3 + 1) \times 96$ | 34,944 |
| C2: Conv2D(256, 5x5) | $(5 \times 5 \times 96 + 1) \times 256$ | 614,656 |
| C3: Conv2D(384, 3x3) | $(3 \times 3 \times 256 + 1) \times 384$ | 885,120 |
| C4: Conv2D(384, 3x3) | $(3 \times 3 \times 384 + 1) \times 384$ | 1,327,488 |
| C5: Conv2D(256, 3x3) | $(3 \times 3 \times 384 + 1) \times 256$ | 884,992 |
| F6: Dense(4096) | $(6 \times 6 \times 256 + 1) \times 4096$ | 37,752,832 |
| F7: Dense(4096) | $(4096 + 1) \times 4096$ | 16,781,312 |
| 输出: Dense(1000) | $(4096 + 1) \times 1000$ | 4,097,000 |
| **合计** | | **约62,378,344 (~62M)** |

可以看出，AlexNet的参数主要集中在**全连接层**（F6和F7），占总参数量的约88%。这也是后续网络（如GoogLeNet）试图减少或替换全连接层的原因之一。

## 四、实验步骤

### 步骤1：数据准备

**代码解析**：

- **`ImageDataGenerator`**：Keras提供的图像数据生成器，用于对图像进行实时数据增强和预处理。它不会一次性将所有图像加载到内存中，而是在训练时按需（lazily）从磁盘读取，非常适合大规模数据集。
- **`rescale=1./255`**：将像素值从 [0, 255] 归一化到 [0, 1] 范围。神经网络更适合处理较小范围的数值，归一化可以加速训练收敛并提高数值稳定性。
- **`flow_from_directory()`**：从指定文件夹中读取图像。文件夹结构要求每个子文件夹代表一个类别（如 `data/train/cat/`、`data/train/dog/`），函数会自动根据子文件夹名分配标签。
- **`target_size=(IMSIZE, IMSIZE)`**：将所有图像统一缩放为227x227像素，以匹配AlexNet的输入尺寸要求。
- **`batch_size=200`**：每次从磁盘读取200张图像组成一个批次，用于一次梯度更新。
- **`class_mode='categorical'`**：使用one-hot编码的标签（如 [1,0] 表示猫，[0,1] 表示狗），适用于多分类任务的`categorical_crossentropy`损失函数。

```python
# [旧写法] from keras.preprocessing.image import ImageDataGenerator
# ↑ 独立 keras 包在 TF 2.0+ 中已不推荐使用

# [新写法] 适用于 TensorFlow >= 2.0
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMSIZE = 227

train_generator = ImageDataGenerator(rescale=1./255).flow_from_directory(
    'data/train/',
    target_size=(IMSIZE, IMSIZE),
    batch_size=200,
    class_mode='categorical')

validation_generator = ImageDataGenerator(rescale=1./255).flow_from_directory(
    'data/test/',
    target_size=(IMSIZE, IMSIZE),
    batch_size=200,
    class_mode='categorical')
```

### 步骤2：构建AlexNet模型

**代码解析**：

- **`Input([227, 227, 3])`**：定义输入张量的形状，227x227像素的RGB三通道图像。
- **`Conv2D(96, [11, 11], strides=[4, 4], activation='relu')`**：第一个卷积层使用96个大小为11x11的卷积核，步长为4。使用大卷积核是因为输入图像较大（227x227），需要快速提取大范围的低级特征（边缘、纹理等）。步长为4可以快速降低空间维度（从227降到55），减少后续计算量。参数量=$(11 \times 11 \times 3 + 1) \times 96 = 34,944$。
- **`MaxPooling2D([3, 3], strides=[2, 2])`**：最大池化层，使用3x3的池化窗口，步长为2，进一步下采样。最大池化保留最显著的特征，同时提供一定的平移不变性。
- **`Conv2D(256, [5, 5], padding="same")`**：第二个卷积层使用256个5x5卷积核，`padding="same"`表示在输入周围补零使输出空间尺寸与输入相同。
- **三个连续卷积层（C3-C5）不加池化层**：这是AlexNet的一个重要设计。连续堆叠多个卷积层可以在不损失空间分辨率的前提下提取更深层次的特征。C3(384个3x3) -> C4(384个3x3) -> C5(256个3x3)，逐步学习更抽象的高级语义特征。
- **`Flatten()`**：将三维特征图（6x6x256=9216）展平为一维向量，为全连接层做准备。
- **`Dense(4096, activation='relu')`**：全连接层，包含4096个神经元。全连接层负责将卷积层提取的局部特征组合成全局特征，学习特征之间的非线性组合。参数量巨大，F6层有 $9216 \times 4096 + 4096 = 37,752,832$ 个参数。
- **`Dropout(0.5)`**：以50%的概率随机丢弃神经元的输出，防止全连接层中神经元之间形成"共适应"关系，是AlexNet防止过拟合的关键手段。
- **`Dense(2, activation='softmax')`**：输出层，此处以2分类为例。Softmax函数将输出转换为概率分布，使各类别概率之和为1。

```python
# [旧写法] from keras.layers import Activation, Conv2D, BatchNormalization, Dense
# [旧写法] from keras.layers import Dropout, Flatten, Input, MaxPooling2D, ZeroPadding2D
# [旧写法] from keras import Model
# ↑ 独立 keras 包在 TF 2.0+ 中已不推荐使用

# [新写法] 适用于 TensorFlow >= 2.0
from tensorflow.keras.layers import Activation, Conv2D, BatchNormalization, Dense
from tensorflow.keras.layers import Dropout, Flatten, Input, MaxPooling2D, ZeroPadding2D
from tensorflow.keras import Model

IMSIZE = 227
input_layer = Input([IMSIZE, IMSIZE, 3])
x = input_layer

# 第1层：96个11×11卷积核，步长4
x = Conv2D(96, [11, 11], strides=[4, 4], activation='relu')(x)
x = MaxPooling2D([3, 3], strides=[2, 2])(x)

# 第2层：256个5×5卷积核
x = Conv2D(256, [5, 5], padding="same", activation='relu')(x)
x = MaxPooling2D([3, 3], strides=[2, 2])(x)

# 第3-5层：连续3个卷积层
x = Conv2D(384, [3, 3], padding="same", activation='relu')(x)
x = Conv2D(384, [3, 3], padding="same", activation='relu')(x)
x = Conv2D(256, [3, 3], padding="same", activation='relu')(x)
x = MaxPooling2D([3, 3], strides=[2, 2])(x)

# 全连接层
x = Flatten()(x)
x = Dense(4096, activation='relu')(x)
x = Dropout(0.5)(x)                # Dropout防止过拟合
x = Dense(4096, activation='relu')(x)
x = Dropout(0.5)(x)

# 输出层（此处以2分类为例）
x = Dense(2, activation='softmax')(x)
output_layer = x

model = Model(input_layer, output_layer)
model.summary()
```

### 步骤3：编译与训练

**代码解析**：

- **`Adam(learning_rate=0.001)`**：Adam（Adaptive Moment Estimation）优化器，结合了Momentum和RMSProp的优点。它为每个参数自适应地调整学习率，在大多数情况下比普通SGD收敛更快。`learning_rate=0.001` 是Adam的默认推荐值，是一个较为稳健的起始学习率。学习率过大会导致训练不稳定，过小则收敛缓慢。
- **`loss='categorical_crossentropy'`**：分类交叉熵损失函数，用于多分类任务（标签为one-hot编码）。交叉熵衡量预测概率分布与真实分布之间的差距，值越小表示预测越准确。公式为 $L = -\sum_{i} y_i \log(\hat{y}_i)$。
- **`metrics=['accuracy']`**：在训练过程中额外监控准确率指标，方便观察训练效果。
- **`model.fit()`**：执行模型训练。在TF 2.1之前需要使用 `model.fit_generator()` 来处理生成器数据，现在 `model.fit()` 可以直接接受生成器、数据集等多种数据格式。
- **`epochs=5`**：完整遍历训练数据5次。每个epoch后会在验证集上评估模型性能。
- **`validation_data=validation_generator`**：指定验证数据，每个epoch结束后在验证集上计算损失和准确率，用于监控过拟合情况。

```python
# [旧写法] from keras.optimizers import Adam
# [新写法] 适用于 TensorFlow >= 2.0
from tensorflow.keras.optimizers import Adam

# [旧写法] model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.001), metrics=['accuracy'])
# ↑ 参数 lr 在 TF 2.11+ 中已废弃
# [新写法] 适用于 TensorFlow >= 2.11
model.compile(
    loss='categorical_crossentropy',
    optimizer=Adam(learning_rate=0.001),
    metrics=['accuracy']
)

# [旧写法] model.fit_generator(train_generator, epochs=5, validation_data=validation_generator)
# ↑ fit_generator 在 TF 2.1 中已废弃，TF 2.20 中已移除
# [新写法] 适用于 TensorFlow >= 2.1
model.fit(train_generator, epochs=5, validation_data=validation_generator)
```

## 五、LeNet-5 vs AlexNet 对比

| 特征 | LeNet-5 (1998) | AlexNet (2012) |
|------|----------------|----------------|
| 输入大小 | 28×28×1 | 227×227×3 |
| 深度 | 5层 | 8层 |
| 激活函数 | Sigmoid/Tanh | ReLU |
| 正则化 | 无 | Dropout |
| 参数量 | ~6万 | ~6000万 |
| 训练方式 | CPU | GPU |

## 六、新旧写法对照表

| 功能 | 旧写法 | 新写法 |
|------|--------|--------|
| 导入层 | `from keras.layers import Conv2D, Dense, Dropout` | `from tensorflow.keras.layers import Conv2D, Dense, Dropout` |
| 导入模型 | `from keras import Model` | `from tensorflow.keras import Model` |
| 学习率 | `Adam(lr=0.001)` | `Adam(learning_rate=0.001)` |
| 训练 | `model.fit_generator(generator, ...)` | `model.fit(generator, ...)` |

## 七、思考题

1. AlexNet相比LeNet-5有哪些关键改进？这些改进为什么重要？
2. Dropout为什么能防止过拟合？训练和测试时的行为有什么不同？
3. AlexNet为什么使用11×11的大卷积核？后续网络为什么不再使用大卷积核？
4. 如果训练数据量很小，AlexNet可能会出现什么问题？如何解决？
