# 实验5：ResNet-50的网络结构及其实现代码

## 一、实验目的

1. 理解深层网络的退化问题
2. 掌握残差学习（Residual Learning）的核心思想
3. 理解残差块（Residual Block）的结构
4. 使用TensorFlow/Keras实现ResNet残差块及完整网络
5. 理解跳跃连接（Skip Connection）的作用

## 二、实验环境

- Python 3.8+
- TensorFlow 2.x（推荐 2.10+）
- NumPy、Matplotlib

## 三、实验原理

### 3.1 深层网络的退化问题

理论上，更深的网络应该有更强的表达能力。但实验发现，简单地堆叠卷积层，当网络深度超过一定值后，训练误差反而会增大——这就是**退化问题（Degradation Problem）**。

在何恺明等人的原始论文（*Deep Residual Learning for Image Recognition*, 2015）中给出了一个经典的实验证据：在CIFAR-10数据集上，一个56层的普通卷积网络的**训练误差**比20层网络更高。这一现象非常关键——它说明退化问题**不是过拟合**。如果是过拟合，56层网络的训练误差应该更低（训练集上表现好）、而测试误差更高。但事实是训练误差和测试误差都更高，说明更深的网络在优化过程中难以收敛到好的解。

从理论上讲，一个更深的网络至少不应该比浅层网络差——因为它可以让多余的层学习恒等映射（即什么都不做，直接把输入传到输出）。然而实际训练中，让网络通过标准的卷积层去学习恒等映射是非常困难的，这就是退化问题的根本原因。

### 3.2 残差学习

ResNet的核心思想是：不直接学习目标映射 H(x)，而是学习残差映射 F(x) = H(x) - x。

$$H(x) = F(x) + x$$

通过跳跃连接（Skip Connection）将输入直接加到输出上，即使F(x)难以学习，网络至少能保证恒等映射（F(x)=0时H(x)=x），不会比浅层网络更差。

**直觉理解：** 可以这样理解残差学习的思路——与其让网络从零开始学习一个完整的复杂映射 H(x)，不如让网络在输入 x 的基础上学习一个"小修正" F(x)。换句话说，网络只需要回答"需要对输入做什么调整？"而不是"最终输出是什么？"。学习小的修正量通常比学习完整的映射要容易得多。

**为什么恒等映射是一个好的默认值？** 在深层网络中，很多层实际上并不需要做太多变换。如果某一层最优的操作就是"什么都不做"（即恒等映射），那么残差网络只需要让 F(x) 学习到接近零即可，而零映射比恒等映射容易学习得多——将所有权重推向零比精确学习一个单位矩阵简单得多。这就是为什么跳跃连接能够显著改善深层网络的训练效果。

### 3.3 残差块结构

ResNet-50使用"瓶颈"（Bottleneck）残差块：

```
输入x (256通道)
  ├─────────────────────────┐
  │                         │ (跳跃连接/恒等映射)
  ▼                         │
Conv2D(64, 1×1) + BN + ReLU │ (降维)
  ▼                         │
Conv2D(64, 3×3) + BN + ReLU │ (提取特征)
  ▼                         │
Conv2D(256, 1×1) + BN       │ (升维)
  ▼                         │
  +  ◄──────────────────────┘ (逐元素相加)
  ▼
  ReLU
  ▼
输出 (256通道)
```

**瓶颈结构各层详解：**

1. **1×1卷积（降维）：256→64通道**
   作用是将输入的256个通道压缩到64个通道。这一步大幅减少了后续3×3卷积需要处理的数据量，是"瓶颈"名称的由来。1×1卷积本质上是对每个空间位置的通道维度做一次线性变换。

2. **3×3卷积（提取特征）：64→64通道**
   这是残差块中真正进行空间特征提取的层。由于输入通道数已从256降到64，3×3卷积的计算量大大减少。这一层负责捕获局部空间模式（如边缘、纹理等）。

3. **1×1卷积（升维）：64→256通道**
   将通道数恢复到256，以便与跳跃连接中的原始输入（256通道）维度匹配，从而进行逐元素相加。

**为什么瓶颈结构更高效？参数量对比：**

- **瓶颈结构（1×1→3×3→1×1）的参数量：**
  - 1×1卷积：256 × 64 × 1 × 1 = 16,384
  - 3×3卷积：64 × 64 × 3 × 3 = 36,864
  - 1×1卷积：64 × 256 × 1 × 1 = 16,384
  - **总计：69,632个参数**

- **如果使用两层3×3卷积（标准残差块）的参数量：**
  - 3×3卷积：256 × 256 × 3 × 3 = 589,824
  - 3×3卷积：256 × 256 × 3 × 3 = 589,824
  - **总计：1,179,648个参数**

瓶颈结构的参数量仅为标准结构的约 **5.9%**，计算效率提升近17倍！这使得在相同计算预算下可以构建更深的网络。

当输入和输出通道数不同时，需要在跳跃连接上添加1×1卷积进行通道对齐。

### 3.4 ResNet-50结构概览

| 层名 | 输出尺寸 | 结构 |
|------|----------|------|
| conv1 | 112×112 | 7×7, 64, stride 2 + MaxPool |
| conv2_x | 56×56 | [1×1,64; 3×3,64; 1×1,256] × 3 |
| conv3_x | 28×28 | [1×1,128; 3×3,128; 1×1,512] × 4 |
| conv4_x | 14×14 | [1×1,256; 3×3,256; 1×1,1024] × 6 |
| conv5_x | 7×7 | [1×1,512; 3×3,512; 1×1,2048] × 3 |
| | 1×1 | GlobalAvgPool + FC + Softmax |

**各阶段功能说明：**

- **conv1（Stem层）：** 使用大感受野的7×7卷积核快速降低空间分辨率（224→112），同时提取最基础的视觉特征（如边缘、颜色梯度）。MaxPool进一步将尺寸减半（112→56）。

- **conv2_x（低级特征提取）：** 包含3个瓶颈残差块。在56×56的分辨率上提取低级特征，如边缘、角点、简单纹理等基础视觉模式。通道数从64扩展到256。

- **conv3_x（中级特征提取）：** 包含4个瓶颈残差块。空间分辨率降至28×28，开始提取中级特征，如局部形状、复杂纹理组合等。通道数从256增加到512。

- **conv4_x（高级特征提取）：** 包含6个瓶颈残差块（是最厚的一个阶段）。在14×14的分辨率上提取高级特征，如物体的局部部件（眼睛、车轮等）。通道数增加到1024。

- **conv5_x（语义特征提取）：** 包含3个瓶颈残差块。空间分辨率最小（7×7），提取最高级的语义特征，如物体整体类别相关的抽象表示。通道数达到2048。

- **GlobalAvgPool + FC + Softmax：** 全局平均池化将每个通道的7×7特征图压缩为单个值，然后通过全连接层和Softmax进行最终分类。

从浅到深，网络逐步从"看到边缘"到"理解物体"，空间分辨率逐渐降低而通道数逐渐增多，这体现了深度学习中的一个核心设计原则：用更多的通道来补偿空间信息的损失。

### 3.5 BatchNormalization（批量归一化）

BatchNormalization（BN）是ResNet中不可或缺的组成部分，几乎出现在每一层卷积之后。

**BN的核心操作：** 对每个mini-batch的数据，在每个通道维度上进行归一化处理，使其均值为0、方差为1，然后通过可学习的参数 $\gamma$（缩放）和 $\beta$（偏移）进行线性变换：

$$\hat{x} = \frac{x - \mu_{batch}}{\sqrt{\sigma^2_{batch} + \epsilon}} \cdot \gamma + \beta$$

**BN的主要作用：**

1. **减少内部协变量偏移（Internal Covariate Shift）：** 在训练过程中，每一层的输入分布会随着前面层参数的更新而不断变化。BN通过归一化稳定了每一层的输入分布，使得各层可以独立地、更高效地学习。

2. **允许使用更大的学习率：** 没有BN时，过大的学习率容易导致梯度爆炸或训练不稳定。BN对输入进行了归一化，使得损失曲面更加平滑，因此可以安全地使用更大的学习率，加快收敛速度。

3. **正则化效果：** 由于BN使用mini-batch的统计量（均值和方差），每次计算带有随机噪声（不同batch的统计量略有不同），这在一定程度上起到了类似Dropout的正则化作用，有助于减少过拟合。

4. **缓解梯度消失：** 通过保持激活值在合理范围内，BN有效缓解了深层网络中常见的梯度消失问题。

## 四、实验步骤

### 步骤1：数据准备

```python
from matplotlib import pyplot as plt
# [旧写法] from keras.preprocessing.image import ImageDataGenerator
# [新写法] 适用于 TensorFlow >= 2.0
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMSIZE = 224
train_generator = ImageDataGenerator(rescale=1./255).flow_from_directory(
    'data/train/',
    target_size=(IMSIZE, IMSIZE),
    batch_size=100,
    class_mode='categorical')

validation_generator = ImageDataGenerator(rescale=1./255).flow_from_directory(
    'data/test/',
    target_size=(IMSIZE, IMSIZE),
    batch_size=100,
    class_mode='categorical')
```

### 步骤2：构建残差块

```python
# [旧写法] from keras.layers import Input, Activation, Conv2D, BatchNormalization, add, MaxPooling2D
# [新写法] 适用于 TensorFlow >= 2.0
from tensorflow.keras.layers import Input, Activation, Conv2D, BatchNormalization, add, MaxPooling2D

NB_CLASS = 3
IM_WIDTH = 224
IM_HEIGHT = 224

inpt = Input(shape=(IM_WIDTH, IM_HEIGHT, 3))

# 初始卷积层
x = Conv2D(64, (7, 7), padding='same', strides=(2, 2), activation='relu')(inpt)
x = BatchNormalization()(x)
x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)
x0 = x  # 保存输入，用于跳跃连接
```

**代码解析：**
- `Conv2D(64, (7, 7), padding='same', strides=(2, 2), activation='relu')`：使用64个7×7大小的卷积核，步长为2。大卷积核能在初始阶段捕获较大范围的空间信息。步长为2意味着每隔一个像素采样一次，因此输出空间尺寸减半：224 ÷ 2 = **112×112**。
- `BatchNormalization()`：对卷积输出进行批量归一化，稳定训练过程。
- `MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')`：3×3的最大池化，步长为2，进一步将空间尺寸减半：112 ÷ 2 = **56×56**。最大池化保留局部区域中最显著的特征。
- `x0 = x`：将当前的特征图保存到变量 x0 中。这是跳跃连接的关键——x0 将在后续残差块中与主路径的输出进行逐元素相加，实现残差学习。如果不保存这个引用，输入信息经过卷积层后就会丢失，跳跃连接也就无法实现。

### 步骤3：实现一个完整的残差块

```python
# 残差块内部：三层卷积
x = Conv2D(64, (1, 1), padding='same', strides=(1, 1), activation='relu')(x)
x = BatchNormalization()(x)

x = Conv2D(64, (3, 3), padding='same', strides=(1, 1), activation='relu')(x)
x = BatchNormalization()(x)

x = Conv2D(256, (1, 1), padding='same', strides=(1, 1), activation=None)(x)
x = BatchNormalization()(x)

# 跳跃连接：调整通道数（64→256）
x0 = Conv2D(256, (1, 1), padding='same', strides=(1, 1), activation='relu')(x0)
x0 = BatchNormalization()(x0)

# 逐元素相加 + ReLU
x = add([x, x0])      # 跳跃连接：将输入和残差相加
x = Activation('relu')(x)
x0 = x                 # 更新x0，作为下一个残差块的输入
```

**代码解析：**
- **第1层 `Conv2D(64, (1, 1), activation='relu')`：** 1×1卷积将通道数从64维持在64（在第一个块中通道数未变，但在后续使用256通道输入时起到降维作用）。1×1卷积不改变空间尺寸，仅对通道维度进行线性组合。ReLU激活函数引入非线性。
- **第2层 `Conv2D(64, (3, 3), activation='relu')`：** 3×3卷积是真正提取空间特征的层，`padding='same'`确保输出尺寸不变。在64通道上进行3×3卷积的计算量远小于在256通道上进行。
- **第3层 `Conv2D(256, (1, 1), activation=None)`：** 1×1卷积将通道数从64升到256。**这里 `activation=None` 非常关键**——这一层不使用激活函数，因为接下来要先与跳跃连接相加（`add`操作），相加之后才统一做ReLU激活。如果在相加之前就做了ReLU，负值信息会丢失，跳跃连接的效果会被削弱。
- **跳跃连接 `Conv2D(256, (1, 1))(x0)`：** 因为输入 x0 的通道数是64，而主路径输出的通道数是256，两者维度不匹配无法直接相加。所以需要通过1×1卷积将 x0 的通道数从64对齐到256。这在论文中称为"投影捷径"（Projection Shortcut）。
- **`add([x, x0])`：** 将主路径输出和跳跃连接进行**逐元素相加**（不是拼接concatenate）。这要求两者的形状完全相同（空间尺寸和通道数都一致）。逐元素相加实现了 H(x) = F(x) + x 的残差学习公式。
- **`x0 = x`：** 将当前残差块的输出保存为下一个残差块的跳跃连接输入，形成链式的残差学习结构。

### 步骤4：添加分类头并编译

```python
# [旧写法] from keras.models import Model
# [旧写法] from keras.layers import Dense, Flatten
# [新写法] 适用于 TensorFlow >= 2.0
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten

model = Model(inputs=inpt, outputs=x)

# 添加分类层
x = model.output
x = Flatten()(x)
predictions = Dense(NB_CLASS, activation='softmax')(x)
model_res = Model(inputs=model.input, outputs=predictions)
```

**代码解析：**
- **`Flatten()`：** 将残差块输出的三维特征图（高×宽×通道数，如56×56×256）展平为一维向量（56×56×256 = 802,816维）。这一步是从卷积特征提取过渡到全连接分类的桥梁，将空间维度信息转化为一个固定长度的特征向量。
- **`Dense(NB_CLASS, activation='softmax')`：** 全连接层将高维特征向量映射到类别数（NB_CLASS=3）。Softmax激活函数将输出转化为概率分布，每个输出值表示属于对应类别的概率，所有概率之和为1。
- **模型构建方式：** 先用 `Model(inputs=inpt, outputs=x)` 创建只包含残差块的子模型，然后在其输出上追加分类层，再用 `Model(inputs=model.input, outputs=predictions)` 创建完整模型。这种分步构建方式清晰地将特征提取部分和分类部分分开。

### 步骤5：编译与训练

```python
# [旧写法] from keras.optimizers import Adam
# [新写法] 适用于 TensorFlow >= 2.0
from tensorflow.keras.optimizers import Adam

# [旧写法] model_res.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.001), metrics=['accuracy'])
# [新写法] 适用于 TensorFlow >= 2.11
model_res.compile(
    loss='categorical_crossentropy',
    optimizer=Adam(learning_rate=0.001),
    metrics=['accuracy']
)

# [旧写法] model_res.fit_generator(train_generator, ...)
# [新写法] 适用于 TensorFlow >= 2.1
model_res.fit(
    train_generator,
    steps_per_epoch=100,
    epochs=5,
    validation_data=validation_generator,
    validation_steps=100
)
```

**代码解析：**
- **`loss='categorical_crossentropy'`：** 分类交叉熵损失函数，适用于多分类任务（标签为one-hot编码）。它衡量预测概率分布与真实标签分布之间的差异。
- **`optimizer=Adam(learning_rate=0.001)`：** Adam优化器结合了动量（Momentum）和自适应学习率（RMSProp）的优点，learning_rate=0.001是常用的默认学习率。
- **`metrics=['accuracy']`：** 在训练过程中同时监控分类准确率。
- **`steps_per_epoch=100`：** 每个epoch执行100个batch的训练。这个值通常设为 `训练样本总数 ÷ batch_size`。例如有10000个训练样本、batch_size=100，则 steps_per_epoch=100 正好遍历整个训练集一次。
- **`validation_steps=100`：** 每个epoch结束后，用100个batch的验证数据评估模型性能。设置原理同上。
- **`epochs=5`：** 整个训练集被遍历5次。实际项目中通常需要更多的epoch（如50-100），但为了快速演示这里使用较少的epoch。

## 五、新旧写法对照表

| 功能 | 旧写法 | 新写法 |
|------|--------|--------|
| 导入层 | `from keras.layers import add, Conv2D` | `from tensorflow.keras.layers import add, Conv2D` |
| 导入模型 | `from keras.models import Model` | `from tensorflow.keras.models import Model` |
| 学习率 | `Adam(lr=0.001)` | `Adam(learning_rate=0.001)` |
| 训练 | `model.fit_generator(...)` | `model.fit(...)` |

## 六、思考题

1. 残差学习为什么能解决深层网络的退化问题？
2. 跳跃连接中的1×1卷积有什么作用？什么时候需要？
3. ResNet-50中的"瓶颈"结构（1×1→3×3→1×1）有什么优势？
4. 如果去掉所有的跳跃连接，ResNet还能正常训练吗？
5. BatchNormalization放在ReLU之前还是之后更好？
