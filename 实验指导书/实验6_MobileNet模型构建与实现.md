# 实验6：MobileNet模型构建及代码实现

## 一、实验目的

1. 理解轻量级网络的设计动机
2. 掌握深度可分离卷积（Depthwise Separable Convolution）的原理
3. 理解MobileNet的网络结构和宽度乘子（alpha）的作用
4. 使用TensorFlow/Keras实现MobileNet
5. 对比标准卷积与深度可分离卷积的参数量

## 二、实验环境

- Python 3.8+
- TensorFlow 2.x（推荐 2.10+）
- NumPy、Matplotlib

## 三、实验原理

### 3.1 轻量级网络的需求

传统CNN（如VGG、ResNet）参数量巨大，难以部署在移动设备和嵌入式设备上。MobileNet通过深度可分离卷积大幅减少计算量和参数量。

**实际应用背景**：随着深度学习从云端走向边缘，越来越多的AI任务需要在资源受限的设备上完成：
- **智能手机**：实时人脸识别、拍照场景分类、AR特效等，要求模型在手机CPU/GPU上达到实时推理（≥30 FPS）
- **IoT设备**：智能摄像头、无人机、机器人等嵌入式设备，内存通常只有几十MB，算力远低于服务器GPU
- **自动驾驶**：车载芯片需要在严格的功耗和延迟约束下运行目标检测模型

**计算量的衡量——FLOPs**：衡量模型计算开销的常用指标是FLOPs（Floating Point Operations，浮点运算次数）。例如，VGG-16约需15.5G FLOPs，而MobileNet（α=1.0）仅需约569M FLOPs，计算量减少了约**27倍**。FLOPs越低，推理速度越快、能耗越低，更适合部署在移动和嵌入式设备上。

### 3.2 标准卷积 vs 深度可分离卷积

#### 直观理解——分步拆解

可以将标准卷积到深度可分离卷积的转变理解为"分而治之"：

**第一步：标准卷积（Standard Convolution）**
```
输入特征图 (H×W×Cin)  →  一个卷积核 (K×K×Cin) 同时处理所有通道  →  输出一个通道
共需要 Cout 个这样的卷积核  →  输出 (H×W×Cout)
```
- 每个卷积核必须覆盖全部输入通道，**空间滤波和通道混合同时进行**，计算量大
- 参数量：$K \times K \times C_{in} \times C_{out}$

**第二步：逐深度卷积（Depthwise Convolution）——只做空间滤波**
```
输入特征图 (H×W×Cin)  →  每个通道各自拥有一个 K×K 卷积核  →  输出 (H×W×Cin)
```
- 每个通道**独立**进行空间卷积，通道之间互不干扰
- 优点：计算量极低；缺点：**没有跨通道信息交互**
- 参数量：$K \times K \times C_{in}$

**第三步：逐点卷积（Pointwise Convolution, 1x1卷积）——只做通道混合**
```
输入 (H×W×Cin)  →  1×1×Cin 的卷积核, 共 Cout 个  →  输出 (H×W×Cout)
```
- 在每个空间位置上对所有通道做线性组合，**完成跨通道信息融合**
- 参数量：$C_{in} \times C_{out}$

**总参数量**：$K \times K \times C_{in} + C_{in} \times C_{out}$

#### 计算量（FLOPs）对比

除了参数量，更重要的是计算量（FLOPs），它直接决定推理速度：

| | 计算公式 | 说明 |
|---|---|---|
| 标准卷积 FLOPs | $K^2 \times C_{in} \times C_{out} \times H \times W$ | 每个输出位置需要 $K^2 \times C_{in}$ 次乘加 |
| 逐深度卷积 FLOPs | $K^2 \times C_{in} \times H \times W$ | 每个通道独立做空间卷积 |
| 逐点卷积 FLOPs | $C_{in} \times C_{out} \times H \times W$ | 每个位置做通道线性组合 |
| 深度可分离总 FLOPs | $K^2 \times C_{in} \times H \times W + C_{in} \times C_{out} \times H \times W$ | 两步之和 |

**压缩比**：

$$\frac{\text{深度可分离 FLOPs}}{\text{标准卷积 FLOPs}} = \frac{K^2 \times C_{in} + C_{in} \times C_{out}}{K^2 \times C_{in} \times C_{out}} = \frac{1}{C_{out}} + \frac{1}{K^2}$$

对于常见的 $3 \times 3$ 卷积（$K=3$），压缩比约为 $\frac{1}{C_{out}} + \frac{1}{9}$。当输出通道数较大时（如$C_{out}=256$），$\frac{1}{C_{out}}$ 可以忽略，**计算量约减少为标准卷积的 1/9**。

#### 参数量数值对比

以3×3卷积，输入输出都是256通道为例：
- 标准卷积：$3 \times 3 \times 256 \times 256 = 589,824$
- 深度可分离：$3 \times 3 \times 256 + 256 \times 256 = 67,840$（**减少约8.7倍**）

### 3.3 MobileNet结构与ReLU6激活函数

每个深度可分离卷积块的结构：
```
输入
  ▼
DepthwiseConv2D(3×3) → BN → ReLU6
  ▼
Conv2D(1×1, pointwise) → BN → ReLU6
  ▼
输出
```

**ReLU6激活函数详解**：

ReLU6的数学公式为：

$$\text{ReLU6}(x) = \min(\max(0, x), 6)$$

即在标准ReLU的基础上，将输出值**截断在6**。对比如下：

| 激活函数 | 公式 | 输出范围 |
|---------|------|---------|
| ReLU | $\max(0, x)$ | $[0, +\infty)$ |
| ReLU6 | $\min(\max(0, x), 6)$ | $[0, 6]$ |

**为什么要截断在6？**
1. **量化友好（Quantization-Friendly）**：移动设备上常用8位定点数（INT8）或16位半精度浮点数（FP16）来加速推理。如果激活值范围无限大（如标准ReLU），量化时精度损失严重；而ReLU6将输出限制在[0, 6]的有限区间内，量化后数值分布更均匀，精度损失更小。
2. **数值稳定性**：有界的激活值使得梯度更加稳定，不容易出现数值溢出。
3. **实验验证**：Google的研究表明，截断值设为6在精度和数值稳定性之间取得了最佳平衡。

### 3.4 宽度乘子α与分辨率乘子ρ

**宽度乘子α（Width Multiplier）**：

宽度乘子α用于控制网络宽度，通过缩放每层的通道数来平衡精度和速度：
- α=1.0：标准MobileNet（完整通道数）
- α=0.75：75%通道，更快但精度略低
- α=0.5：50%通道，速度进一步提升

应用宽度乘子后，每层的通道数变为 $\lfloor \alpha \times C \rfloor$，计算量与 $\alpha^2$ 大致成正比。

**分辨率乘子ρ（Resolution Multiplier）**：

分辨率乘子ρ用于缩放输入图像的分辨率：
- ρ=1.0：输入224×224
- ρ=0.857：输入192×192
- ρ=0.714：输入160×160
- ρ=0.5：输入112×112

分辨率乘子不改变模型参数量，但计算量与 $\rho^2$ 成正比。本实验中 `IMSIZE=112` 即相当于使用了 ρ≈0.5。

**MobileNet v1 与 v2 的主要区别**（了解即可）：
- **MobileNet v1**（本实验实现）：使用深度可分离卷积替代标准卷积，结构简洁
- **MobileNet v2**：引入了**倒残差结构（Inverted Residual）**和**线性瓶颈（Linear Bottleneck）**，即先用1×1卷积扩展通道数，再做深度可分离卷积，最后用1×1卷积压缩通道，并在输入输出通道数相同时添加残差连接。v2在精度和效率上进一步优于v1

### 3.5 GlobalAveragePooling2D（全局平均池化）

全局平均池化（Global Average Pooling, GAP）是将每个特征图（feature map）取平均值，压缩为一个标量：

$$\text{GAP}(F_k) = \frac{1}{H \times W} \sum_{i=1}^{H} \sum_{j=1}^{W} F_k(i, j)$$

其中 $F_k$ 是第 $k$ 个通道的特征图。如果输入尺寸为 $H \times W \times C$，经过GAP后输出为 $1 \times C$ 的向量。

**GAP vs Flatten+Dense（全连接层）**：

| 对比维度 | Flatten + Dense | GlobalAveragePooling2D |
|---------|----------------|----------------------|
| 参数量 | 极大（如 7×7×1024=50176 个输入 → Dense(10) 需要 501,760 个参数） | **零参数**（仅做平均运算） |
| 过拟合风险 | 高（大量参数容易记忆训练数据） | 低（无可学习参数） |
| 输入尺寸依赖 | 是（Flatten后维度固定） | 否（任意输入尺寸均可） |
| 空间信息利用 | 粗暴展平，丢失空间结构 | 对每个通道做全局汇总，保留通道语义 |

在轻量级网络中，GAP是分类头的标准选择，既减少参数量，又能有效防止过拟合。

## 四、实验步骤

### 步骤1：定义深度可分离卷积块

```python
# [旧写法] from keras.layers import ZeroPadding2D, ReLU, DepthwiseConv2D
# [旧写法] from keras.layers import Conv2D, BatchNormalization
# ↑ 独立 keras 包在 TF 2.0+ 中已不推荐使用

# [新写法] 适用于 TensorFlow >= 2.0
from tensorflow.keras.layers import ZeroPadding2D, ReLU, DepthwiseConv2D
from tensorflow.keras.layers import Conv2D, BatchNormalization

def _depthwise_conv_block(inputs, pointwise_conv_filters, alpha,
                           strides=(1, 1), block_id=1):
    """
    深度可分离卷积块：
    DepthwiseConv2D(3×3) → BN → ReLU6 → Conv2D(1×1) → BN → ReLU6

    参数：
        inputs: 输入张量
        pointwise_conv_filters: 逐点卷积的输出通道数
        alpha: 宽度乘子
        strides: 步长
        block_id: 块编号（用于层命名）
    """
    pointwise_conv_filters = int(pointwise_conv_filters * alpha)

    if strides == (1, 1):
        x = inputs
    else:
        x = ZeroPadding2D(((0, 1), (0, 1)),
                          name='conv_pad_%d' % block_id)(inputs)

    # 逐深度卷积
    x = DepthwiseConv2D((3, 3),
                        padding='same' if strides == (1, 1) else 'valid',
                        strides=strides,
                        use_bias=False,
                        name='conv_dw_%d' % block_id)(x)
    x = BatchNormalization(axis=-1, name='conv_dw_%d_bn' % block_id)(x)
    x = ReLU(6., name='conv_dw_%d_relu' % block_id)(x)

    # 逐点卷积（1×1）
    x = Conv2D(pointwise_conv_filters, (1, 1),
               padding='same',
               use_bias=False,
               strides=(1, 1),
               name='conv_pw_%d' % block_id)(x)
    x = BatchNormalization(axis=-1, name='conv_pw_%d_bn' % block_id)(x)

    return ReLU(6., name='conv_pw_%d_relu' % block_id)(x)
```

**代码解析**：

- `pointwise_conv_filters = int(pointwise_conv_filters * alpha)`：将目标输出通道数乘以宽度乘子α，实现网络宽度的灵活缩放。例如，若 `pointwise_conv_filters=128, alpha=0.75`，则实际输出通道数为 `int(128×0.75)=96`。
- **ZeroPadding2D的条件使用**：当 `strides=(1,1)` 时不需要额外填充，直接使用 `padding='same'`；当 `strides=(2,2)` 时，手动在右侧和下侧各填充1个像素（`((0,1),(0,1))`），配合 `padding='valid'`，确保下采样时的空间对齐，使输出尺寸恰好为输入的一半。
- **DepthwiseConv2D((3,3))**：逐深度卷积层，对输入的**每个通道独立**使用一个3×3卷积核进行空间滤波。输入有多少个通道，就有多少个独立的卷积核，输出通道数与输入相同。
- **Conv2D(pointwise_conv_filters, (1,1))**：逐点卷积层（1×1卷积），在每个空间位置上对所有通道进行线性组合，实现**跨通道信息融合**，并将通道数从 $C_{in}$ 变换为 $C_{out}$。
- **use_bias=False**：卷积层不使用偏置项，因为紧随其后的BatchNormalization层自带偏置参数（β），再加偏置是冗余的。
- **BatchNormalization(axis=-1)**：沿通道轴做批归一化，加速训练收敛、稳定梯度。
- **ReLU(6.)**：ReLU6激活函数，将输出截断在[0, 6]区间，适合移动端量化部署。

### 步骤2：构建MobileNet模型

```python
# [旧写法] from keras.layers import Input, GlobalAveragePooling2D, Dense
# [旧写法] from keras import Model
# [新写法] 适用于 TensorFlow >= 2.0
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dense
from tensorflow.keras import Model

IMSIZE = 112
alpha = 1

# 输入层
input_layer = Input([IMSIZE, IMSIZE, 3])
x = input_layer

# 初始卷积层
x = ZeroPadding2D(padding=((0, 1), (0, 1)), name='conv1_pad')(x)
x = Conv2D(32, (3, 3), padding='valid', use_bias=False, strides=(2, 2), name='conv1')(x)
x = BatchNormalization(axis=-1, name='conv1_bn')(x)
x = ReLU(6, name='conv1_relu')(x)

# 深度可分离卷积块
x = _depthwise_conv_block(x, 64, alpha, block_id=1)
x = _depthwise_conv_block(x, 128, alpha, strides=(2, 2), block_id=2)
x = _depthwise_conv_block(x, 256, alpha, strides=(2, 2), block_id=3)
x = _depthwise_conv_block(x, 512, alpha, strides=(2, 2), block_id=4)
x = _depthwise_conv_block(x, 1024, alpha, strides=(2, 2), block_id=5)

# 全局平均池化 + 分类层
x = GlobalAveragePooling2D()(x)
x = Dense(10, activation='softmax')(x)

model = Model(inputs=input_layer, outputs=x)
model.summary()
```

**代码解析**：

- **输入层** `Input([112, 112, 3])`：定义输入为112×112的RGB三通道图像。
- **初始卷积层** `Conv2D(32, (3,3), strides=(2,2))`：使用32个3×3卷积核，步长为2进行下采样。输入112×112经过ZeroPadding后变为113×113，再经步长2的卷积输出为56×56×32。这一层使用标准卷积而非深度可分离卷积，因为输入通道数仅为3，深度可分离的优势不明显。
- **深度可分离卷积块的逐层分析**：
  - `block_id=1`：64个输出通道，`strides=(1,1)`，特征图尺寸不变 → 输出 **56×56×64**
  - `block_id=2`：128个输出通道，`strides=(2,2)`，下采样 → 输出 **28×28×128**
  - `block_id=3`：256个输出通道，`strides=(2,2)`，下采样 → 输出 **14×14×256**
  - `block_id=4`：512个输出通道，`strides=(2,2)`，下采样 → 输出 **7×7×512**
  - `block_id=5`：1024个输出通道，`strides=(2,2)`，下采样 → 输出 **4×4×1024**
- **GlobalAveragePooling2D()**：将4×4×1024的特征图做全局平均池化，每个通道取平均值，输出为长度1024的一维向量。无需学习任何参数，同时消除了对输入尺寸的依赖。
- **Dense(10, activation='softmax')**：全连接分类层，10个输出节点对应10个类别，softmax将输出转化为概率分布。

### 步骤3：编译与训练

```python
# [旧写法] from keras.optimizers import Adam
# [新写法] 适用于 TensorFlow >= 2.0
from tensorflow.keras.optimizers import Adam

model.compile(
    loss='categorical_crossentropy',
    # [旧写法] optimizer=Adam(lr=0.001),
    # [新写法] 适用于 TensorFlow >= 2.11
    optimizer=Adam(learning_rate=0.001),
    metrics=['accuracy']
)

# [旧写法] model.fit_generator(train_generator, steps_per_epoch=100, epochs=5, ...)
# [新写法] 适用于 TensorFlow >= 2.1
model.fit(
    train_generator,
    steps_per_epoch=100,
    epochs=5,
    validation_data=validation_generator,
    validation_steps=100
)
```

**代码解析**：

- **损失函数** `categorical_crossentropy`：多分类交叉熵损失，适用于多类别（互斥）分类任务。要求标签为one-hot编码格式（如 `[0,0,1,...,0]`）。公式为 $L = -\sum_{i} y_i \log(\hat{y}_i)$，其中 $y_i$ 是真实标签，$\hat{y}_i$ 是预测概率。
- **优化器** `Adam(learning_rate=0.001)`：Adam是自适应学习率优化器，结合了Momentum和RMSProp的优点，能自动调整每个参数的学习率。`learning_rate=0.001` 是默认值，适合大多数情况。
- **评估指标** `metrics=['accuracy']`：在训练过程中同时监控分类准确率。
- **model.fit()参数说明**：
  - `train_generator`：训练数据生成器，实时进行数据增强（旋转、平移、缩放等）
  - `steps_per_epoch=100`：每个epoch从生成器中取100个batch
  - `epochs=5`：训练5个完整周期
  - `validation_data`：验证数据生成器，用于评估模型在未见过数据上的表现
  - `validation_steps=100`：每次验证使用100个batch

## 五、新旧写法对照表

| 功能 | 旧写法 | 新写法 |
|------|--------|--------|
| 导入层 | `from keras.layers import DepthwiseConv2D, ReLU` | `from tensorflow.keras.layers import DepthwiseConv2D, ReLU` |
| 导入模型 | `from keras import Model` | `from tensorflow.keras import Model` |
| 学习率 | `Adam(lr=0.001)` | `Adam(learning_rate=0.001)` |
| 训练 | `model.fit_generator(...)` | `model.fit(...)` |

## 六、思考题

1. 深度可分离卷积为什么能大幅减少参数量？具体减少了多少倍？
2. ReLU6相比普通ReLU有什么优势？在什么场景下特别有用？
3. 宽度乘子α如何影响模型的精度和速度？
4. MobileNet适合哪些应用场景？它与ResNet相比有什么优劣？
5. 如何在精度和推理速度之间找到平衡？
