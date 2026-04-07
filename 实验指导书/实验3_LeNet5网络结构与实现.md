# 实验3：LeNet-5的网络结构及其实现代码

## 一、实验目的

1. 理解卷积神经网络（CNN）的基本概念
2. 掌握LeNet-5的网络结构及各层作用
3. 使用TensorFlow/Keras实现LeNet-5
4. 在MNIST数据集上训练并评估模型性能

## 二、实验环境

- Python 3.8+
- TensorFlow 2.x（推荐 2.10+）
- NumPy、Matplotlib

## 三、实验原理

### 3.1 LeNet-5简介

LeNet-5由Yann LeCun等人于1998年在论文《Gradient-Based Learning Applied to Document Recognition》中正式提出，是最早成功应用于实际场景的卷积神经网络之一。该网络专门为手写数字识别任务设计，在当时取得了突破性的成果。

**历史背景：** 在LeNet-5之前，LeCun团队已经从1989年开始探索卷积神经网络（LeNet-1、LeNet-4等早期版本）。LeNet-5是该系列网络的成熟版本，它被美国邮政服务（USPS）采用，用于自动识别信件上手写的邮政编码（ZIP codes），每天处理数百万封邮件。这是深度学习技术首次在大规模商业系统中成功部署的案例之一。

**核心贡献：** LeNet-5确立了现代CNN的基本架构范式——卷积层与池化层交替堆叠，后接全连接层进行分类。这一"特征提取+分类"的设计思想至今仍是深度学习图像分类模型的核心框架，后续的AlexNet、VGG、ResNet等经典网络均延续了这一思路。

### 3.2 网络结构

LeNet-5共7层（不含输入层）：

| 层号 | 层类型 | 输出形状 | 说明 |
|------|--------|----------|------|
| 输入 | Input | 28×28×1 | 灰度图像 |
| C1 | Conv2D(6, 5×5, same) + ReLU | 28×28×6 | 6个5×5卷积核，same padding |
| S2 | MaxPooling(2×2) | 14×14×6 | 下采样，尺寸减半 |
| C3 | Conv2D(16, 5×5, valid) + ReLU | 10×10×16 | 16个5×5卷积核，valid padding |
| S4 | MaxPooling(2×2) | 5×5×16 | 下采样，尺寸减半 |
| F5 | Flatten + Dense(120) + ReLU | 120 | 展平后全连接 |
| F6 | Dense(84) + ReLU | 84 | 全连接层 |
| 输出 | Dense(10) + Softmax | 10 | 10类分类输出 |

### 3.3 关键概念

- **卷积层（Conv2D）**：提取空间特征，通过卷积核在图像上滑动
- **池化层（MaxPooling）**：降低特征图尺寸，减少参数量，增强平移不变性
- **Same Padding**：输出尺寸与输入相同
- **Valid Padding**：不填充，输出尺寸缩小
- **Flatten**：将多维特征图展平为一维，衔接全连接层

**卷积输出尺寸计算公式：**

$$output = \lfloor \frac{input - kernel + 2 \times padding}{stride} \rfloor + 1$$

其中：
- `input`：输入特征图的宽或高
- `kernel`：卷积核的宽或高
- `padding`：填充的像素数（same padding时自动计算使输出=输入，valid padding时为0）
- `stride`：步长

**LeNet-5各层输出尺寸推导：**

| 层 | 计算过程 | 输出尺寸 | 参数量 |
|----|----------|----------|--------|
| 输入 | — | 28×28×1 | 0 |
| C1 (Conv2D, same) | same padding保持尺寸不变 | 28×28×6 | (5×5×1+1)×6 = **156** |
| S2 (MaxPool 2×2) | 28/2 = 14 | 14×14×6 | 0 |
| C3 (Conv2D, valid) | (14−5+2×0)/1+1 = 10 | 10×10×16 | (5×5×6+1)×16 = **2,416** |
| S4 (MaxPool 2×2) | 10/2 = 5 | 5×5×16 | 0 |
| Flatten | 5×5×16 = 400 | 400 | 0 |
| F5 (Dense 120) | 全连接 | 120 | 400×120+120 = **48,120** |
| F6 (Dense 84) | 全连接 | 84 | 120×84+84 = **10,164** |
| 输出 (Dense 10) | 全连接 | 10 | 84×10+10 = **850** |
| **总计** | | | **61,706** |

> **注意**：池化层没有可学习参数，它仅执行固定的取最大值操作。全连接层的参数量 = 输入维度 × 输出维度 + 偏置数（即输出维度）。

### 3.4 卷积运算详解

**卷积操作的基本过程：**

卷积运算是CNN的核心操作。它通过一个小的权重矩阵（称为**卷积核**或**滤波器**）在输入图像上滑动，在每个位置计算卷积核与对应区域的逐元素乘积之和，生成一个输出值。所有输出值组成的矩阵称为**特征图（Feature Map）**。

**示例：3×3卷积核在5×5输入上的运算（valid padding, stride=1）**

```
输入 (5×5):                卷积核 (3×3):
1  0  1  2  1             1  0  1
0  1  2  1  0             0  1  0
1  0  1  0  1             1  0  1
2  1  0  1  2
1  0  1  2  1

计算第一个输出值（左上角3×3区域）：
1×1 + 0×0 + 1×1 + 0×0 + 1×1 + 2×0 + 1×1 + 0×0 + 1×1 = 5

输出特征图 (3×3):    ← 输出尺寸 = (5-3)/1+1 = 3
5  ...  ...
.  ...  ...
.  ...  ...
```

卷积核从输入的左上角开始，逐步向右、向下滑动，每个位置计算一个输出值。

**特征图（Feature Map）：**

每个卷积核可以检测输入中的一种特定模式（如边缘、角点、纹理等）。一个卷积层通常包含多个卷积核，每个卷积核生成一张特征图。例如，LeNet-5的C1层有6个卷积核，因此输出6张特征图，分别捕捉输入图像中6种不同的局部特征。

**参数共享（Parameter Sharing）：**

参数共享是CNN的关键设计思想。同一个卷积核在整张图像上滑动时，使用的是**完全相同的一组权重参数**。这意味着无论目标特征出现在图像的哪个位置，同一个卷积核都能检测到它。参数共享带来两个重要好处：
1. **大幅减少参数量**：相比全连接层（每个连接都有独立权重），卷积层的参数量仅取决于卷积核大小和数量，与输入图像尺寸无关。
2. **平移等变性**：如果输入图像中的某个特征发生了平移，输出特征图中对应的响应也会发生相同的平移。

**局部连接（Local Connectivity）：**

在卷积层中，输出特征图上的每个值只与输入中一个**局部区域**（即卷积核覆盖的范围，也称为感受野）相连，而非与整个输入相连。这种局部连接的设计基于一个重要假设：图像中的有用特征通常是局部的（如边缘、角点），不需要看到整张图像才能检测。局部连接进一步减少了参数量，并使网络能够更高效地学习空间层次特征。

### 3.5 激活函数ReLU

**ReLU（Rectified Linear Unit，修正线性单元）：**

$$f(x) = \max(0, x)$$

ReLU是目前深度学习中最常用的激活函数，其规则极其简单：如果输入为正数，直接输出该值；如果输入为负数或零，输出为0。

**为什么使用ReLU而非Sigmoid？**

LeNet-5原始论文中使用的是Sigmoid/Tanh激活函数，但现代实现（包括本实验）通常使用ReLU，原因如下：

| 特性 | Sigmoid: σ(x)=1/(1+e⁻ˣ) | ReLU: f(x)=max(0,x) |
|------|--------------------------|----------------------|
| 输出范围 | (0, 1) | [0, +∞) |
| 梯度消失 | 严重：当x很大或很小时，梯度接近0 | 正半区无梯度消失问题 |
| 计算复杂度 | 需要指数运算 | 仅需比较和取值，计算极快 |
| 稀疏性 | 输出总是非零 | 约50%的神经元输出为0，产生稀疏表示 |

**梯度消失问题详解：** Sigmoid函数在输入值较大或较小时，其导数趋近于0。在深层网络中，反向传播时梯度逐层相乘，会导致靠近输入层的梯度变得极小（"消失"），使这些层几乎无法学习。ReLU在正半区的导数恒为1，有效缓解了这一问题。

## 四、实验步骤

### 步骤1：数据准备

```python
# [旧写法] from keras.datasets import mnist
# [旧写法] from keras.utils import np_utils
# ↑ 独立 keras 包及 np_utils 在 TF 2.0+ 中已不推荐/已移除

# [新写法] 适用于 TensorFlow >= 2.0
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

(X0, Y0), (X1, Y1) = mnist.load_data()
print(X0.shape)  # (60000, 28, 28)

# 可视化数字样本
from matplotlib import pyplot as plt
fig, ax = plt.subplots(2, 5)
ax = ax.flatten()
for i in range(10):
    Im = X0[Y0 == i][0]
    ax[i].imshow(Im)
plt.show()
```

**代码解析：**

- `mnist.load_data()`：Keras内置的数据加载函数，自动从网络下载MNIST数据集（首次运行时）并返回两个元组：`(X0, Y0)` 为训练集（60000张图像及其标签），`(X1, Y1)` 为测试集（10000张图像及其标签）。`X0` 的形状为 `(60000, 28, 28)`，表示60000张28×28像素的灰度图像；`Y0` 的形状为 `(60000,)`，每个元素是0-9的整数标签。
- `X0[Y0 == i]`：这是NumPy的**布尔索引（Boolean Indexing）**操作。`Y0 == i` 会生成一个与 `Y0` 等长的布尔数组（True/False），其中标签等于 `i` 的位置为True。将这个布尔数组作为 `X0` 的索引，就能筛选出所有标签为 `i` 的图像。`[0]` 取其中第一张，用于可视化展示。
- `plt.subplots(2, 5)`：创建一个2行5列共10个子图的画布，`ax.flatten()` 将二维的子图数组展平为一维，方便用索引逐个访问。
- `ax[i].imshow(Im)`：将28×28的数组以图像形式显示。`imshow` 默认使用伪彩色显示灰度图，可加参数 `cmap='gray'` 显示为灰度。

### 步骤2：数据预处理

```python
import numpy as np

N0 = X0.shape[0]
N1 = X1.shape[0]

# 重塑为 (N, 28, 28, 1) 并归一化
X0 = X0.reshape(N0, 28, 28, 1) / 255
X1 = X1.reshape(N1, 28, 28, 1) / 255

# One-Hot编码
# [旧写法] YY0 = np_utils.to_categorical(Y0)
# [新写法] 适用于 TensorFlow >= 2.0
YY0 = to_categorical(Y0)
YY1 = to_categorical(Y1)
```

**代码解析：**

- `X0.reshape(N0, 28, 28, 1)`：将原始形状 `(60000, 28, 28)` 重塑为 `(60000, 28, 28, 1)`。新增的最后一个维度 `1` 表示**通道数（channel）**。灰度图只有1个通道，彩色图（RGB）有3个通道。Keras的Conv2D层要求输入必须包含通道维度，格式为 `(batch, height, width, channels)`。
- `/ 255`：**归一化**操作，将像素值从 `[0, 255]` 的整数范围缩放到 `[0, 1]` 的浮点数范围。归一化的好处：(1) 使不同特征的数值范围一致，避免某些特征因数值过大而主导梯度更新；(2) 加速模型收敛，因为较小的输入值配合合适的学习率，梯度更新更加稳定。
- `to_categorical(Y0)`：将整数标签转换为**One-Hot编码**。例如标签 `3` 变为 `[0,0,0,1,0,0,0,0,0,0]`（长度为10的向量，仅第3个位置为1）。One-Hot编码是多分类任务的标准标签格式，配合Softmax输出层和交叉熵损失函数使用。

### 步骤3：构建LeNet-5模型

```python
# [旧写法] from keras.layers import Conv2D, Dense, Flatten, Input, MaxPooling2D
# [旧写法] from keras import Model
# ↑ 独立 keras 包在 TF 2.0+ 中已不推荐使用

# [新写法] 适用于 TensorFlow >= 2.0
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Input, MaxPooling2D
from tensorflow.keras import Model

input_layer = Input([28, 28, 1])
x = input_layer

# C1: 卷积层 —— 6个5×5卷积核，same padding，ReLU激活
x = Conv2D(6, [5, 5], padding="same", activation='relu')(x)     # 输出: 28×28×6

# S2: 池化层 —— 2×2最大值池化
x = MaxPooling2D(pool_size=[2, 2], strides=[2, 2])(x)           # 输出: 14×14×6

# C3: 卷积层 —— 16个5×5卷积核，valid padding，ReLU激活
x = Conv2D(16, [5, 5], padding="valid", activation='relu')(x)   # 输出: 10×10×16

# S4: 池化层 —— 2×2最大值池化
x = MaxPooling2D(pool_size=[2, 2], strides=[2, 2])(x)           # 输出: 5×5×16

# 展平
x = Flatten()(x)                                                 # 输出: 400

# F5: 全连接层
x = Dense(120, activation='relu')(x)                              # 输出: 120

# F6: 全连接层
x = Dense(84, activation='relu')(x)                               # 输出: 84

# 输出层
x = Dense(10, activation='softmax')(x)                            # 输出: 10

output_layer = x
model = Model(input_layer, output_layer)
model.summary()
```

**代码解析：**

- `Input([28, 28, 1])`：定义模型的输入层，指定输入张量的形状为28×28×1（高×宽×通道）。这里不包含batch维度，Keras会自动处理。
- **C1层 — `Conv2D(6, [5, 5], padding="same", activation='relu')`**：
  - `6`：使用6个卷积核，生成6张特征图
  - `[5, 5]`：每个卷积核的尺寸为5×5
  - `padding="same"`：在输入周围自动补零，使输出的空间尺寸与输入相同，即28×28
  - `activation='relu'`：对卷积结果应用ReLU激活函数
  - 输出形状：28×28×6
  - 参数量：每个卷积核有 5×5×1=25 个权重 + 1个偏置 = 26个参数，6个卷积核共 26×6 = **156个参数**
- **S2层 — `MaxPooling2D(pool_size=[2, 2], strides=[2, 2])`**：
  - 在每个2×2的窗口内取最大值，步长为2，因此输出尺寸减半：28/2 = 14
  - 输出形状：14×14×6
  - 参数量：**0**（池化操作无可学习参数）
- **C3层 — `Conv2D(16, [5, 5], padding="valid", activation='relu')`**：
  - `16`：使用16个卷积核
  - `padding="valid"`：不补零，输出尺寸计算为 (14−5+0)/1+1 = 10
  - 输出形状：10×10×16
  - 参数量：每个卷积核有 5×5×6=150 个权重（注意输入有6个通道）+ 1个偏置 = 151个参数，16个卷积核共 151×16 = **2,416个参数**
- **S4层 — `MaxPooling2D`**：同S2层，输出尺寸减半：10/2 = 5，输出形状：5×5×16，参数量：**0**
- **Flatten层**：将三维特征图 5×5×16 展平为一维向量，长度 = 5×5×16 = **400**
- **F5层 — `Dense(120, activation='relu')`**：全连接层，将400维输入映射到120维输出。参数量 = 400×120 + 120（偏置）= **48,120个参数**
- **F6层 — `Dense(84, activation='relu')`**：全连接层，120→84。参数量 = 120×84 + 84 = **10,164个参数**
- **输出层 — `Dense(10, activation='softmax')`**：输出层，84→10（对应10个数字类别）。`softmax` 将输出转换为概率分布，所有输出值之和为1。参数量 = 84×10 + 10 = **850个参数**
- `Model(input_layer, output_layer)`：使用Keras函数式API创建模型，指定输入和输出张量。
- `model.summary()`：打印模型结构摘要，包括每层的名称、输出形状和参数数量。

### 步骤4：编译与训练

```python
model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

model.fit(X0, YY0, epochs=10, batch_size=200, validation_data=[X1, YY1])
```

**代码解析：**

- `model.compile(...)`：编译模型，配置训练所需的三个关键组件：
  - `loss='categorical_crossentropy'`：**分类交叉熵损失函数**，用于多分类任务。它衡量模型预测的概率分布与真实标签（One-Hot编码）之间的差距。损失值越小，说明模型预测越接近真实标签。公式为：$L = -\sum_{i} y_i \log(\hat{y}_i)$，其中 $y_i$ 是真实标签，$\hat{y}_i$ 是预测概率。
  - `optimizer='adam'`：**Adam优化器**，是目前最常用的优化算法之一。它结合了动量（Momentum）和自适应学习率（RMSProp）的优点，能够自动调整每个参数的学习率，通常无需手动调参就能获得良好效果。
  - `metrics=['accuracy']`：在训练过程中额外计算**准确率（Accuracy）**，即正确分类的样本数占总样本数的比例。准确率不参与反向传播，仅用于监控训练进展。
- `model.fit(...)`：开始训练模型：
  - `X0, YY0`：训练数据和标签
  - `epochs=10`：完整遍历训练集10次。每个epoch结束后，模型会在验证集上评估一次
  - `batch_size=200`：每次从训练集中取200个样本计算梯度并更新权重。60000个样本 ÷ 200 = 300次更新/epoch
  - `validation_data=[X1, YY1]`：指定验证集，每个epoch结束后评估模型在验证集上的损失和准确率
  - **训练输出解读**：每个epoch会显示 `loss`（训练损失）、`accuracy`（训练准确率）、`val_loss`（验证损失）、`val_accuracy`（验证准确率）。理想情况下，随着训练进行，loss应逐渐下降，accuracy应逐渐上升——这表示模型正在学习。如果val_loss开始上升而val_accuracy下降，说明模型可能出现了**过拟合**。

## 五、新旧写法对照表

| 功能 | 旧写法 | 新写法 |
|------|--------|--------|
| 导入数据集 | `from keras.datasets import mnist` | `from tensorflow.keras.datasets import mnist` |
| 导入层 | `from keras.layers import Conv2D, Dense` | `from tensorflow.keras.layers import Conv2D, Dense` |
| 导入模型 | `from keras import Model` | `from tensorflow.keras import Model` |
| One-Hot | `from keras.utils import np_utils` | `from tensorflow.keras.utils import to_categorical` |

## 六、思考题

1. LeNet-5中使用same padding和valid padding的区别是什么？输出尺寸如何计算？
2. 池化层的作用是什么？最大池化和平均池化有什么区别？
3. 为什么LeNet-5能够显著优于逻辑回归模型？卷积层带来了什么优势？
4. 如果将LeNet-5中所有的ReLU换成Sigmoid，会有什么影响？
