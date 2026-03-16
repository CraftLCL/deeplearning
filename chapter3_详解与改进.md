# Chapter 3 详细解析与模型改进建议

## 一、章节总览

Chapter 3 的核心内容是 **数据读取与简单模型搭建**，涵盖以下主题：

| 部分 | 代码示例 | 内容 |
|------|---------|------|
| 数据表示基础 | 3-1 ~ 3-4 | 标量、向量、矩阵、三维数组 |
| 图片数据处理 | 3-5 ~ 3-10 | 图片读取、缩放、数组化、归一化、像素变换 |
| CSV 表格数据 | 3-11 ~ 3-13 | Pandas 读取 CSV、提取标签 |
| 批量图片读取 | 3-14 ~ 3-15 | 基于 CSV 索引批量加载食物图片 |
| 食物评分回归模型 | 3-16 ~ 3-20 | 数据划分、构建简单全连接回归模型、训练与预测 |
| MNIST 分类 | 3-21 ~ 3-31 | 读取公开数据集、one-hot 编码、构建 softmax 分类器、可视化权重 |

---

## 二、逐部分详解

### 2.1 数据表示基础（代码 3-1 ~ 3-4）

这部分通过 NumPy 演示了不同维度的数据：

- **标量（0维）**：`np.array(888)` — 一个单独的数值，`ndim = 0`
- **向量（1维）**：`np.array([1,2,3,4,5])` — 一组有序数值，`ndim = 1`
- **矩阵（2维）**：3×4 的数组，`ndim = 2`
- **三维数组**：3×2×5 的数组，`ndim = 3`

**为什么重要？**  
深度学习中所有数据（图片、文本、音频）最终都要转换为多维数组（张量）进行计算。理解维度概念是后续理解模型输入输出的基础。

---

### 2.2 图片数据处理（代码 3-5 ~ 3-10）

**处理流程：**

```
原始图片 → 读取(PIL) → 缩放(resize) → 转数组(np.array) → 归一化(/255) → 可视化
```

**关键概念：**

1. **统一尺寸**：不同图片分辨率不同，但神经网络要求输入维度固定，所以必须 `resize` 到统一大小（如 128×128）。
2. **数组化**：图片转为 NumPy 数组后，形状为 `(高, 宽, 通道数)`。RGB 彩色图有 3 个通道。
3. **归一化**：像素值从 [0, 255] 缩放到 [0, 1]，使数值范围更小，有助于梯度计算稳定性和训练收敛速度。
4. **像素变换示例**：
   - `Im + 0.5` → 整体变亮
   - `1 - Im` → 反色
   - `0.5 * Im` → 整体变暗
   - `Im / 0.5` → 亮度增强

---

### 2.3 CSV 数据读取（代码 3-11 ~ 3-13）

使用 Pandas 读取 `FoodScore.csv`，包含食物图片的 ID 和评分（score）。

- `MasterFile['ID']`：提取图片文件名列表
- `MasterFile['score']`：提取食物吸引力评分
- `reshape([N,1])`：将一维标签变为列向量，符合模型输入要求

注释中的 `Y=(Y-np.mean(Y))/np.std(Y)` 是标准化处理（Z-score），此处未启用，但在实际训练中通常会提升效果。

---

### 2.4 批量图片加载（代码 3-14 ~ 3-15）

根据 CSV 中的文件名，循环读取所有图片并存入 4D 数组 `X`：
- 形状为 `(N, 128, 128, 3)`，即 N 张 128×128 的 RGB 图片
- 每张图片都经过 resize 和归一化处理
- 显示前 10 张图片及其对应评分，验证数据对齐

---

### 2.5 食物评分回归模型（代码 3-16 ~ 3-20）

#### 模型结构

```
输入 Input(128, 128, 3)
  ↓
Flatten()          # 将 128×128×3 = 49152 维展平
  ↓
Dense(1)           # 全连接层，输出 1 个值（回归预测）
  ↓
输出：食物评分预测值
```

#### 训练配置

| 参数 | 值 | 说明 |
|------|-----|------|
| 损失函数 | MSE（均方误差） | 回归任务标准损失 |
| 优化器 | Adam | 自适应学习率优化器 |
| 学习率 | 0.001 | 默认值 |
| batch_size | 100 | 每批 100 个样本 |
| epochs | 100 | 训练 100 轮 |
| 数据划分 | 50/50 train/test | 使用 train_test_split |

#### 模型本质

这实际上是一个 **线性回归模型**。Flatten 层只是把图片展平，Dense(1) 没有激活函数，所以整个模型等价于：

$$\hat{y} = \mathbf{w}^T \mathbf{x} + b$$

其中 $\mathbf{x}$ 是 49152 维的像素向量，$\mathbf{w}$ 是对应的权重向量。

---

### 2.6 MNIST 手写数字分类（代码 3-21 ~ 3-31）

#### 数据集

- 训练集：55000 张 28×28 灰度图（展平为 784 维向量）
- 验证集：5000 张
- 测试集：10000 张
- 标签：0~9，经 one-hot 编码后变为 10 维向量

#### 模型结构

```
输入 Input(784)
  ↓
Dense(10)          # 全连接层，输出 10 个值
  ↓
Softmax            # 将输出转为概率分布
  ↓
输出：10 个类别的概率
```

#### 训练配置

| 参数 | 值 | 说明 |
|------|-----|------|
| 损失函数 | categorical_crossentropy | 多分类标准损失 |
| 优化器 | Adam | 学习率 0.01 |
| batch_size | 1000 | 每批 1000 个样本 |
| epochs | 10 | 训练 10 轮 |

#### 模型本质

这是一个 **Softmax 回归（多类逻辑回归）** 模型：

$$P(y=k|\mathbf{x}) = \frac{e^{\mathbf{w}_k^T \mathbf{x} + b_k}}{\sum_{j=0}^{9} e^{\mathbf{w}_j^T \mathbf{x} + b_j}}$$

没有隐藏层，是最简单的分类器之一。

#### 权重可视化（代码 3-31）

将 Dense 层的权重矩阵（784×10）中每一列 reshape 为 28×28 图像，可以看到模型"学到"的每个数字的模板。这直观展示了线性分类器的本质——通过模板匹配来分类。

---

## 三、模型局限性分析

### 3.1 食物评分回归模型的问题

| 问题 | 说明 |
|------|------|
| **模型过于简单** | 单层线性模型无法学习图像中的复杂特征（边缘、纹理、形状等） |
| **无特征提取** | 直接将原始像素输入全连接层，没有利用图像的空间结构 |
| **参数量过大** | Flatten 后有 49152 个输入，单层就有约 5 万参数，但全是线性的 |
| **数据划分不合理** | 50/50 的划分比例浪费了训练数据，通常建议 80/20 或 70/30 |
| **缺少数据增强** | 食物图片数据量较少时，没有使用任何数据增强策略 |

### 3.2 MNIST 分类模型的问题

| 问题 | 说明 |
|------|------|
| **无隐藏层** | Softmax 回归只能学习线性决策边界，准确率上限约 92-93% |
| **忽略空间信息** | 将 2D 图像展平为 1D 向量，丢失了像素间的位置关系 |
| **batch_size 过大** | 1000 的 batch_size 对 MNIST 来说偏大，可能影响泛化 |

---

## 四、更好的训练方法与改进建议

### 4.1 食物评分回归模型的改进

#### 改进方案一：多层全连接网络（MLP）

增加隐藏层和非线性激活函数：

```python
from keras.layers import Dense, Flatten, Input, Dropout, BatchNormalization
from keras import Model

input_layer = Input([IMSIZE, IMSIZE, 3])
x = Flatten()(input_layer)
x = Dense(256, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.3)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)
x = Dense(1)(x)
output_layer = x

model = Model(input_layer, output_layer)
```

**改进点：**
- 多层结构可以学习非线性特征
- `BatchNormalization` 加速训练收敛
- `Dropout` 防止过拟合

#### 改进方案二：卷积神经网络（CNN）⭐ 推荐

CNN 是图像任务的标准方法，能有效提取空间特征：

```python
from keras.layers import (Conv2D, MaxPooling2D, Dense, Flatten,
                          Input, Dropout, BatchNormalization)
from keras import Model

input_layer = Input([IMSIZE, IMSIZE, 3])
x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_layer)
x = MaxPooling2D((2, 2))(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2))(x)
x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2))(x)
x = Flatten()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(1)(x)
output_layer = x

model = Model(input_layer, output_layer)
```

**优势：**
- 卷积层利用局部感受野提取边缘、纹理等特征
- 池化层降低维度，减少参数
- 参数共享大幅减少参数量
- 预期 MSE 显著降低

#### 改进方案三：迁移学习 ⭐⭐ 强烈推荐

在数据量有限时，使用预训练模型效果最好：

```python
from keras.applications import MobileNetV2
from keras.layers import Dense, GlobalAveragePooling2D, Input
from keras import Model

base_model = MobileNetV2(weights='imagenet',
                         include_top=False,
                         input_shape=(IMSIZE, IMSIZE, 3))
base_model.trainable = False  # 冻结预训练权重

input_layer = Input([IMSIZE, IMSIZE, 3])
x = base_model(input_layer, training=False)
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = Dense(1)(x)
output_layer = x

model = Model(input_layer, output_layer)
```

**优势：**
- 利用 ImageNet 上预训练的视觉特征，适合小数据集
- `GlobalAveragePooling2D` 替代 Flatten，大幅减少参数
- 先冻结训练顶部，再微调（fine-tune）效果更好

#### 其他训练技巧

```python
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

# 数据划分改为 80/20
X0, X1, Y0, Y1 = train_test_split(X, Y, test_size=0.2, random_state=42)

# 标签标准化
Y_mean, Y_std = Y0.mean(), Y0.std()
Y0_norm = (Y0 - Y_mean) / Y_std
Y1_norm = (Y1 - Y_mean) / Y_std

model.compile(loss='mse', optimizer=Adam(learning_rate=0.001))

# 回调函数
callbacks = [
    EarlyStopping(patience=10, restore_best_weights=True),  # 早停
    ReduceLROnPlateau(factor=0.5, patience=5),               # 学习率衰减
]

model.fit(X0, Y0_norm,
          validation_data=(X1, Y1_norm),
          batch_size=32,      # 更小的 batch_size
          epochs=200,
          callbacks=callbacks)
```

**改进点：**
- **标签标准化**：回归任务中对标签做 Z-score 标准化通常能加速收敛
- **EarlyStopping**：防止过拟合，自动停止训练
- **ReduceLROnPlateau**：验证集Loss不再下降时自动降低学习率
- **更小的 batch_size**（32）：增加梯度噪声，有助于泛化
- **80/20 划分**：保留更多训练数据

---

### 4.2 MNIST 分类模型的改进

#### 改进方案一：多层感知机（MLP）

```python
from keras.layers import Activation, Dense, Input, Dropout

input_layer = Input((784,))
x = Dense(512, activation='relu')(input_layer)
x = Dropout(0.2)(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.2)(x)
x = Dense(10)(x)
x = Activation('softmax')(x)
output_layer = x

model = Model(input_layer, output_layer)
```

**预期准确率**：~98%（相比原始 Softmax 回归的 ~92%）

#### 改进方案二：卷积神经网络 ⭐ 推荐

```python
from keras.layers import (Conv2D, MaxPooling2D, Dense, Flatten,
                          Input, Dropout, Activation, Reshape)
from keras import Model

input_layer = Input((784,))
x = Reshape((28, 28, 1))(input_layer)       # 恢复为 2D 图像
x = Conv2D(32, (3, 3), activation='relu')(x)
x = MaxPooling2D((2, 2))(x)
x = Conv2D(64, (3, 3), activation='relu')(x)
x = MaxPooling2D((2, 2))(x)
x = Flatten()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(10)(x)
x = Activation('softmax')(x)
output_layer = x

model = Model(input_layer, output_layer)
```

**预期准确率**：~99.2%

#### 训练改进

```python
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping

model.compile(optimizer=Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X0, YY0,
          validation_data=(X1, YY1),
          batch_size=128,        # 比原来的 1000 更合理
          epochs=50,
          callbacks=[EarlyStopping(patience=5, restore_best_weights=True)])
```

---

### 4.3 数据增强（两个模型通用）

对于图像任务，数据增强是提升泛化能力的关键手段：

```python
from keras.preprocessing.image import ImageDataGenerator
# 或在新版 Keras 中使用 keras.layers 中的数据增强层

datagen = ImageDataGenerator(
    rotation_range=15,       # 随机旋转 ±15°
    width_shift_range=0.1,   # 水平平移 10%
    height_shift_range=0.1,  # 垂直平移 10%
    horizontal_flip=True,    # 水平翻转（食物图片适用，手写数字不适用）
    zoom_range=0.1,          # 随机缩放
)

model.fit(datagen.flow(X0, Y0, batch_size=32),
          validation_data=(X1, Y1),
          epochs=100)
```

---

## 五、模型对比总结

### 食物评分回归

| 方法 | 结构 | 预期效果 | 适用场景 |
|------|------|---------|---------|
| **原始模型** | Flatten → Dense(1) | 较差，MSE 高 | 仅作为教学演示 |
| MLP | Flatten → Dense×n → Dense(1) | 适中 | 快速实验 |
| **CNN** | Conv2D×n → Dense(1) | 较好 | 中等数据量 |
| **迁移学习** | 预训练模型 → Dense(1) | **最好** | **数据量少时首选** |

### MNIST 分类

| 方法 | 结构 | 预期准确率 | 说明 |
|------|------|----------|------|
| **原始模型** | Dense(10) + Softmax | ~92% | 线性分类器 |
| MLP | Dense×n + Softmax | ~98% | 增加非线性 |
| **CNN** | Conv2D×n + Softmax | **~99.2%** | **图像分类标准方法** |

---

## 六、核心要点总结

1. **原始模型都是线性模型**：Chapter 3 中的两个模型本质上都是线性模型（线性回归和 Softmax 回归），它们的作用是帮助理解深度学习的基本流程，而非追求最佳性能。

2. **CNN 是图像任务的关键进步**：卷积层能够利用图像的空间结构，提取层次化的视觉特征（边缘 → 纹理 → 部件 → 物体），是从简单模型到实用模型的关键一步。

3. **迁移学习适合小数据**：Chapter 3 的食物评分数据集只有约 150 张图片，这种规模的数据最适合用迁移学习，借助 ImageNet 预训练权重来弥补数据不足。

4. **训练策略同样重要**：除了模型架构，合理的学习率调度、早停策略、数据增强、batch size 选择等训练技巧对最终效果的影响也非常大。

5. **理解流程比追求精度更重要**：Chapter 3 的核心教学目标是让学生掌握"数据准备 → 模型构建 → 训练 → 预测"的完整流程，后续章节再逐步引入更复杂的模型。
