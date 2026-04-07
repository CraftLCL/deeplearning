# 实验1：TensorFlow安装及几种常见的数据读取方法

## 一、实验目的

1. 掌握TensorFlow的安装方法
2. 了解TensorFlow 1.x与2.x的主要区别
3. 掌握图片数据的读取与预处理方法
4. 掌握CSV表格数据的读取方法
5. 掌握MNIST等内置数据集的加载方法
6. 掌握使用ImageDataGenerator从文件夹读取数据的方法

## 二、实验环境

- Python 3.8+
- TensorFlow 2.x（推荐 2.10+）
- NumPy、Pandas、Pillow、Matplotlib、scikit-learn

## 三、实验原理

### 3.1 TensorFlow简介

TensorFlow是Google开发的开源深度学习框架。TF 1.x采用静态计算图（需要Session），TF 2.x默认启用Eager Execution（即时执行），代码更加Pythonic。

### 3.2 张量（Tensor）的概念

TensorFlow的名字来源于"Tensor"（张量），张量是深度学习中最基本的数据结构。可以把张量理解为多维数组的推广：

| 维度 | 名称 | 示例 | 形状（Shape） |
|------|------|------|---------------|
| 0维 | **标量（Scalar）** | 一个数字，如 `7` | `()` |
| 1维 | **向量（Vector）** | 一组数字，如 `[1, 2, 3]` | `(3,)` |
| 2维 | **矩阵（Matrix）** | 表格数据，如一张灰度图 | `(28, 28)` |
| 3维 | **3阶张量** | 一张彩色图片 | `(高, 宽, 通道)` |
| 4维 | **4阶张量** | 一批彩色图片 | `(批量, 高, 宽, 通道)` |

在图像处理中，一张彩色图片是一个3维张量，形状为 **(高×宽×通道)**。例如一张128×128的RGB图片，其形状为 `(128, 128, 3)`。

### 3.3 RGB颜色模型

计算机中彩色图片采用 **RGB颜色模型**，每个像素由三个通道组成：
- **R（Red）**：红色通道
- **G（Green）**：绿色通道
- **B（Blue）**：蓝色通道

每个通道的取值范围为 **0～255**（8位无符号整数），其中0表示该颜色分量为零，255表示该颜色分量最强。例如：
- `(255, 0, 0)` → 纯红色
- `(0, 255, 0)` → 纯绿色
- `(0, 0, 0)` → 黑色
- `(255, 255, 255)` → 白色

### 3.4 数据读取的重要性

深度学习模型训练的第一步是数据准备。常见的数据格式包括：
- **图片文件**：JPG、PNG等，需转为NumPy数组
- **CSV表格**：结构化数据，使用Pandas读取
- **内置数据集**：如MNIST，框架自带加载函数
- **文件夹数据集**：按类别组织在文件夹中，使用ImageDataGenerator读取

### 3.5 数据预处理

- **统一尺寸**：神经网络要求所有输入维度一致
- **归一化**：将像素值从[0,255]缩放到[0,1]，有助于训练稳定
- **数据划分**：将数据分为训练集和测试集

#### 为什么要进行归一化？

将像素值从 `[0, 255]` 归一化到 `[0, 1]`（即除以255）的原因：

1. **梯度稳定性**：神经网络通过梯度下降优化参数。如果输入值很大（如0～255），会导致梯度值也很大，参数更新幅度过大，训练过程容易震荡甚至发散。归一化后输入范围较小，梯度更加稳定。
2. **加快收敛速度**：归一化使得不同特征处于相同的数值范围，优化算法能更高效地找到最优解，训练所需的迭代次数更少。
3. **数值精度**：浮点运算在数值较小时精度更高，归一化有助于减少计算中的数值误差。

#### 训练集与测试集划分

将数据分为**训练集（Training Set）**和**测试集（Test Set）**是机器学习的基本原则：

- **训练集**：用于训练模型、更新参数
- **测试集**：用于评估模型在**未见过的数据**上的表现（即泛化能力）

为什么不能用全部数据训练？因为模型可能会出现**过拟合（Overfitting）**——即模型"记住"了训练数据的噪声和细节，在训练集上表现很好，但在新数据上表现很差。通过划分测试集，我们能够检测模型是否具有良好的泛化能力。

#### One-Hot编码

在分类任务中，类别标签（如数字0～9）需要转换为**One-Hot编码**。One-Hot编码将每个类别表示为一个向量，其中只有对应类别位置为1，其余全为0。

例如，对于10个类别（数字0～9），数字 **3** 的One-Hot编码为：
```
[0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
```

为什么需要One-Hot编码？因为如果直接使用数字标签（0, 1, 2, ...），模型可能会误认为标签之间存在大小关系（如3 > 1），但实际上分类标签之间是**无序的**。One-Hot编码消除了这种隐含的大小关系，让模型正确地学习分类。

## 四、实验步骤

### 步骤1：安装TensorFlow

```bash
# 安装最新版TensorFlow（CPU版本）
pip install tensorflow

# 如果需要GPU支持
pip install tensorflow[and-cuda]

# 验证安装
python -c "import tensorflow as tf; print(tf.__version__)"
```

### 步骤2：读取单张图片并预处理

```python
from PIL import Image
import numpy as np

# 读取图片
photo = Image.open('./photos/xiongda.jpg')

# 查看原始尺寸
print(photo.size)

# 统一尺寸为128x128
photo = photo.resize([128, 128])

# 转为NumPy数组
Im = np.array(photo)
print(Im.shape)  # (128, 128, 3) —— 高x宽x通道(RGB)

# 归一化到[0,1]
Im = Im / 255
```

**代码解析**：
- `Image.open('./photos/xiongda.jpg')`：使用PIL（Python Imaging Library）库打开图片文件，返回一个Image对象。此时图片数据尚未转为数组，仍以PIL内部格式存储。
- `photo.size`：返回图片的宽和高（注意PIL的size返回的是 `(宽, 高)`，与NumPy数组的维度顺序相反）。
- `photo.resize([128, 128])`：将图片统一缩放为128×128像素。神经网络要求所有输入尺寸相同，因此必须先统一尺寸。
- `np.array(photo)`：将PIL图片对象转换为NumPy数组。对于RGB图片，得到的数组形状为 `(128, 128, 3)`，即**高×宽×通道**。每个像素有R、G、B三个值，取值范围为0～255。
- `Im / 255`：将像素值从整数范围 `[0, 255]` 归一化到浮点数范围 `[0, 1]`，这是深度学习数据预处理的标准做法。

### 步骤3：像素变换与可视化

```python
from matplotlib import pyplot as plt

Im1 = Im + 0.5    # 加亮
Im2 = 1 - Im      # 反色
Im3 = 0.5 * Im    # 变暗
Im4 = Im / 0.5    # 增强亮度

plt.figure()
fig, ax = plt.subplots(1, 4)
fig.set_figwidth(15)
ax[0].imshow(Im1)
ax[1].imshow(Im2)
ax[2].imshow(Im3)
ax[3].imshow(Im4)
plt.show()
```

**代码解析**：
- `Im + 0.5`（加亮）：对每个像素值加0.5，整体亮度增加。超过1.0的值在显示时会被截断为1.0（白色）。数学表示：`Im'(x,y) = Im(x,y) + 0.5`。
- `1 - Im`（反色/负片效果）：用1减去每个像素值，亮的变暗、暗的变亮，颜色也会翻转。数学表示：`Im'(x,y) = 1 - Im(x,y)`。
- `0.5 * Im`（变暗）：将每个像素值乘以0.5，所有颜色亮度减半。数学表示：`Im'(x,y) = 0.5 × Im(x,y)`。
- `Im / 0.5`（增强亮度）：等价于乘以2，亮度加倍。超过1.0的值会被截断。数学表示：`Im'(x,y) = Im(x,y) / 0.5 = 2 × Im(x,y)`。
- `plt.subplots(1, 4)`：创建1行4列的子图布局，用于并排展示4种变换效果。
- `ax[i].imshow()`：在第i个子图中显示图片。`imshow`接受 `[0, 1]` 范围的浮点数组或 `[0, 255]` 范围的整数数组。

### 步骤4：读取CSV表格数据

```python
import pandas as pd
import numpy as np

# 读取CSV文件
MasterFile = pd.read_csv('./FoodScore.csv')
print(MasterFile.shape)
MasterFile.head()

# 查看数据分布
MasterFile.hist()

# 提取文件名和标签
FileNames = MasterFile['ID']
N = len(FileNames)
Y = np.array(MasterFile['score']).reshape([N, 1])
```

**代码解析**：
- `pd.read_csv('./FoodScore.csv')`：使用Pandas库读取CSV（逗号分隔值）文件，返回一个DataFrame对象。DataFrame是Pandas中最核心的数据结构，类似于Excel表格，有行索引和列名。
- `MasterFile.shape`：返回数据的形状 `(行数, 列数)`，即样本数量和特征数量。
- `MasterFile.head()`：显示前5行数据，用于快速预览数据内容和格式。
- `MasterFile.hist()`：为每一列数据绘制直方图，可以直观地查看数据的分布情况（如是否均匀、是否有偏斜等）。
- `MasterFile['ID']`：通过列名访问DataFrame中的某一列，返回一个Series对象。
- `.reshape([N, 1])`：将一维数组重塑为二维列向量。形状从 `(N,)` 变为 `(N, 1)`，这是许多机器学习函数要求的标签格式。

### 步骤5：批量读取图片数据集

```python
from PIL import Image
import numpy as np

IMSIZE = 128
X = np.zeros([N, IMSIZE, IMSIZE, 3])

for i in range(N):
    MyFile = FileNames[i]
    Im = Image.open('./data_foodscore/' + MyFile + '.jpg')
    Im = Im.resize([IMSIZE, IMSIZE])
    Im = np.array(Im) / 255
    X[i,] = Im
```

**代码解析**：
- `IMSIZE = 128`：定义统一的图片尺寸常量，方便后续修改。
- `np.zeros([N, IMSIZE, IMSIZE, 3])`：预先分配一个全零的4维数组，形状为 `(样本数, 高, 宽, 通道)`。预分配内存比在循环中逐步拼接数组效率高得多，因为NumPy数组大小固定，动态拼接需要反复申请内存和复制数据。
- `for i in range(N)`：遍历所有样本，逐张读取图片。这是批量加载图片数据集的常见模式。
- `Image.open(...)`：根据文件名拼接路径，打开对应的图片文件。
- `Im.resize([IMSIZE, IMSIZE])`：统一缩放到目标尺寸。
- `np.array(Im) / 255`：转为数组并归一化。
- `X[i,] = Im`：将处理后的图片数据存入预分配数组的第i个位置。`X[i,]` 等价于 `X[i, :, :, :]`，表示第i张图片的所有像素。

### 步骤6：划分训练集和测试集

```python
# [旧写法] from sklearn.cross_validation import train_test_split
# ↑ sklearn.cross_validation 在 sklearn 0.18 中已废弃，0.20 中已移除

# [新写法] 适用于 sklearn >= 0.18
from sklearn.model_selection import train_test_split

X0, X1, Y0, Y1 = train_test_split(X, Y, test_size=0.5, random_state=0)
```

**代码解析**：
- `from sklearn.model_selection import train_test_split`：从scikit-learn库导入数据划分函数。注意旧版本（sklearn < 0.18）使用 `sklearn.cross_validation`，该模块已被移除。
- `train_test_split(X, Y, ...)`：该函数自动将特征矩阵X和标签Y**同步**打乱并划分，确保特征和标签的对应关系不会错乱。
- `test_size=0.5`：测试集占总数据的50%。常见的比例有 `0.2`（80%训练/20%测试）、`0.3`（70%/30%）等。此处使用0.5是为了演示，实际项目中通常训练集比例更大。
- `random_state=0`：随机种子。设置固定的随机种子可以保证每次运行代码时划分结果完全一致，便于实验结果复现和调试。如果不设置，每次运行会得到不同的划分。
- 返回值 `X0, X1, Y0, Y1`：分别对应训练集特征、测试集特征、训练集标签、测试集标签。

### 步骤7：加载MNIST内置数据集

```python
# [旧写法] TensorFlow 1.x 教程接口（已废弃）
# from tensorflow.examples.tutorials.mnist import input_data
# data = input_data.read_data_sets("data/MNIST/", one_hot=False)

# [新写法] 适用于 TensorFlow >= 2.0
from tensorflow.keras.datasets import mnist

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

# 展平 + 归一化
X_train = X_train.reshape(-1, 28 * 28).astype("float32") / 255.0
X_test = X_test.reshape(-1, 28 * 28).astype("float32") / 255.0

print(X_train.shape)  # (60000, 784)
```

**代码解析**：
- `mnist.load_data()`：加载MNIST手写数字数据集。该函数会自动从网络下载（首次使用时）并缓存到本地。返回两个元组：训练集和测试集。
- `(X_train, Y_train), (X_test, Y_test)`：使用元组解包获取数据。`X_train` 的原始形状为 `(60000, 28, 28)`，即60000张28×28像素的灰度图；`X_test` 的形状为 `(10000, 28, 28)`。
- `reshape(-1, 28 * 28)`：将二维图片展平为一维向量。`28 * 28 = 784`，即每张图片变成一个784维的向量。参数 `-1` 表示"自动计算该维度的大小"，NumPy会根据总元素数量自动推断，此处 `-1` 会被计算为60000（或10000）。这样做是因为全连接神经网络要求输入是一维向量。
- `.astype("float32")`：将数据类型从uint8（0～255的整数）转换为float32（浮点数），这是神经网络计算所需的数据类型。
- `/ 255.0`：归一化到 `[0, 1]` 范围。

### 步骤8：使用ImageDataGenerator从文件夹读取数据

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

# 查看一个批次的数据
X, Y = next(train_generator)
print(X.shape, Y.shape)
```

**代码解析**：
- `ImageDataGenerator(rescale=1./255)`：创建图片数据生成器，`rescale=1./255` 表示自动将像素值乘以 `1/255`，即归一化到 `[0, 1]`。
- `.flow_from_directory('data/train/', ...)`：从指定文件夹中读取图片。该方法要求文件夹按类别组织，例如 `data/train/cat/`、`data/train/dog/`，每个子文件夹名即为类别名。
- `target_size=(IMSIZE, IMSIZE)`：将所有图片统一缩放到指定尺寸（227×227），这是自动完成的，无需手动resize。
- `batch_size=200`：每次迭代返回200张图片及其标签。批量加载而非一次性全部加载，可以**节省内存**，特别适合大规模数据集。
- `class_mode='categorical'`：标签格式为One-Hot编码（多分类任务）。如果是二分类任务，可以使用 `'binary'`。
- `next(train_generator)`：获取一个批次的数据，返回 `(图片数组, 标签数组)`。生成器模式的优势是**按需加载**，不需要将所有数据一次性读入内存。

### 步骤9：数据增强

```python
train_generator = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.5,
    rotation_range=30,
    zoom_range=0.2,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
).flow_from_directory(
    'data/train/',
    target_size=(IMSIZE, IMSIZE),
    batch_size=200,
    class_mode='categorical')
```

**代码解析**：

数据增强（Data Augmentation）是通过对训练图片施加随机变换来人为扩大训练集规模的技术。每个参数的含义如下：

- `rescale=1./255`：归一化，将像素值缩放到 `[0, 1]`。
- `shear_range=0.5`：**剪切变换**，随机对图片进行剪切（错切）变换，强度最大为0.5弧度。效果类似于将图片沿对角方向拉伸，使矩形变成平行四边形。
- `rotation_range=30`：**随机旋转**，在 `[-30°, +30°]` 范围内随机旋转图片。适用于目标物体可能以不同角度出现的场景。
- `zoom_range=0.2`：**随机缩放**，在 `[1-0.2, 1+0.2]` 即 `[0.8, 1.2]` 范围内随机缩放图片。模拟物体远近不同的情况。
- `width_shift_range=0.2`：**水平平移**，随机将图片水平移动最多20%的宽度。模拟物体不在画面中央的情况。
- `height_shift_range=0.2`：**垂直平移**，随机将图片垂直移动最多20%的高度。
- `horizontal_flip=True`：**水平翻转**，随机左右翻转图片。适用于左右对称的目标（如猫、狗），但不适用于有方向性的目标（如文字）。

数据增强的目的是让模型"见到"更多样化的训练样本，从而提高模型的泛化能力，减少过拟合。

### 步骤10：One-Hot编码

```python
# [旧写法] from keras.utils import to_categorical
# [旧写法] from keras.utils import np_utils; np_utils.to_categorical(Y)
# ↑ 独立 keras 包及 np_utils 在 TF 2.0+ 中已不推荐/已移除

# [新写法] 适用于 TensorFlow >= 2.0
from tensorflow.keras.utils import to_categorical

YY_train = to_categorical(Y_train)
YY_test = to_categorical(Y_test)
print(YY_train.shape)  # (60000, 10)
```

**代码解析**：
- `from tensorflow.keras.utils import to_categorical`：导入One-Hot编码工具函数。
- `to_categorical(Y_train)`：将整数标签数组转换为One-Hot编码矩阵。例如：
  - 标签 `0` → `[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]`
  - 标签 `3` → `[0, 0, 0, 1, 0, 0, 0, 0, 0, 0]`
  - 标签 `9` → `[0, 0, 0, 0, 0, 0, 0, 0, 0, 1]`
- `YY_train.shape` 为 `(60000, 10)`：60000个样本，每个样本有10个值（对应0～9共10个类别）。
- 该函数会自动检测类别数量（此处为10），也可以通过参数 `num_classes` 手动指定。
- 分类任务中，神经网络输出层通常使用 **Softmax** 激活函数，输出每个类别的概率，与One-Hot编码的标签格式相匹配，然后使用**交叉熵损失函数**计算误差。

## 五、新旧写法对照总表

| 功能 | 旧写法 (TF 1.x / 独立keras) | 新写法 (TF 2.x) |
|------|------|------|
| 数据集加载 | `tensorflow.examples.tutorials.mnist` | `tensorflow.keras.datasets.mnist` |
| 图片生成器 | `from keras.preprocessing.image import ImageDataGenerator` | `from tensorflow.keras.preprocessing.image import ImageDataGenerator` |
| One-Hot编码 | `from keras.utils import np_utils` | `from tensorflow.keras.utils import to_categorical` |
| 数据划分 | `from sklearn.cross_validation import train_test_split` | `from sklearn.model_selection import train_test_split` |
| Session执行 | `session = tf.Session(); session.run(x)` | `x.numpy()` (Eager Execution) |
| 训练模型 | `model.fit_generator(generator, ...)` | `model.fit(generator, ...)` |

## 六、思考题

1. TensorFlow 1.x与2.x的最大区别是什么？为什么2.x更适合教学？
2. 为什么要对图片进行归一化处理？如果不归一化会怎样？
3. `train_test_split`中的`random_state`参数有什么作用？
4. `ImageDataGenerator`中的数据增强有哪些方式？各适用于什么场景？
