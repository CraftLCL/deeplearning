# 实验指导书：TensorFlow 安装与常见数据读取方法

本次实验是一节配套实验，内容以 [chapter3.ipynb](chapter3.ipynb) 中已经讲解过的示例为参考，重点不是完整训练复杂模型，而是掌握几种常见数据的读取、查看与简单预处理方法。

## 一、实验目的

1. 掌握 TensorFlow 的安装与版本验证方法。
2. 理解数据在程序中的基本表示形式，如标量、向量、矩阵和高维数组。
3. 掌握图片数据的读取、缩放、数组化和归一化方法。
4. 掌握 CSV 表格数据的读取方法。
5. 了解 TensorFlow 或相关工具中常见的数据集读取方式。

---

## 二、实验环境

- 操作系统：Windows / Linux / macOS
- Python：3.10 或 3.11
- 主要库：`tensorflow`、`numpy`、`pandas`、`matplotlib`、`Pillow`

安装命令：

```bash
pip install -U pip
pip install tensorflow numpy pandas matplotlib pillow
```

验证 TensorFlow 是否安装成功：

```bash
python -c "import tensorflow as tf; print(tf.__version__)"
```

代码含义说明：

- `python -c` 表示直接在命令行执行一小段 Python 代码。
- `import tensorflow as tf` 表示导入 TensorFlow，并简写成 `tf`。
- `print(tf.__version__)` 用于输出 TensorFlow 的版本号。
- 如果终端正常输出版本号，说明 TensorFlow 基本安装成功。

---

## 三、实验内容

## 实验 1：查看不同维度的数据

这一部分对应 [chapter3.ipynb](chapter3.ipynb) 中的代码示例 3-1 到 3-4，目的是理解数据在程序中的表示形式。

### 代码示例 1：标量

```python
import numpy as np
x = np.array(888)
print(x)
x.ndim
```

代码说明：

- `import numpy as np`：导入 NumPy 库，后面用 `np` 调用其函数。
- `x = np.array(888)`：把数字 `888` 转成 NumPy 数组对象。
- `print(x)`：输出数组中的值。
- `x.ndim`：查看数组的维度。
- 这里的结果是 0，说明它是一个 0 维数据，也就是标量。

### 代码示例 2：向量

```python
x = np.array([1,2,3,4,5])
print(x)
x.ndim
```

代码说明：

- `[1,2,3,4,5]` 表示一个一维列表。
- `np.array(...)` 把它转成 NumPy 数组。
- `x.ndim` 的结果是 1，说明这是一个一维数组，也可以理解为向量。

### 代码示例 3：矩阵

```python
x = np.array([[1,2,3,4],
              [5,6,7,8],
              [9,6,7,4]])
print(x)
x.ndim
```

代码说明：

- 最外层是一个列表，里面有 3 个子列表。
- 每个子列表可以看成矩阵的一行。
- 因此该数组是一个二维数组。
- `x.ndim` 的结果是 2，表示矩阵。

### 代码示例 4：三维数组

```python
x = np.array([[[12,4,6,8,23],
               [45,1,2,6,67]],
              [[32,7,3,5,14],
               [56,1,2,8,18]],
              [[23,7,2,5,78],
               [14,2,7,2,15]]])
print(x)
x.ndim
```

代码说明：

- 这一段数据有 3 层嵌套结构。
- 它不再是简单的行列结构，而是更高维的数据表示。
- `x.ndim` 的结果是 3，表示三维数组。
- 在深度学习中，图片、视频、批量数据通常都不是简单的一维或二维数据，而是更高维的张量。

实验结论：

- 标量是 0 维。
- 向量是 1 维。
- 矩阵是 2 维。
- 更复杂的数据可以表示为 3 维或更高维数组。

---

## 实验 2：读取图片数据并进行简单预处理

这一部分对应 [chapter3.ipynb](chapter3.ipynb) 中的代码示例 3-5 到 3-10。

### 步骤 1：读取图片

```python
from PIL import Image
photo = Image.open('./photos/xiongda.jpg')
photo
```

代码说明：

- `from PIL import Image`：导入 Pillow 库中的 `Image` 模块，用来处理图片。
- `Image.open('./photos/xiongda.jpg')`：打开指定路径的图片。
- `photo`：在 notebook 中直接输出图片对象，通常会显示图片。

### 步骤 2：查看并调整图片大小

```python
print(photo.size)
photo = photo.resize([128, 128])
print(photo.size)
photo
```

代码说明：

- `photo.size`：查看原始图片尺寸，返回 `(宽, 高)`。
- `photo.resize([128, 128])`：把图片统一调整为 `128×128`。
- 再次输出 `photo.size`，可以确认缩放结果。
- 统一图片尺寸是图像处理和模型输入中的常见步骤。

### 步骤 3：将图片转为数组

```python
import numpy as np
Im = np.array(photo)
print(Im.shape)
Im[:,:,0]
```

代码说明：

- `np.array(photo)`：把图片对象转换成 NumPy 数组。
- `Im.shape`：查看数组形状。
- 一般彩色图片的形状类似 `(128, 128, 3)`。
- 前两个数表示高和宽，最后一个 `3` 表示红、绿、蓝三个通道。
- `Im[:,:,0]` 表示取出第 1 个颜色通道的数据。

### 步骤 4：归一化

```python
Im = Im/255
print(Im[:,:,0])
```

代码说明：

- 原始图片像素一般在 0 到 255 之间。
- `Im/255` 是把像素值缩放到 0 到 1 之间。
- 这种处理称为归一化。
- 归一化后，数据更适合后续神经网络计算。

### 步骤 5：显示处理后的图片

```python
from matplotlib import pyplot as plt
plt.imshow(Im);
```

代码说明：

- `pyplot` 是 Matplotlib 中常用的绘图模块。
- `plt.imshow(Im)` 用于显示图片数组。
- 这里显示的是归一化之后的图片。

### 步骤 6：观察简单变换结果

```python
Im1=Im+0.5
Im2=1-Im
Im3=0.5*Im
Im4=Im/0.5
plt.figure()
fig,ax=plt.subplots(1,4)
fig.set_figwidth(15)
ax[0].imshow(Im1)
ax[1].imshow(Im2)
ax[2].imshow(Im3)
ax[3].imshow(Im4)
```

代码说明：

- `Im1=Im+0.5`：整体提高像素值，图像会变亮。
- `Im2=1-Im`：得到类似反色效果。
- `Im3=0.5*Im`：整体减弱像素值，图像会变暗。
- `Im4=Im/0.5`：相当于放大像素值，图像亮度增强。
- `plt.subplots(1,4)`：创建 1 行 4 列的子图。
- `ax[i].imshow(...)`：在每个子图中显示不同处理后的图片。

实验结论：

- 图片本质上可以表示成多维数组。
- 图像读取后通常需要统一大小和归一化。
- 简单的像素变换会直接影响图像显示效果。

---

## 实验 3：读取 CSV 表格数据

这一部分对应 [chapter3.ipynb](chapter3.ipynb) 中的代码示例 3-11 到 3-13。

### 步骤 1：读取 CSV 文件

```python
import pandas as pd
MasterFile=pd.read_csv('./FoodScore.csv')
print(MasterFile.shape)
MasterFile[0:5]
```

代码说明：

- `import pandas as pd`：导入 Pandas 库。
- `pd.read_csv('./FoodScore.csv')`：读取当前目录下的 CSV 文件。
- `MasterFile` 是读取后的表格数据对象。
- `MasterFile.shape` 返回数据的行数和列数。
- `MasterFile[0:5]` 表示查看前 5 行数据。

### 步骤 2：查看数据分布

```python
MasterFile.hist()
```

代码说明：

- `hist()` 会对表格中的数值列绘制直方图。
- 这样可以初步观察各列数据的大致分布情况。

### 步骤 3：提取文件名和标签

```python
import numpy as np
FileNames=MasterFile['ID']
N=len(FileNames)
Y=np.array(MasterFile['score']).reshape([N,1])
#Y=(Y-np.mean(Y))/np.std(Y)
```

代码说明：

- `MasterFile['ID']`：读取表格中的 `ID` 列。
- 这一列通常对应图片文件名。
- `N=len(FileNames)`：统计样本数量。
- `MasterFile['score']`：读取评分列。
- `np.array(...).reshape([N,1])`：将评分转换为 `N×1` 的数组。
- 注释掉的那一行表示标准化处理，即减去均值后再除以标准差。
- 本实验中先不强制使用标准化，只需要理解其作用即可。

实验结论：

- CSV 文件适合存储结构化数据。
- Pandas 可以方便地读取和查看表格内容。
- 读取表格后，常常需要把某些列提取出来作为标签或文件索引。

---

## 实验 4：根据 CSV 信息批量读取图片

这一部分对应 [chapter3.ipynb](chapter3.ipynb) 中的代码示例 3-14 和 3-15，目的是把“表格信息”和“图片数据”对应起来。

### 步骤 1：批量读取图片

```python
from PIL import Image

IMSIZE=128
X=np.zeros([N,IMSIZE,IMSIZE,3])
for i in range(N):
    MyFile=FileNames[i]
    Im=Image.open('../case3-food/data/'+MyFile+'.jpg')
    Im=Im.resize([IMSIZE,IMSIZE])
    Im=np.array(Im)/255
    X[i,]=Im
```

代码说明：

- `IMSIZE=128`：规定所有图片统一大小为 `128×128`。
- `X=np.zeros([N,IMSIZE,IMSIZE,3])`：先创建一个全 0 的四维数组，用来存放全部图片数据。
- `N` 表示图片数量。
- `for i in range(N)`：依次读取每一张图片。
- `MyFile=FileNames[i]`：取出第 `i` 个图片文件名。
- `Image.open(...)`：按文件名读取图片。
- `resize([IMSIZE,IMSIZE])`：统一缩放大小。
- `np.array(Im)/255`：转成数组并归一化。
- `X[i,]=Im`：把当前图片存入总数组 `X` 中。

### 步骤 2：显示部分样本

```python
from matplotlib import pyplot as plt
plt.figure()
fig,ax=plt.subplots(2,5)
fig.set_figheight(7.5)
fig.set_figwidth(15)
ax=ax.flatten()
for i in range(10):
    ax[i].imshow(X[i,:,:,:])
    ax[i].set_title(np.round(Y[i],2))
```

代码说明：

- `plt.subplots(2,5)`：生成 2 行 5 列的子图。
- `ax.flatten()`：把二维坐标轴数组展平成一维，便于循环使用。
- `ax[i].imshow(X[i,:,:,:])`：显示第 `i` 张图片。
- `ax[i].set_title(np.round(Y[i],2))`：把对应评分作为标题显示出来。
- 这一部分的作用是检查“图片”和“标签”是否正确对应。

实验结论：

- 表格中的文件名可以作为索引，用来批量读取图片。
- 图片数据和标签数据可以分别组织成 `X` 和 `Y`。
- 这是很多图像任务中常见的数据准备方式。

---

## 实验 5：读取常见公开数据集

这一部分对应 [chapter3.ipynb](chapter3.ipynb) 中的代码示例 3-21 到 3-23。这里以 MNIST 手写数字数据集为例，说明现成数据集的读取方式。

### 步骤 1：读取 MNIST 数据集

```python
from tensorflow.examples.tutorials.mnist import input_data
data = input_data.read_data_sets("data/MNIST/",one_hot=False)
```

代码说明：

- `input_data` 是一个用于读取 MNIST 数据集的工具。
- `read_data_sets(...)` 表示读取或下载数据集。
- `"data/MNIST/"` 是数据保存目录。
- `one_hot=False` 表示标签先按普通整数形式读取，例如 0 到 9。

### 步骤 2：取出训练集和验证集

```python
X0 = data.train.images
Y0 = data.train.labels
X1 = data.validation.images
Y1 = data.validation.labels
print(X0.shape)
```

代码说明：

- `data.train.images`：训练集图片数据。
- `data.train.labels`：训练集标签。
- `data.validation.images`：验证集图片数据。
- `data.validation.labels`：验证集标签。
- `print(X0.shape)`：查看训练集输入数据的形状。
- 对 MNIST 来说，每张图片通常会被展开为长度为 784 的向量。

### 步骤 3：显示不同数字样本

```python
from matplotlib import pyplot as plt
plt.figure()
fig,ax = plt.subplots(2,5)
ax=ax.flatten()
for i in range(10):
    Im=X0[Y0==i][0].reshape(28,28)
    ax[i].imshow(Im)
plt.show()
```

代码说明：

- `for i in range(10)`：依次处理数字 0 到 9。
- `X0[Y0==i]`：找出标签等于 `i` 的所有图片。
- `[0]`：取这一类中的第一张图。
- `reshape(28,28)`：把长度为 784 的向量恢复成 `28×28` 图片。
- `ax[i].imshow(Im)`：显示该数字样本。
- 这样可以直观看到 MNIST 数据集中不同数字的图像形式。

实验结论：

- 除了本地图片和 CSV 文件，还可以直接读取现成公开数据集。
- 这类数据集通常已经完成了基本整理，适合教学和模型验证。

---

## 四、实验总结

通过本实验，应掌握以下几点：

1. TensorFlow 可以通过命令行安装并验证版本。
2. 深度学习中的数据通常以数组或张量形式表示。
3. 图片读取的基本流程是：打开图片、调整大小、转成数组、归一化。
4. CSV 读取的基本流程是：读取表格、查看字段、提取需要的列。
5. 数据既可以来自本地文件，也可以来自现成公开数据集。

---

## 五、实验报告要求

实验报告建议包含以下内容：

1. TensorFlow 安装与版本验证结果。
2. 标量、向量、矩阵、三维数组的运行结果截图或说明。
3. 图片读取与归一化的主要代码及结果说明。
4. CSV 数据读取与字段提取的主要代码及结果说明。
5. 你对“常见数据读取方法”的理解与总结。

---

## 六、思考题

1. 为什么图片在进入模型前通常要统一尺寸？
2. 为什么图片像素值常常要除以 255？
3. CSV 文件和图片文件在数据组织方式上有什么不同？
4. 公开数据集读取和本地文件读取，各自适合什么场景？
