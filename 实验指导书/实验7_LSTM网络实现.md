# 实验7：LSTM网络实现

## 一、实验目的

1. 理解循环神经网络（RNN）和长短期记忆网络（LSTM）的原理
2. 掌握LSTM解决序列数据建模的方法
3. 使用TensorFlow/Keras实现基于LSTM的古诗生成模型
4. 理解Embedding层、序列补全（padding）、序列到序列的训练方式

## 二、实验环境

- Python 3.8+
- TensorFlow 2.x（推荐 2.10+）
- NumPy

## 三、实验原理

### 3.1 RNN的局限性

标准RNN在处理长序列时存在**梯度消失/爆炸**问题，导致无法有效学习长距离依赖关系。

**直觉理解**：在反向传播过程中，梯度需要沿着时间步逐步传递。每经过一个时间步，梯度都要乘以一个权重矩阵。设想一下：

- **梯度消失**：如果每次乘以一个小于1的数（比如0.5），经过100步后：$0.5^{100} \approx 10^{-31}$，梯度趋近于零。这就像复印机反复复印——每次复印都会丢失一些信息，复印100次后几乎什么都看不清了。
- **梯度爆炸**：如果每次乘以一个大于1的数（比如2.0），经过100步后：$2.0^{100} \approx 10^{30}$，梯度趋近于无穷大。这就像麦克风对着音箱——声音不断放大，最终产生刺耳的尖啸。

**为什么这会阻碍学习长距离依赖？** 举个具体例子：假设模型正在处理一首诗"**床**前明月光疑是地上霜举头望明月低头思故**乡**"。当模型试图学习"第1个字'床'对预测第20个字'乡'的影响"时，梯度需要从第20个位置一路传回第1个位置，经过19次乘法运算。如果梯度消失了，模型就无法学到"床"和"乡"之间的关联——它"记不住"这么远的信息。这就是为什么标准RNN在处理长序列时表现很差。

### 3.2 LSTM的核心思想

LSTM通过引入**门控机制（Gate Mechanism）**来控制信息的流动。我们可以用**白板**来类比理解LSTM的工作方式：想象你面前有一块白板（细胞状态），你需要一边听课一边在白板上做笔记。

- **遗忘门（Forget Gate）**——"擦白板"：决定擦掉哪些旧笔记
  $$f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$$

  > **直觉理解**：遗忘门就像橡皮擦，决定白板上哪些旧内容需要擦掉。$f_t$ 的值在0到1之间：0表示"完全擦掉"，1表示"完全保留"。例如，在处理古诗时，当模型读到句号（一句结束），遗忘门可能会选择擦掉上一句的主语信息，因为新的一句可能有新的主语。

- **输入门（Input Gate）**——"写白板"：决定把哪些新信息写上去
  $$i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)$$
  $$\tilde{C}_t = \tanh(W_C \cdot [h_{t-1}, x_t] + b_C)$$

  > **直觉理解**：输入门就像你的笔，决定要把哪些新内容写到白板上。这里分两步：$i_t$ 决定"要不要写"（0=不写，1=写），$\tilde{C}_t$ 决定"写什么内容"。例如，当模型读到"明月"，输入门可能会决定把"月亮/夜晚/思念"等语义信息写到白板上。

- **细胞状态更新（Cell State）**——"白板本身"：长期记忆的载体
  $$C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$$

  > **直觉理解**：细胞状态就是白板本身，是LSTM的"长期记忆"。更新过程很直观：先用遗忘门擦掉一部分旧内容（$f_t \odot C_{t-1}$），再用输入门写上一些新内容（$i_t \odot \tilde{C}_t$）。正是因为细胞状态的更新是**加法**而非乘法，梯度可以沿着细胞状态"高速公路"顺畅流动，从而解决了梯度消失问题。

- **输出门（Output Gate）**——"读白板"：决定从白板上读出什么来汇报
  $$o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)$$
  $$h_t = o_t \odot \tanh(C_t)$$

  > **直觉理解**：输出门就像你看着白板决定"这次汇报什么"。白板上可能记了很多内容，但当前时刻只需要输出一部分。例如，白板上同时记着"夜晚""思念""月亮"等信息，但当前需要预测的下一个字与"月亮"最相关，输出门就会选择性地输出与"月亮"相关的信息。

### 3.3 Embedding层

将离散的字/词索引映射为稠密的向量表示。例如将"月"(索引14)映射为一个128维的实数向量。

**为什么需要Embedding？** 在自然语言处理中，最朴素的文字表示方式是**One-Hot编码**：

- 假设词汇表有5000个字，那么每个字用一个5000维的向量表示，其中只有一个位置是1，其余全是0。
- 例如："月"=[0,0,...,1,...,0,0]（第14个位置为1），"光"=[0,0,...,1,...,0,0]（第27个位置为1）
- **问题1：维度太高**。5000个字就需要5000维的向量，计算量巨大，浪费存储空间。
- **问题2：无法表达语义相似性**。在One-Hot中，任意两个字之间的距离（余弦相似度）都完全相同——"月"和"日"跟"月"和"马"一样"远"，但显然"月"和"日"在语义上更接近。

**Embedding的优势**：

- **低维稠密**：将5000个字映射到128维的向量空间中（5000维 → 128维），每个维度都是有意义的实数值。
- **语义相近的字，向量也相近**：训练后，"月"和"日"的向量距离会比"月"和"马"更近，因为它们在古诗中经常出现在相似的上下文中。
- **Embedding是通过训练学习的**：Embedding层的权重（即查找表）不是人工设定的，而是在训练过程中由模型自动学习得到的。模型会根据"哪些字出现在相似的上下文中"来调整向量，使语义相近的字自然聚集在向量空间中。
- **本质是查找表**：Embedding层本质上是一个大小为 `(vocab_size, embedding_dim)` 的矩阵。给定一个字的索引（如14），直接查找矩阵的第14行，返回对应的128维向量。这比One-Hot乘以权重矩阵的方式高效得多。

### 3.4 序列到序列的训练

对于诗歌生成任务：
- 输入X：`[寒, 随, 穷, 律, 变, 春, 逐, ...]`（前N-1个字）
- 输出Y：`[随, 穷, 律, 变, 春, 逐, 鸟, ...]`（后N-1个字）
- 模型学习根据前文预测下一个字

**具体示例**：以"床前明月光"这5个字为例，展示X和Y的构造过程：

| 位置 | X（输入） | Y（目标输出） | 模型要学的规律 |
|------|-----------|---------------|----------------|
| 1    | 床        | 前            | 看到"床"，预测下一个字是"前" |
| 2    | 前        | 明            | 看到"床前"，预测下一个字是"明" |
| 3    | 明        | 月            | 看到"床前明"，预测下一个字是"月" |
| 4    | 月        | 光            | 看到"床前明月"，预测下一个字是"光" |

也就是说，X = [床, 前, 明, 月]，Y = [前, 明, 月, 光]。Y相对于X向后偏移了一个位置。通过这种方式，模型在每个时间步都在学习"根据已经看到的所有前文，预测下一个字是什么"。由于设置了`return_sequences=True`，模型会在**每个时间步**都输出预测结果，而不是只在最后一步输出。

### 3.5 Tokenizer与pad_sequences

在将文本送入神经网络之前，需要进行两个关键的预处理步骤：

**Tokenizer（分词器/词表构建器）**：

- **作用**：扫描所有文本，为每个不同的字分配一个唯一的整数索引，构建"字→整数"的映射词表。
- **例子**：假设语料中出现了"床前明月光"这几个字，Tokenizer可能建立如下映射：
  - "月"→3, "不"→5, "人"→7, "床"→42, "前"→58, "明"→14, "光"→27 ...
  - （按出现频率排序，越常见的字索引越小）
- **`word_index`属性**：返回完整的"字→索引"字典，如 `{'月': 3, '不': 5, '人': 7, ...}`
- **`texts_to_sequences()`方法**：将文本列表转为整数序列，如将 `['床','前','明','月','光']` 转为 `[42, 58, 14, 3, 27]`
- **为什么索引从1开始**：索引0被保留给**padding（填充符）**，表示"没有字"的空位。

**pad_sequences（序列补全/填充）**：

- **问题**：不同的诗长度不同（有的20个字，有的28个字），但神经网络要求输入是**固定长度**的矩阵。
- **解决**：`pad_sequences` 将所有序列补全到相同长度（`maxlen`），不够长的用0填充。
- **`padding='post'`**：在序列**末尾**填充0（如 `[42,58,14,3,27]` → `[42,58,14,3,27,0,0,...,0]`）；如果设为 `'pre'`，则在序列**开头**填充0。
- **`mask_zero=True`的配合**：Embedding层设置 `mask_zero=True` 后，模型会自动识别并**忽略**这些填充的0，不会把填充位置的信息纳入训练计算。这样填充的0就不会干扰模型学习。

### 3.6 return_sequences参数

LSTM层有一个重要的参数 `return_sequences`，它决定了LSTM的输出形态：

- **`return_sequences=True`（输出每个时间步的结果）**：
  - 输入形状：`(batch_size, timesteps, features)`
  - 输出形状：`(batch_size, timesteps, hidden_size)`
  - LSTM在每个时间步都输出一个隐藏状态向量
  - **适用场景**：序列到序列任务（如本实验的诗歌生成——每个时间步都要预测下一个字）

- **`return_sequences=False`（只输出最后一个时间步的结果）**：
  - 输入形状：`(batch_size, timesteps, features)`
  - 输出形状：`(batch_size, hidden_size)`
  - LSTM只输出最后一个时间步的隐藏状态
  - **适用场景**：分类任务（如情感分析——读完整个句子后给出一个分类结果）

在本实验中，我们需要模型在**每个位置**都预测下一个字，因此必须使用 `return_sequences=True`。

## 四、实验步骤

### 步骤1：数据准备 —— 读取和处理古诗

```python
import string
import numpy as np

f = open('poems_clean.txt', "r", encoding='utf-8')
poems = []
for line in f.readlines():
    title, poem = line.split(':')
    poem = poem.replace(' ', '')    # 去空格
    poem = poem.replace('\n', '')   # 去换行符
    poems.append(list(poem))

print(poems[0][:])
```

**代码解析**：

1. **文件读取**：`open('poems_clean.txt', "r", encoding='utf-8')` 以UTF-8编码打开古诗文件。该文件每行存储一首诗，格式为 `标题:诗句内容`。
2. **按冒号分割**：`line.split(':')` 将每行按冒号分割为标题和诗句两部分。我们只需要诗句内容，标题不参与训练。
3. **清理文本**：`replace(' ', '')` 去掉空格，`replace('\n', '')` 去掉换行符，确保诗句是纯净的汉字序列。
4. **转为字符列表**：`list(poem)` 将字符串拆分为单字列表，如 `"床前明月光"` → `['床', '前', '明', '月', '光']`。在古诗生成中，我们以**单个汉字**为基本单位（而非词语），因为古诗的语言结构以字为核心。

### 步骤2：文本编码与序列补全

```python
# [旧写法] from keras.preprocessing.text import Tokenizer
# [旧写法] from keras.preprocessing.sequence import pad_sequences
# ↑ 独立 keras 包在 TF 2.0+ 中已不推荐使用

# [新写法] 适用于 TensorFlow >= 2.0
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

tokenizer = Tokenizer()
tokenizer.fit_on_texts(poems)
vocab_size = len(tokenizer.word_index) + 1   # 加上停止词0的位置

# 将文本转为数字序列
poems_digit = tokenizer.texts_to_sequences(poems)

# 补全到统一长度（50个字）
poems_digit = pad_sequences(poems_digit, maxlen=50, padding='post')
print(poems_digit.shape)  # (24117, 50)
```

**代码解析**：

1. **创建Tokenizer**：`Tokenizer()` 创建一个分词器对象。`fit_on_texts(poems)` 扫描所有诗句，统计每个字的出现频率，并按频率从高到低为每个字分配一个唯一整数索引。可以通过 `tokenizer.word_index` 查看完整的"字→索引"映射字典。
2. **词汇表大小+1**：`len(tokenizer.word_index) + 1` 中的 `+1` 是因为索引0被保留给padding（填充符），Tokenizer分配的索引从1开始。所以如果有4500个不同的字，`vocab_size = 4501`。
3. **文本转数字**：`texts_to_sequences(poems)` 将每首诗的字符列表转为对应的整数索引列表。例如 `['床','前','明','月','光']` → `[42, 58, 14, 3, 27]`。
4. **序列补全**：`pad_sequences(poems_digit, maxlen=50, padding='post')` 将所有序列统一为长度50。不足50个字的诗在**末尾**（`post`）补0。例如一首20字的诗：`[42,58,...,27]` → `[42,58,...,27,0,0,...,0]`（后面补30个0）。超过50字的诗则被截断。输出形状为 `(24117, 50)`，即24117首诗，每首50个字（含padding）。

### 步骤3：构造训练数据

```python
# X是前49个字，Y是后49个字（向后错一位）
X = poems_digit[:, :-1]   # 形状: (24117, 49)
Y = poems_digit[:, 1:]    # 形状: (24117, 49)

print("X示例", "\t", "Y示例")
for i in range(10):
    print(X[0][i], "\t", Y[0][i])

# One-Hot编码Y
# [旧写法] from keras.utils import to_categorical
# [新写法] 适用于 TensorFlow >= 2.0
from tensorflow.keras.utils import to_categorical
Y = to_categorical(Y, num_classes=vocab_size)
print(Y.shape)  # (24117, 49, vocab_size)
```

**代码解析**：

1. **构造X和Y——向后错一位**：`X = poems_digit[:, :-1]` 取每首诗的前49个字作为输入，`Y = poems_digit[:, 1:]` 取后49个字作为目标。Y相对于X偏移了一个位置，这样模型在每个时间步学习的都是"给定当前及之前的字，预测下一个字"。
2. **形状解释**：X和Y的形状都是 `(24117, 49)`——24117首诗，每首49个时间步。
3. **One-Hot编码Y**：`to_categorical(Y, num_classes=vocab_size)` 将Y中的每个整数索引转为One-Hot向量。例如，如果 `vocab_size=4501`，索引3会被转为一个4501维的向量 `[0,0,0,1,0,...,0]`。
4. **Y的最终形状**：`(24117, 49, vocab_size)` 表示24117首诗，每首49个时间步，每个时间步是一个vocab_size维的One-Hot向量。这是因为我们使用 `categorical_crossentropy` 损失函数，它要求目标是One-Hot格式。

### 步骤4：构建LSTM模型

```python
# [旧写法] from keras.layers import Input, LSTM, Dense, Embedding, Activation
# [旧写法] from keras import Model
# ↑ 独立 keras 包在 TF 2.0+ 中已不推荐使用

# [新写法] 适用于 TensorFlow >= 2.0
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding, Activation
from tensorflow.keras import Model

hidden_size1 = 128   # Embedding维度
hidden_size2 = 64    # LSTM隐藏状态维度

inp = Input(shape=(49,))

# Embedding层：将字索引映射为dense向量
x = Embedding(vocab_size, hidden_size1, input_length=49, mask_zero=True)(inp)

# LSTM层：处理序列
x = LSTM(hidden_size2, return_sequences=True)(x)
# return_sequences=True 表示输出每个时间步的结果（用于序列到序列）

# 全连接 + Softmax：预测下一个字的概率分布
x = Dense(vocab_size)(x)
pred = Activation('softmax')(x)

model = Model(inp, pred)
model.summary()
```

**代码解析**：

1. **Input层**：`Input(shape=(49,))` 定义输入形状为长度49的整数序列（每首诗49个时间步）。
2. **Embedding层**：`Embedding(vocab_size, 128, input_length=49, mask_zero=True)`
   - 这是一个大小为 `(vocab_size, 128)` 的查找表（矩阵）。给定字的索引，直接查找对应行，返回128维向量。
   - `mask_zero=True`：告诉模型索引0是padding，应该被忽略。这样填充的0不会对模型的学习产生干扰。
   - 输出形状：`(batch_size, 49, 128)`——每个时间步的字索引被映射为128维向量。
3. **LSTM层**：`LSTM(64, return_sequences=True)`
   - `64`：隐藏状态的维度，即LSTM在每个时间步输出一个64维的向量。隐藏维度越大，模型的表达能力越强，但计算量和参数量也越大。
   - `return_sequences=True`：在**每个时间步**都输出隐藏状态。因为我们需要在每个位置都预测下一个字（序列到序列任务），所以必须设为True。
   - 输出形状：`(batch_size, 49, 64)`——49个时间步，每步输出64维。
4. **Dense层**：`Dense(vocab_size)` 全连接层，将LSTM输出的64维向量映射为vocab_size维的**分数（logits）**。每个维度对应词汇表中一个字的"得分"，得分越高表示该字越可能是下一个字。
5. **Softmax激活**：`Activation('softmax')` 将logits转为概率分布，所有值在0到1之间且总和为1。例如输出 `[0.01, 0.02, 0.85, 0.03, ...]` 表示索引2对应的字有85%的概率是下一个字。

### 步骤5：编译与训练

```python
# [旧写法] from keras.optimizers import Adam
# [新写法] 适用于 TensorFlow >= 2.0
from tensorflow.keras.optimizers import Adam

# [旧写法] model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.01), metrics=['accuracy'])
# ↑ 参数 lr 在 TF 2.11+ 中已废弃
# [新写法] 适用于 TensorFlow >= 2.11
model.compile(
    loss='categorical_crossentropy',
    optimizer=Adam(learning_rate=0.01),
    metrics=['accuracy']
)

model.fit(X, Y, epochs=10, batch_size=128, validation_split=0.2)
```

**代码解析**：

1. **损失函数 `categorical_crossentropy`**：用于多分类任务，衡量模型预测的概率分布与真实One-Hot标签之间的差距。当目标Y是One-Hot编码时使用此损失函数。如果Y保持为整数索引格式（不做 `to_categorical`），则应使用 `sparse_categorical_crossentropy`，效果相同但节省内存。
2. **优化器 `Adam(learning_rate=0.01)`**：Adam是一种自适应学习率优化器，综合了动量（Momentum）和自适应学习率（RMSProp）的优点。`learning_rate=0.01` 是学习率，控制每次参数更新的步长。学习率过大可能导致训练不稳定（损失震荡），过小则训练速度慢。0.01对于本任务来说是较大的学习率，可以加快收敛。
3. **训练参数**：
   - `epochs=10`：整个训练集遍历10遍。
   - `batch_size=128`：每次用128首诗计算梯度并更新参数。较大的batch_size训练更稳定但需要更多内存。
   - `validation_split=0.2`：留出20%的数据作为验证集，用于监控模型是否过拟合。

### 步骤6：生成古诗

```python
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

poem_incomplete = '熊****大****很****帅****'
poem_index = []
poem_text = ''

for i in range(len(poem_incomplete)):
    current_word = poem_incomplete[i]

    if current_word != '*':
        index = tokenizer.word_index[current_word]
    else:
        x = np.expand_dims(poem_index, axis=0)
        x = pad_sequences(x, maxlen=49, padding='post')
        y = model.predict(x)[0, i]
        y[0] = 0            # 去掉停止词
        index = y.argmax()
        current_word = tokenizer.index_word[index]

    poem_index.append(index)
    poem_text = poem_text + current_word

# 打印五言绝句
print(poem_text[0:5])
print(poem_text[5:10])
print(poem_text[10:15])
print(poem_text[15:20])
```

**代码解析**：

1. **生成策略——贪心解码（Greedy Decoding）**：本代码采用的是**贪心解码**策略，即在每个位置直接选择概率最高的字。虽然简单高效，但可能导致生成结果缺乏多样性（每次生成都一样）。
2. **模板机制**：`poem_incomplete = '熊****大****很****帅****'` 中，已知字（如"熊"）直接使用，`*` 号位置由模型预测。这允许用户控制诗的部分内容（如每句首字），其余由模型填充。
3. **逐字生成过程**：
   - 遇到已知字：查找其索引 `tokenizer.word_index[current_word]`，加入已生成序列。
   - 遇到 `*`：将已生成序列补全到长度49并送入模型，模型输出每个位置的概率分布。取当前位置 `i` 的概率向量 `y = model.predict(x)[0, i]`。
4. **`y[0] = 0` 的含义**：将索引0（padding符号）的概率置为0。因为索引0是填充用的，不是一个真实的字，我们不希望模型"预测"出一个填充符号。
5. **`y.argmax()` 选最大概率**：返回概率最高的字的索引，即贪心地选择最可能的下一个字。
6. **改进方向**：可以用**温度采样（Temperature Sampling）**增加多样性——不总是选最大概率的字，而是按概率分布随机采样。温度越高（如T=1.5），生成越随机；温度越低（如T=0.3），越趋向贪心。还可以用**Beam Search**同时保留多个候选序列，最终选最优的。

## 五、SimpleRNN vs LSTM 对比

| 特征 | SimpleRNN | LSTM |
|------|-----------|------|
| 结构 | 单个循环单元 | 遗忘门+输入门+输出门+细胞状态 |
| 长距离依赖 | 难以捕捉 | 有效捕捉 |
| 梯度问题 | 容易梯度消失 | 通过门控缓解 |
| 参数量 | 较少 | 约4倍于SimpleRNN |
| 适用场景 | 短序列 | 长序列、复杂依赖 |

## 六、新旧写法对照表

| 功能 | 旧写法 | 新写法 |
|------|--------|--------|
| 导入LSTM | `from keras.layers import LSTM` | `from tensorflow.keras.layers import LSTM` |
| 导入Tokenizer | `from keras.preprocessing.text import Tokenizer` | `from tensorflow.keras.preprocessing.text import Tokenizer` |
| 导入pad_sequences | `from keras.preprocessing.sequence import pad_sequences` | `from tensorflow.keras.preprocessing.sequence import pad_sequences` |
| 导入Embedding | `from keras.layers import Embedding` | `from tensorflow.keras.layers import Embedding` |
| 学习率 | `Adam(lr=0.01)` | `Adam(learning_rate=0.01)` |
| NumPy整型 | `np.int` | `np.int64`（NumPy >= 1.20） |

## 七、思考题

1. LSTM的三个门各自起什么作用？如果去掉遗忘门会怎样？
2. Embedding层的作用是什么？为什么不直接用One-Hot编码作为输入？
3. `mask_zero=True`的作用是什么？为什么需要它？
4. `return_sequences=True`和`return_sequences=False`有什么区别？
5. 如何改进诗歌生成的质量？（提示：考虑温度采样、Beam Search等）
