# 深度学习环境搭建教程

本教程提供两种方案搭建深度学习环境：
- **方案一**：纯 Windows + Conda（无需 WSL，适合不想装 WSL 的用户）
- **方案二**：Windows + WSL2 + Conda（推荐，Linux 生态兼容性更好）

每种方案都包含 **GPU 版** 和 **CPU 版** 的安装说明，没有 NVIDIA 显卡的用户也可以正常使用。

---

## 方案一：纯 Windows 环境下使用 Conda 安装 TensorFlow & PyTorch

> 适用于：Windows 10/11，不使用 WSL。支持 GPU 和纯 CPU 两种模式。

### 前置条件

1. **Miniconda / Anaconda**：已安装。若未安装，从 [Miniconda 官网](https://docs.anaconda.com/miniconda/) 下载 Windows 64-bit 安装包并安装。
2. **NVIDIA 驱动**（仅 GPU 用户）：安装最新版 NVIDIA 驱动（≥ 535+），在 CMD 中运行 `nvidia-smi` 确认显卡被识别。没有 NVIDIA 显卡的用户跳过此步，直接按 CPU 版安装。
3. 安装完成后，使用 **Anaconda Prompt**（开始菜单搜索）执行以下所有命令。

### Step 1：配置国内镜像源（加速下载）

#### 配置 Conda 镜像（清华源）

打开 Anaconda Prompt，执行：

```bash
conda config --set show_channel_urls yes
```

然后用记事本编辑 `C:\Users\你的用户名\.condarc`（或在 Prompt 中执行 `notepad %USERPROFILE%\.condarc`），替换为以下内容：

```yaml
channels:
  - defaults
show_channel_urls: true
default_channels:
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/msys2
custom_channels:
  conda-forge: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  pytorch: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  pytorch-lts: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
```

清理缓存：

```bash
conda clean -i
```

#### 配置 pip 镜像（清华源）

```bash
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
pip config set global.trusted-host pypi.tuna.tsinghua.edu.cn
```

---

### Step 2：创建 TensorFlow 环境

#### GPU 版（有 NVIDIA 显卡）

```bash
conda create -n tf-gpu python=3.11 -y
conda activate tf-gpu
pip install --upgrade pip setuptools wheel
pip install tensorflow[and-cuda]
```

> **说明**：`tensorflow[and-cuda]` 会自动安装兼容版本的 CUDA runtime 和 cuDNN，无需手动安装 CUDA Toolkit。

验证：

```bash
python -c "import tensorflow as tf; print('TF版本:', tf.__version__); print('GPU:', tf.config.list_physical_devices('GPU'))"
```

预期：看到版本号和 GPU 设备列表（非空 `[]`）。

#### CPU 版（无 NVIDIA 显卡）

```bash
conda create -n tf-cpu python=3.11 -y
conda activate tf-cpu
pip install --upgrade pip setuptools wheel
pip install tensorflow
```

> **说明**：直接 `pip install tensorflow`（不带 `[and-cuda]`）即为 CPU 版，不需要任何显卡驱动。

验证：

```bash
python -c "import tensorflow as tf; print('TF版本:', tf.__version__); print('设备:', tf.config.list_physical_devices())"
```

预期：看到版本号和 `[PhysicalDevice(name='/physical_device:CPU:0', device_type='CPU')]`。

---

### Step 3：创建 PyTorch 环境

#### GPU 版（有 NVIDIA 显卡，CUDA 12.4）

```bash
conda create -n pytorch-gpu python=3.11 -y
conda activate pytorch-gpu
pip install --upgrade pip setuptools wheel
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

> **注意**：PyTorch 的 CUDA 版本需与你的 NVIDIA 驱动兼容。`nvidia-smi` 显示的 CUDA 版本是驱动支持的最高版本，安装的 PyTorch CUDA 版本须 ≤ 该版本。如果驱动较旧，可选择 `cu118` 或 `cu121`。

访问 [PyTorch 官网 Get Started](https://pytorch.org/get-started/locally/) 可获取最新安装命令。

验证：

```bash
python -c "import torch; print('PyTorch版本:', torch.__version__); print('CUDA可用:', torch.cuda.is_available()); print('GPU名称:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
```

预期：CUDA 可用为 `True`，并显示 GPU 名称。

#### CPU 版（无 NVIDIA 显卡）

```bash
conda create -n pytorch-cpu python=3.11 -y
conda activate pytorch-cpu
pip install --upgrade pip setuptools wheel
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

> **说明**：使用 `--index-url https://download.pytorch.org/whl/cpu` 安装 CPU 专用版本，体积更小，不包含 CUDA 组件。

验证：

```bash
python -c "import torch; print('PyTorch版本:', torch.__version__); print('CUDA可用:', torch.cuda.is_available())"
```

预期：版本号正常显示，CUDA 可用为 `False`（CPU 版预期行为）。

---

### Step 4：在 IDE 中使用环境

#### VS Code
1. 安装 Python 扩展。
2. `Ctrl+Shift+P` → 输入 `Python: Select Interpreter`。
3. 选择对应的 Conda 环境（`tf-gpu` / `tf-cpu` / `pytorch-gpu` / `pytorch-cpu`）。

#### PyCharm
1. File → Settings → Project → Python Interpreter → Add Interpreter。
2. 选择 **Conda Environment** → **Existing environment**。
3. Interpreter 路径示例：
   - `C:\Users\你的用户名\miniconda3\envs\tf-gpu\python.exe`
   - `C:\Users\你的用户名\miniconda3\envs\pytorch-cpu\python.exe`

#### Jupyter Notebook
在对应环境中安装并注册 kernel：

```bash
# 示例：注册所有环境（按需选择）
conda activate tf-gpu
pip install ipykernel
python -m ipykernel install --user --name tf-gpu --display-name "TensorFlow GPU"

conda activate tf-cpu
pip install ipykernel
python -m ipykernel install --user --name tf-cpu --display-name "TensorFlow CPU"

conda activate pytorch-gpu
pip install ipykernel
python -m ipykernel install --user --name pytorch-gpu --display-name "PyTorch GPU"

conda activate pytorch-cpu
pip install ipykernel
python -m ipykernel install --user --name pytorch-cpu --display-name "PyTorch CPU"
```

然后在 Jupyter 中切换 Kernel 即可使用对应环境。

---

### 常见问题

| 问题 | 解决办法 |
|------|----------|
| `nvidia-smi` 无法识别 | 更新 NVIDIA 驱动至最新版；无显卡用户请使用 CPU 版 |
| TensorFlow GPU 列表为空 `[]` | 确认驱动版本 ≥ 535；尝试 `pip install tensorflow[and-cuda]` 重装 |
| PyTorch `torch.cuda.is_available()` 返回 `False` | GPU 版：确认安装了 CUDA 版（非 CPU 版），检查 CUDA 版本兼容性；CPU 版：`False` 是正常的 |
| Conda 安装/下载很慢 | 确认已配置清华镜像；`conda clean --all` 后重试 |
| `DLL load failed` 错误 | 安装 [Visual C++ Redistributable](https://aka.ms/vs/17/release/vc_redist.x64.exe) |
| 环境冲突 | TensorFlow 和 PyTorch 建议分开装在不同 Conda 环境中，避免依赖冲突 |
| CPU 训练太慢 | CPU 版适合学习和小规模实验；大规模训练建议使用 GPU 或云服务（如 Google Colab） |
## 方案二：Windows + WSL2 环境下使用 Conda 安装 TensorFlow & PyTorch

> 适用于：Windows 11 用户（已开启 WSL2 + Ubuntu），希望在 Linux 环境中运行深度学习框架。支持 GPU 和纯 CPU 两种模式。
>
> **推荐理由**：TensorFlow/PyTorch 在 Linux 上兼容性最好，官方文档和社区方案均以 Linux 为主；WSL2 支持 GPU passthrough，兼顾 Windows 桌面体验。即使没有 NVIDIA 显卡，WSL2 的 Linux 环境依然有优势。

### 前置条件

| 序号 | 条件 | 说明 |
|------|------|------|
| 1 | **WSL2 已开启** | PowerShell（管理员）运行 `wsl --install`，确认已装 Ubuntu 22.04/24.04 LTS |
| 2 | **NVIDIA 驱动**（仅 GPU 用户） | Windows 侧安装最新版（≥ 535+），WSL 终端运行 `nvidia-smi` 看到显卡信息。无显卡用户跳过 |
| 3 | **IDE**（可选） | PyCharm Professional 支持 WSL 解释器；VS Code 免费且无限制 |
| 4 | **项目路径** | 建议放在 `/home/你的用户名/` 下，避免 `/mnt/c` 跨文件系统性能问题 |

> **提示**：在 Windows 文件管理器地址栏输入 `\\wsl$\Ubuntu\home\你的用户名` 即可访问 WSL 文件。

---

### 第一部分：安装 Miniconda + 配置国内镜像

打开 WSL Ubuntu 终端，依次执行：

#### 1. 更新系统

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install wget curl git -y
```

#### 2. 安装 Miniconda

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda
```

关闭终端 → 重新打开（或执行 `source ~/.bashrc`）。

#### 3. 初始化 Conda

```bash
~/miniconda/bin/conda init
```

关闭 → 重新打开终端，看到提示符前出现 `(base)` 即成功。

#### 4. 配置清华镜像源

**Conda 镜像**（编辑 `~/.condarc`）：

```bash
conda config --set show_channel_urls yes
nano ~/.condarc
```

覆盖粘贴以下内容（保存：`Ctrl+O` → 回车 → `Ctrl+X`）：

```yaml
channels:
  - defaults
show_channel_urls: true
default_channels:
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/msys2
custom_channels:
  conda-forge: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  pytorch: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  pytorch-lts: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
```

清理缓存：

```bash
conda clean -i
```

**pip 镜像**（创建 `~/.pip/pip.conf`）：

```bash
mkdir -p ~/.pip
cat > ~/.pip/pip.conf << 'EOF'
[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
trusted-host = pypi.tuna.tsinghua.edu.cn
EOF
```

---

### 第二部分：安装 TensorFlow 环境

#### GPU 版（有 NVIDIA 显卡）

```bash
conda create -n tf-gpu python=3.11 -y
conda activate tf-gpu
pip install --upgrade pip setuptools wheel
pip install tensorflow[and-cuda]
```

> `tensorflow[and-cuda]` 会自动安装兼容的 CUDA runtime + cuDNN，无需手动安装 CUDA Toolkit。

验证：

```bash
python -c "import tensorflow as tf; print('TF版本:', tf.__version__); print('GPU:', tf.config.list_physical_devices('GPU'))"
```

预期：看到版本号和 GPU 列表（非空 `[]`）。

#### CPU 版（无 NVIDIA 显卡）

```bash
conda create -n tf-cpu python=3.11 -y
conda activate tf-cpu
pip install --upgrade pip setuptools wheel
pip install tensorflow
```

> 直接 `pip install tensorflow`（不带 `[and-cuda]`）即为 CPU 版，不需要任何显卡驱动。

验证：

```bash
python -c "import tensorflow as tf; print('TF版本:', tf.__version__); print('设备:', tf.config.list_physical_devices())"
```

预期：看到版本号和 `[PhysicalDevice(name='/physical_device:CPU:0', device_type='CPU')]`。

---

### 第三部分：安装 PyTorch 环境

#### GPU 版（有 NVIDIA 显卡，CUDA 12.4）

```bash
conda create -n pytorch-gpu python=3.11 -y
conda activate pytorch-gpu
pip install --upgrade pip setuptools wheel
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

> `nvidia-smi` 显示的 CUDA 版本是驱动支持的上限，PyTorch 的 CUDA 版本须 ≤ 该版本。驱动较旧可选 `cu118` 或 `cu121`。
>
> 访问 [PyTorch 官网](https://pytorch.org/get-started/locally/) 获取最新安装命令。

验证：

```bash
python -c "import torch; print('PyTorch版本:', torch.__version__); print('CUDA可用:', torch.cuda.is_available()); print('GPU名称:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
```

#### CPU 版（无 NVIDIA 显卡）

```bash
conda create -n pytorch-cpu python=3.11 -y
conda activate pytorch-cpu
pip install --upgrade pip setuptools wheel
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

> 使用 `--index-url https://download.pytorch.org/whl/cpu` 安装 CPU 专用版本，体积更小，不包含 CUDA 组件。

验证：

```bash
python -c "import torch; print('PyTorch版本:', torch.__version__); print('CUDA可用:', torch.cuda.is_available())"
```

预期：版本号正常显示，CUDA 可用为 `False`（CPU 版预期行为）。

---

### 第四部分：在 IDE 中使用 WSL Conda 环境

#### VS Code（推荐）
1. 安装 **Remote - WSL** 扩展。
2. 左下角点击远程连接图标 → Connect to WSL。
3. 在 WSL 中打开项目文件夹，VS Code 会自动检测 Conda 环境。
4. `Ctrl+Shift+P` → `Python: Select Interpreter` → 选择对应环境。

#### PyCharm Professional
1. File → Settings → Project → Python Interpreter → Add Interpreter。
2. 选择 **On WSL** → 选择 Ubuntu 发行版 → Next。
3. 选择 **Conda Environment** → **Existing environment**。
4. Interpreter 路径示例：
   - `/home/你的用户名/miniconda/envs/tf-gpu/bin/python`
   - `/home/你的用户名/miniconda/envs/pytorch-cpu/bin/python`
5. 如果 Conda 选项未出现，手动指定 Conda 可执行文件：`/home/你的用户名/miniconda/bin/conda`。

#### Jupyter Notebook（按需注册 kernel）

```bash
# GPU 环境
conda activate tf-gpu
pip install ipykernel
python -m ipykernel install --user --name tf-gpu --display-name "TensorFlow GPU (WSL)"

conda activate pytorch-gpu
pip install ipykernel
python -m ipykernel install --user --name pytorch-gpu --display-name "PyTorch GPU (WSL)"

# CPU 环境
conda activate tf-cpu
pip install ipykernel
python -m ipykernel install --user --name tf-cpu --display-name "TensorFlow CPU (WSL)"

conda activate pytorch-cpu
pip install ipykernel
python -m ipykernel install --user --name pytorch-cpu --display-name "PyTorch CPU (WSL)"
```

---

### 常见问题速查表

| 问题 | 解决办法 |
|------|----------|
| GPU 列表为空 `[]` | 检查 `nvidia-smi` → `wsl --shutdown` 重启 WSL → 重装 `tensorflow[and-cuda]` |
| `nvidia-smi` 在 WSL 失败 | 重装 Windows 侧 NVIDIA 驱动（≥ 535）；确认 WSL2（非 WSL1）；无显卡用户请用 CPU 版 |
| PyCharm 无 "On WSL" 选项 | 确认使用 Professional 版 → 更新至最新版 |
| Conda 路径找不到 | 激活环境后用 `which python` 查路径，手动输入 |
| 下载/安装很慢 | 确认镜像已配 → `conda clean --all` 或 `pip cache purge` 后重试 |
| 索引/保存卡顿 | 项目放 WSL `/home` 内，不要放 `/mnt/c` |
| PyTorch CUDA 不可用 | GPU 版：确认安装了 CUDA 版，检查 CUDA 版本兼容性；CPU 版：`False` 是正常的 |
| CPU 训练太慢 | CPU 版适合学习和小规模实验；大规模训练建议使用 GPU 或云服务（如 Google Colab） |

---

以上两套方案完成后，你就拥有了完整的深度学习开发环境。**建议 TensorFlow 和 PyTorch 始终放在独立的 Conda 环境中**，避免依赖冲突。无论是否有 GPU，都可以正常学习和开发。