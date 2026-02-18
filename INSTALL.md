# nano-vllm 安装教程（WSL + Conda）

本文档记录在 WSL 中安装并验证 `nano-vllm` 的一套可用流程。

## 1. 创建并激活 Conda 虚拟环境（Python 3.12）

```bash
conda create -n nano-vllm python=3.12 -y
conda activate nano-vllm
```

## 2. 安装 PyTorch（CUDA 12.4）

```bash
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu124
```

## 3. 安装 flash-attn（本地 whl）

先从官方 release 下载与当前环境匹配的 wheel 文件：

- 仓库地址：<https://github.com/Dao-AILab/flash-attention/releases>

将下载好的 `.whl` 文件放到 `nano-vllm` 项目根目录，然后执行：

```bash
pip install ./flash_attn-2.8.3+cu12torch2.4cxx11abiFALSE-cp312-cp312-linux_x86_64.whl
```

## 4. 可编辑安装 nano-vllm

在项目根目录执行：

```bash
pip install -e .
```

## 5. 下载并整理模型（Qwen3-0.6B）

### 5.1 在 Windows 执行下载

```powershell
huggingface-cli download Qwen/Qwen3-0.6B
```

### 5.2 在 WSL 整理到固定目录

先创建目标目录：

```bash
mkdir -p ~/huggingface/Qwen3-0.6B
```

再将缓存快照中的模型文件复制/移动到目标目录（把 `xxx` 替换为你的 Windows 用户名和快照哈希）：

```bash
mv /mnt/c/Users/xxx/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/xxx/* \
   ~/huggingface/Qwen3-0.6B/
```

## 6. 验证安装

```bash
python -c "import nanovllm; print('nanovllm ok')"
```

如果输出 `nanovllm ok`，说明安装成功。
