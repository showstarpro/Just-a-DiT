# DDP 多卡训练使用说明

代码已修改为支持 PyTorch DistributedDataParallel (DDP) 多卡训练。

## 主要修改

1. **添加 DDP 支持**
   - 使用 `torch.distributed` 初始化分布式环境
   - 使用 `DistributedSampler` 确保每个 GPU 处理不同的数据
   - 使用 `DDP` 包装模型进行分布式训练

2. **关键特性**
   - 只在主进程（rank 0）进行日志记录、采样和保存模型
   - 使用 `dist.barrier()` 同步所有进程
   - 每个 GPU 的 batch size 独立设置（总 batch size = batch_size × GPU 数量）
   - **wandb 默认为离线模式**，数据保存在本地 `wandb/` 目录

## 使用方法

### 1. 多卡训练（推荐）

**基本用法**（使用默认参数）:

```bash
bash run_ddp.sh
```

**自定义参数**:

```bash
# 设置 register tokens 数量
bash run_ddp.sh --num_register_tokens 4

# 设置多个参数
bash run_ddp.sh --num_register_tokens 8 --batch_size 64 --lr 2e-4

# 查看所有可用参数
torchrun --nproc_per_node=1 train.py --help
```

**可用参数**:
- `--num_register_tokens`: Register tokens 数量（默认: 0）
- `--batch_size`: 每个 GPU 的 batch size（默认: 128）
- `--n_steps`: 训练步数（默认: 200000）
- `--lr`: 学习率（默认: 1e-4）

或者直接使用 torchrun:

```bash
# 使用 4 张 GPU，默认参数
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 --master_port=29500 train.py

# 使用 4 张 GPU，自定义参数
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 --master_port=29500 train.py \
    --num_register_tokens 4 --batch_size 64

# 使用 8 张 GPU
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 --master_port=29500 train.py \
    --num_register_tokens 8
```

### 2. 单卡训练

**基本用法**:

```bash
bash run_single_gpu.sh
```

**自定义参数**:

```bash
# 设置 register tokens 数量
bash run_single_gpu.sh --num_register_tokens 4

# 设置多个参数
bash run_single_gpu.sh --num_register_tokens 8 --batch_size 256 --lr 2e-4
```

或者直接使用 torchrun:

```bash
CUDA_VISIBLE_DEVICES=6 torchrun --nproc_per_node=1 --master_port=29500 train.py \
    --num_register_tokens 4
```

## 训练参数说明

### 命令行参数

所有参数都可以通过命令行指定：

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--num_register_tokens` | int | 0 | Register tokens 数量，用于 DiT 模型 |
| `--batch_size` | int | 128 | 每个 GPU 的 batch size |
| `--n_steps` | int | 200000 | 总训练步数 |
| `--lr` | float | 1e-4 | 学习率 |

### Register Tokens 说明

Register tokens 是一种改进注意力机制的技术，可以帮助模型更好地聚合全局信息。

**推荐设置**:
- 基础训练: `--num_register_tokens 0`（默认）
- 改进性能: `--num_register_tokens 4` 或 `8`
- 更多寄存器: `--num_register_tokens 16`（需要更多显存）

## Batch Size 说明

代码中的 `batch_size = 128` 是**每个 GPU** 的 batch size。

- **单卡训练**: 总 batch size = 128
- **4 卡训练**: 总 batch size = 128 × 4 = 512
- **8 卡训练**: 总 batch size = 128 × 8 = 1024

如果显存不足，可以在 train.py 中调整 `batch_size` 变量。

## 注意事项

1. **端口占用**: 如果出现端口占用错误，修改 `--master_port` 参数（例如 29501, 29502）

2. **显存不足**: 如果出现 OOM 错误，减小每个 GPU 的 batch_size

3. **日志和输出**: 只有主进程（rank 0）会打印日志和进度条，其他进程静默运行

4. **模型保存**: 只有主进程会保存模型文件 `model.pth`

5. **采样和评估**: FID 评估和图像采样只在主进程进行，其他进程会等待（通过 barrier 同步）

## 性能提升

使用 DDP 多卡训练的理论加速比：

- 2 卡: ~1.8x
- 4 卡: ~3.5x
- 8 卡: ~7x

实际加速比取决于模型大小、通信开销等因素。

## Wandb 日志管理

### 离线模式（默认）

代码默认使用离线模式，所有日志保存在本地 `wandb/` 目录中。

查看离线运行记录：
```bash
ls wandb/
```

### 同步到云端（可选）

如果需要将离线日志同步到 wandb 云端：

```bash
# 同步特定的运行
wandb sync wandb/offline-run-<run-id>

# 同步所有离线运行
wandb sync wandb/
```

### 启用在线模式

如果希望直接在线记录，修改 train.py 中的这一行：
```python
os.environ["WANDB_MODE"] = "offline"  # 改为 "online" 或删除这行
```

或者在运行前设置环境变量：
```bash
export WANDB_MODE=online
bash run_ddp.sh
```

## 故障排查

### 问题 1: 进程挂起或卡住

确保所有 GPU 都可用，并且网络配置正确：

```bash
# 检查 GPU 状态
nvidia-smi

# 检查 NCCL 通信（在训练脚本开始前）
export NCCL_DEBUG=INFO
```

### 问题 2: 地址已被使用

修改 master_port:

```bash
torchrun --nproc_per_node=4 --master_port=29501 train.py
```

### 问题 3: 显存不足

减小每个 GPU 的 batch size，在 train.py 中修改：

```python
batch_size = 64  # 从 128 改为 64
```
