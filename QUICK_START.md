# 快速开始指南

## 基本用法

### 1. 单卡训练（默认参数）
```bash
bash run_single_gpu.sh
```

### 2. 四卡训练（默认参数）
```bash
bash run_ddp.sh
```

### 3. 使用 Register Tokens

#### 单卡训练，4个 register tokens
```bash
bash run_single_gpu.sh --num_register_tokens 4
```

#### 四卡训练，8个 register tokens
```bash
bash run_ddp.sh --num_register_tokens 8
```

### 4. 自定义多个参数

```bash
# 单卡训练，自定义多个参数
bash run_single_gpu.sh \
    --num_register_tokens 4 \
    --batch_size 256 \
    --lr 2e-4 \
    --n_steps 100000

# 四卡训练，自定义多个参数
bash run_ddp.sh \
    --num_register_tokens 8 \
    --batch_size 64 \
    --lr 5e-5 \
    --n_steps 150000
```

## 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--num_register_tokens` | 0 | Register tokens 数量 |
| `--batch_size` | 128 | 每个 GPU 的 batch size |
| `--n_steps` | 200000 | 训练步数 |
| `--lr` | 1e-4 | 学习率 |

## 常见配置

### 配置 1: 基础训练（无 register tokens）
```bash
bash run_ddp.sh
```

### 配置 2: 标准 register tokens 训练
```bash
bash run_ddp.sh --num_register_tokens 4
```

### 配置 3: 更多 register tokens
```bash
bash run_ddp.sh --num_register_tokens 8
```

### 配置 4: 小 batch size（显存不足时）
```bash
bash run_ddp.sh --num_register_tokens 4 --batch_size 64
```

### 配置 5: 快速实验（少量步数）
```bash
bash run_single_gpu.sh --num_register_tokens 4 --n_steps 10000
```

## 查看帮助

```bash
python train.py --help
```

## 显存占用估算

基于 CIFAR-10 (32x32) 训练的估算：

| 配置 | 每GPU Batch Size | Register Tokens | 预计显存 |
|------|------------------|-----------------|----------|
| 基础 | 128 | 0 | ~8 GB |
| 标准 | 128 | 4 | ~9 GB |
| 增强 | 128 | 8 | ~10 GB |
| 大量 | 128 | 16 | ~12 GB |
| 显存优化 | 64 | 4 | ~5 GB |

## 注意事项

1. **总 batch size** = batch_size × GPU数量
   - 单卡: batch_size = 128 → 总 batch size = 128
   - 4卡: batch_size = 128 → 总 batch size = 512

2. **Wandb 默认离线模式**
   - 日志保存在 `wandb/` 目录
   - 使用 `wandb sync wandb/offline-run-<id>` 同步到云端

3. **中断恢复**
   - 模型自动保存为 `model.pth`
   - 需要手动实现检查点加载（当前版本未实现）

4. **更改 GPU**
   - 编辑 `run_ddp.sh` 或 `run_single_gpu.sh` 中的 `CUDA_VISIBLE_DEVICES`
