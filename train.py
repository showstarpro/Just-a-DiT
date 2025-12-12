from dit import DiT
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torchvision
from torchvision import transforms as T
from torchvision.utils import make_grid, save_image
from tqdm import tqdm
from ema import LitEma
# from bitsandbytes.optim import AdamW8bit

from model import RectifiedFlow
from fid_evaluation import FIDEvaluation

import moviepy.editor as mpy
import wandb

import os
import argparse
from datetime import datetime

def setup_ddp():
    """初始化 DDP 环境"""
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank, dist.get_rank(), dist.get_world_size()

def cleanup_ddp():
    """清理 DDP 环境"""
    dist.destroy_process_group()

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='DiT Training with DDP')
    parser.add_argument('--num_register_tokens', type=int, default=0,
                        help='Number of register tokens (default: 0)')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size per GPU (default: 128)')
    parser.add_argument('--n_steps', type=int, default=200000,
                        help='Number of training steps (default: 200000)')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate (default: 1e-4)')
    return parser.parse_args()

def main():
    # 解析命令行参数
    args = parse_args()

    # 初始化 DDP
    local_rank, rank, world_size = setup_ddp()
    is_main_process = rank == 0

    # 从命令行参数获取配置
    n_steps = args.n_steps
    device = f"cuda:{local_rank}"
    batch_size = args.batch_size  # 每个 GPU 的 batch size

    if is_main_process:
        print(f"Training configuration:")
        print(f"  - Number of GPUs: {world_size}")
        print(f"  - Batch size per GPU: {batch_size}")
        print(f"  - Total batch size: {batch_size * world_size}")
        print(f"  - Number of register tokens: {args.num_register_tokens}")
        print(f"  - Learning rate: {args.lr}")
        print(f"  - Training steps: {n_steps}")
        print()

    dataset = torchvision.datasets.CIFAR10(
        root="/mnt/volumes/so-volume-bd-ga/lhp/code/Just-a-DiT/data/CIFAR10",
        train=True,
        download=True,
        transform=T.Compose([T.ToTensor(), T.RandomHorizontalFlip()]),
    )

    # 使用 DistributedSampler 确保每个 GPU 处理不同的数据
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        drop_last=True
    )

    def cycle(iterable):
        while True:
            for i in iterable:
                yield i

    train_dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=train_sampler,  # 使用 sampler 而不是 shuffle
        drop_last=True,
        num_workers=8,
        pin_memory=True
    )
    train_dataloader = cycle(train_dataloader)

    model = DiT(
        input_size=32,
        patch_size=2,
        in_channels=3,
        dim=384,
        depth=12,
        num_heads=6,
        num_classes=10,
        learn_sigma=False,
        class_dropout_prob=0.1,
        num_register_tokens=args.num_register_tokens,
    ).to(device)

    # 用 DDP 包装模型
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    # EMA 需要访问原始模型（不带 DDP 包装）
    model_ema = LitEma(model.module)

    # 使用标准 PyTorch AdamW 替代 AdamW8bit 以避免 bitsandbytes CUDA 兼容性问题
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    
    # RectifiedFlow 需要使用原始模型
    sampler = RectifiedFlow(model.module, device=device)
    scaler = torch.cuda.amp.GradScaler()

    # 只在主进程初始化 wandb 和 FID 评估
    if is_main_process:
        # 设置 wandb 为离线模式
        os.environ["WANDB_MODE"] = "offline"
        logger = wandb.init(project="dit-cfm")
        fid_eval = FIDEvaluation(batch_size * 2, train_dataloader, sampler)
    else:
        logger = None
        fid_eval = None
    
    def sample_and_log_images():
        log_imgs = []
        log_gifs = []
        for cfg_scale in [1.0, 2.5, 5.0]:
            print(
                f"Sampling images at step {step} with cfg_scale {cfg_scale}..."
            )
            final_imgs, traj = sampler.sample_each_class(10, cfg_scale=cfg_scale, return_all_steps=True)
            log_img = make_grid(final_imgs, nrow=10)
            img_save_path = f"images/step{step}_cfg{cfg_scale}.png"
            save_image(log_img, img_save_path)
            log_imgs.append(
                wandb.Image(img_save_path, caption=f"cfg_scale: {cfg_scale}")
            )
            # print(f"Saved images to {img_save_path}")
            images_list = [
                make_grid(frame, nrow=10).permute(1, 2, 0).cpu().numpy() * 255
                for frame in traj
            ]
            clip = mpy.ImageSequenceClip(images_list, fps=10)
            clip.write_gif(f"images/step{step}_cfg{cfg_scale}.gif")
            log_gifs.append(
                wandb.Video(
                    f"images/step{step}_cfg{cfg_scale}.gif",
                    caption=f"cfg_scale: {cfg_scale}",
                )
            )

            print("Copying EMA to model...")
            model_ema.store(model.module.parameters())
            model_ema.copy_to(model.module)
            print(
                f"Sampling images with ema model at step {step} with cfg_scale {cfg_scale}..."
            )
            final_imgs_ema, traj_ema = sampler.sample_each_class(10, cfg_scale=cfg_scale, return_all_steps=True)
            log_img = make_grid(final_imgs_ema, nrow=10)
            img_save_path = f"images/step{step}_cfg{cfg_scale}_ema.png"
            save_image(log_img, img_save_path)
            # print(f"Saved images to {img_save_path}")
            log_imgs.append(
                wandb.Image(
                    img_save_path, caption=f"EMA with cfg_scale: {cfg_scale}"
                )
            )

            images_list = [
                make_grid(frame, nrow=10).permute(1, 2, 0).cpu().numpy() * 255
                for frame in traj_ema
            ]
            clip = mpy.ImageSequenceClip(images_list, fps=10)
            clip.write_gif(f"images/step{step}_cfg{cfg_scale}_ema.gif")
            log_gifs.append(
                wandb.Video(
                    f"images/step{step}_cfg{cfg_scale}_ema.gif",
                    caption=f"EMA with cfg_scale: {cfg_scale}",
                )
            )
            model_ema.restore(model.module.parameters())
        if is_main_process:
            logger.log({"Images": log_imgs, "Gifs": log_gifs, "step": step})

    losses = []
    if is_main_process:
        pbar = tqdm(range(n_steps), dynamic_ncols=True)
        pbar.set_description("Training")
    else:
        pbar = range(n_steps)

    for step in pbar:
        # 每个 epoch 设置不同的随机种子
        train_sampler.set_epoch(step)

        data = next(train_dataloader)
        optimizer.zero_grad()
        x1 = data[0].to(device)
        y = data[1].to(device)
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            loss = sampler(x1, y)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        model_ema(model.module)

        if not torch.isnan(loss):
            losses.append(loss.item())
            if is_main_process:
                pbar.set_postfix({"loss": loss.item()})
                logger.log({"loss": loss.item(), "step": step})


        # 只在主进程进行采样和评估
        if step % 10000 == 0 or step == n_steps - 1:
            if is_main_process:
                print(
                    f"Step: {step+1}/{n_steps} | loss: {sum(losses) / len(losses):.4f}"
                )
                losses.clear()
                model.module.eval()
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    sample_and_log_images()
                model.module.train()
            # 同步所有进程
            dist.barrier()

        if step % 50000 == 0 or step == n_steps - 1:
            if is_main_process:
                model.module.eval()
                model_ema.store(model.module.parameters())
                model_ema.copy_to(model.module)

                # FID 评估不使用 autocast，因为 InceptionV3 和 NumPy 不支持 bfloat16
                fid_score = fid_eval.fid_score()
                print(f"FID score with EMA at step {step}: {fid_score}")

                model_ema.restore(model.module.parameters())
                model.module.train()

                wandb.log({"FID": fid_score, "step": step})
            # 同步所有进程
            dist.barrier()

    # 只在主进程保存模型
    if is_main_process:
        state_dict = {
            "model": model.module.state_dict(),
            "ema": model_ema.state_dict(),
        }
        # 创建带时间戳和参数信息的模型文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"model_reg{args.num_register_tokens}_bs{batch_size * world_size}_lr{args.lr}_{timestamp}.pth"
        save_path = os.path.join("/lpai/output/models", model_name)
        torch.save(state_dict, save_path)
        print(f"Model saved to {save_path}")

    # 清理 DDP
    cleanup_ddp()


if __name__ == "__main__":
    main()
