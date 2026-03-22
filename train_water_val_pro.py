import datetime
import os
import random
import argparse
from functools import partial

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim as optim
from torch.utils.data import DataLoader
import csv
import time
import json
from pathlib import Path
from collections import defaultdict

from nets.segformer import SegFormer
from nets.segformer_training import (get_lr_scheduler, set_optimizer_lr,
                                     weights_init, CE_Loss, Dice_loss, Focal_Loss)
from utils.callbacks import EvalCallback, LossHistory
from utils.dataloader import SegmentationDataset, seg_dataset_collate
from utils.utils import (download_weights, seed_everything,
                         show_config, worker_init_fn)


def get_files_from_dir(image_dir, label_dir, label_ext='.png', label_suffix=''):
    """从目录中读取所有有效的图片-标签对"""
    valid_files = []
    image_exts = ['.jpg', '.jpeg', '.png', '.bmp']
    
    if not os.path.exists(image_dir):
        raise ValueError(f"图片目录不存在: {image_dir}")
    if not os.path.exists(label_dir):
        raise ValueError(f"标签目录不存在: {label_dir}")
    
    for fname in os.listdir(image_dir):
        is_image = any(fname.lower().endswith(ext) for ext in image_exts)
        if not is_image:
            continue
            
        name_without_ext = os.path.splitext(fname)[0]
        label_fname = name_without_ext + label_suffix + label_ext
        label_path = os.path.join(label_dir, label_fname)
        
        if os.path.exists(label_path):
            valid_files.append(name_without_ext)
        else:
            print(f"警告: 找不到标签文件 {label_path}，跳过 {fname}")
    
    if len(valid_files) == 0:
        raise ValueError(f"在 {image_dir} 中没有找到有效的图片-标签对！")
    
    valid_files.sort()
    return valid_files


def split_dataset(image_dir, val_split=0.1, seed=42, label_dir=None, 
                  image_exts=['.jpg', '.jpeg', '.png', '.bmp'], 
                  label_ext='.png', label_suffix=''): 
    """自动分割数据集（从同一目录划分）"""
    random.seed(seed)
    
    if label_dir is None:
        label_dir = image_dir
    
    valid_files = get_files_from_dir(image_dir, label_dir, label_ext, label_suffix)
    
    random.shuffle(valid_files)
    val_num = int(len(valid_files) * val_split)
    val_lines = valid_files[:val_num]
    train_lines = valid_files[val_num:]
    
    print(f"\n{'='*50}")
    print(f"数据集自动分割完成:")
    print(f"  总样本数: {len(valid_files)}")
    print(f"  训练集: {len(train_lines)} ({len(train_lines)/len(valid_files)*100:.1f}%)")
    print(f"  验证集: {len(val_lines)} ({len(val_lines)/len(valid_files)*100:.1f}%)")
    print(f"{'='*50}\n")
    
    return train_lines, val_lines


def compute_metrics(pred_mask, true_mask, num_classes):
    """
    计算分割指标：Precision, Recall, F1, IoU（宏平均）
    增加前景类（最后一类）单独指标，避免被背景类平均蒙蔽
    """
    pred_mask = pred_mask.view(-1)
    true_mask = true_mask.view(-1)
    
    metrics_per_class = []
    
    for cls in range(num_classes):
        pred_cls = (pred_mask == cls).float()
        true_cls = (true_mask == cls).float()
        
        tp = (pred_cls * true_cls).sum()
        fp = (pred_cls * (1 - true_cls)).sum()
        fn = ((1 - pred_cls) * true_cls).sum()
        
        precision = tp / (tp + fp + 1e-10)
        recall = tp / (tp + fn + 1e-10)
        f1 = 2 * precision * recall / (precision + recall + 1e-10)
        iou = tp / (tp + fp + fn + 1e-10)
        
        if true_cls.sum() > 0:
            metrics_per_class.append({
                'precision': precision.item(),
                'recall': recall.item(),
                'f1': f1.item(),
                'iou': iou.item(),
                'cls': cls
            })
    
    if not metrics_per_class:
        return {
            'precision': 0, 'recall': 0, 'f1': 0, 'miou': 0,
            'fg_precision': 0, 'fg_recall': 0, 'fg_f1': 0, 'fg_miou': 0
        }
    
    # 宏平均（所有类别）
    macro_metrics = {
        'precision': np.mean([m['precision'] for m in metrics_per_class]),
        'recall': np.mean([m['recall'] for m in metrics_per_class]),
        'f1': np.mean([m['f1'] for m in metrics_per_class]),
        'miou': np.mean([m['iou'] for m in metrics_per_class]),
    }
    
    # 前景类指标（最后一个类别，通常是水域/目标类别）
    fg_metrics = metrics_per_class[-1]
    foreground_metrics = {
        'fg_precision': fg_metrics['precision'],
        'fg_recall': fg_metrics['recall'],
        'fg_f1': fg_metrics['f1'],
        'fg_miou': fg_metrics['iou'],
    }
    
    return {**macro_metrics, **foreground_metrics}


@torch.no_grad()
def evaluate_metrics(model, dataloader, device, num_classes, dice_loss=False, focal_loss=False, cls_weights=None):
    """详细评估模型，返回各项指标和推理时间"""
    model.eval()
    
    all_preds = []
    all_targets = []
    total_loss = 0
    num_batches = 0
    inference_times = []
    
    # 确保 cls_weights 是 tensor 并在正确设备上
    if cls_weights is None:
        cls_weights = torch.ones(num_classes, device=device, dtype=torch.float32)
    else:
        if isinstance(cls_weights, np.ndarray):
            cls_weights = torch.from_numpy(cls_weights).float().to(device)
        elif isinstance(cls_weights, torch.Tensor):
            cls_weights = cls_weights.float().to(device)
    
    for batch in dataloader:
        images, labels = batch[0], batch[1]
        images = images.to(device)
        labels = labels.to(device)
        
        # 测量推理时间
        if device.type == 'cuda':
            torch.cuda.synchronize()
        start = time.time()
        
        outputs = model(images)
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        inference_times.append(time.time() - start)
        
        # 计算 loss（SegFormer 特有的混合损失）
        loss = CE_Loss(outputs, labels, cls_weights, num_classes)
        if dice_loss:
            loss += Dice_loss(outputs, labels, cls_weights, num_classes)
        if focal_loss:
            loss += Focal_Loss(outputs, labels, cls_weights, num_classes)
            
        total_loss += loss.item()
        num_batches += 1
        
        # 获取预测结果
        preds = outputs.argmax(dim=1)
        
        all_preds.append(preds.cpu())
        all_targets.append(labels.cpu())
    
    # 计算指标
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    metrics = compute_metrics(all_preds, all_targets, num_classes)
    metrics['loss'] = total_loss / num_batches
    metrics['inference_time_ms'] = np.mean(inference_times) * 1000
    metrics['fps'] = images.size(0) / np.mean(inference_times) if np.mean(inference_times) > 0 else 0
    
    # 诊断：检测是否全背景预测
    if metrics['fg_recall'] == 0 and metrics['recall'] == 0.5:
        print(f"\n[WARNING] 检测到模型预测全为背景（Foreground Recall=0）！")
        print("          这是训练初期的正常现象，建议：")
        print("          1. 增加训练epoch（SegFormer 通常需要20-50轮）")
        print("          2. 或增大学习率（当前可能过小）")
        print("          3. 检查标签是否为0/1格式（而非0/255）\n")
    
    return metrics


def get_model_info(model):
    """获取模型静态信息"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'model_size_mb': total_params * 4 / (1024 * 1024),
    }


class MetricsLogger:
    """指标记录器，支持 CSV 和 TensorBoard，支持实时写入"""
    
    def __init__(self, log_dir, model, input_shape, local_rank=0, log_name=None):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.local_rank = local_rank
        
        # CSV 设置 - 支持自定义文件名
        if log_name:
            self.csv_path = self.log_dir / f'{log_name}.csv'
        else:
            time_str = datetime.datetime.strftime(datetime.datetime.now(), '%Y%m%d_%H%M%S')
            self.csv_path = self.log_dir / f'training_log_{time_str}.csv'
        
        # 修改header，增加前景类指标
        self.header = [
            'epoch',
            'train_loss', 'train_precision', 'train_recall', 'train_f1', 'train_miou',
            'train_fg_precision', 'train_fg_recall', 'train_fg_f1', 'train_fg_miou',
            'val_loss', 'val_precision', 'val_recall', 'val_f1', 'val_miou',
            'val_fg_precision', 'val_fg_recall', 'val_fg_f1', 'val_fg_miou',
            'inference_time_ms', 'fps', 'learning_rate'
        ]
        
        if local_rank == 0:
            with open(self.csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=self.header)
                writer.writeheader()
            print(f"[INFO] 日志文件创建: {self.csv_path}")
        
        if local_rank == 0:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(self.log_dir)
            self.loss_history = LossHistory(str(self.log_dir), model, input_shape=input_shape)
        else:
            self.writer = None
            self.loss_history = None
            
    def log_epoch(self, epoch, train_metrics, val_metrics, lr):
        """记录一轮数据并立即写入CSV"""
        if self.local_rank != 0:
            return
            
        row = {
            'epoch': epoch,
            'train_loss': f"{train_metrics['loss']:.6f}",
            'train_precision': f"{train_metrics['precision']:.6f}",
            'train_recall': f"{train_metrics['recall']:.6f}",
            'train_f1': f"{train_metrics['f1']:.6f}",
            'train_miou': f"{train_metrics['miou']:.6f}",
            'train_fg_precision': f"{train_metrics.get('fg_precision', 0):.6f}",
            'train_fg_recall': f"{train_metrics.get('fg_recall', 0):.6f}",
            'train_fg_f1': f"{train_metrics.get('fg_f1', 0):.6f}",
            'train_fg_miou': f"{train_metrics.get('fg_miou', 0):.6f}",
            'val_loss': f"{val_metrics['loss']:.6f}",
            'val_precision': f"{val_metrics['precision']:.6f}",
            'val_recall': f"{val_metrics['recall']:.6f}",
            'val_f1': f"{val_metrics['f1']:.6f}",
            'val_miou': f"{val_metrics['miou']:.6f}",
            'val_fg_precision': f"{val_metrics.get('fg_precision', 0):.6f}",
            'val_fg_recall': f"{val_metrics.get('fg_recall', 0):.6f}",
            'val_fg_f1': f"{val_metrics.get('fg_f1', 0):.6f}",
            'val_fg_miou': f"{val_metrics.get('fg_miou', 0):.6f}",
            'inference_time_ms': f"{val_metrics['inference_time_ms']:.4f}",
            'fps': f"{val_metrics['fps']:.2f}",
            'learning_rate': f"{lr:.8f}",
        }
        
        # 立即追加写入CSV
        with open(self.csv_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=self.header)
            writer.writerow(row)
        
        # TensorBoard 记录
        if self.writer is not None:
            self.writer.add_scalar('Train/Loss', train_metrics['loss'], epoch)
            self.writer.add_scalar('Train/Precision', train_metrics['precision'], epoch)
            self.writer.add_scalar('Train/Recall', train_metrics['recall'], epoch)
            self.writer.add_scalar('Train/F1', train_metrics['f1'], epoch)
            self.writer.add_scalar('Train/mIoU', train_metrics['miou'], epoch)
            self.writer.add_scalar('Train/Foreground_Recall', train_metrics.get('fg_recall', 0), epoch)
            self.writer.add_scalar('Train/Foreground_mIoU', train_metrics.get('fg_miou', 0), epoch)
            
            self.writer.add_scalar('Val/Loss', val_metrics['loss'], epoch)
            self.writer.add_scalar('Val/Precision', val_metrics['precision'], epoch)
            self.writer.add_scalar('Val/Recall', val_metrics['recall'], epoch)
            self.writer.add_scalar('Val/F1', val_metrics['f1'], epoch)
            self.writer.add_scalar('Val/mIoU', val_metrics['miou'], epoch)
            self.writer.add_scalar('Val/Foreground_Recall', val_metrics.get('fg_recall', 0), epoch)
            self.writer.add_scalar('Val/Foreground_mIoU', val_metrics.get('fg_miou', 0), epoch)
            self.writer.add_scalar('Val/Inference_Time_ms', val_metrics['inference_time_ms'], epoch)
            self.writer.add_scalar('Val/FPS', val_metrics['fps'], epoch)
            self.writer.add_scalar('Learning_Rate', lr, epoch)
            
    def save_config(self, config_dict):
        """保存训练配置"""
        if self.local_rank == 0:
            config_path = self.log_dir / 'config.json'
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config_dict, f, indent=2, ensure_ascii=False)
                
    def close(self):
        """关闭TensorBoard"""
        if self.writer is not None:
            self.writer.close()
                
    def append_loss(self, epoch, loss, val_loss):
        """兼容原有的 LossHistory 接口"""
        if self.loss_history is not None:
            self.loss_history.append_loss(epoch, loss, val_loss)


def parse_args():
    parser = argparse.ArgumentParser(description='SegFormer 水域分割训练（优化版）')
    
    # 数据路径参数
    parser.add_argument('--images', type=str, default=None,
                        help='训练集图片目录路径')
    parser.add_argument('--masks', type=str, default=None,
                        help='训练集标签目录路径')
    parser.add_argument('--val-images', type=str, default=None,
                        help='验证集图片目录路径（可选，默认从训练集划分）')
    parser.add_argument('--val-masks', type=str, default=None,
                        help='验证集标签目录路径（可选）')
    
    # 兼容旧版的数据集根目录配置
    parser.add_argument('--dataset-root', type=str, default=None,
                        help='数据集根目录（旧版配置方式）')
    parser.add_argument('--image-folder', type=str, default='images',
                        help='图片文件夹名（相对于根目录）')
    parser.add_argument('--mask-folder', type=str, default='masks',
                        help='标签文件夹名（相对于根目录）')
    parser.add_argument('--val-split', type=float, default=0.1,
                        help='验证集比例（当未指定--val-images时使用）')
    
    # 标签格式
    parser.add_argument('--mask-ext', type=str, default='.png',
                        help='标签文件扩展名')
    parser.add_argument('--mask-suffix', type=str, default='',
                        help='标签文件后缀（如 _mask）')
    
    # 训练基本配置
    parser.add_argument('--epochs', type=int, default=50,
                        help='总训练轮数')
    parser.add_argument('--freeze-epochs', type=int, default=5,
                        help='冻结阶段轮数')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='解冻阶段batch size')
    parser.add_argument('--freeze-batch-size', type=int, default=16,
                        help='冻结阶段batch size')
    parser.add_argument('--input-height', type=int, default=512,
                        help='输入图片高度')
    parser.add_argument('--input-width', type=int, default=512,
                        help='输入图片宽度')
    
    # 模型配置
    parser.add_argument('--phi', type=str, default='b0',
                        choices=['b0', 'b1', 'b2', 'b3', 'b4', 'b5'],
                        help='SegFormer 模型大小（b0最小，b5最大）')
    parser.add_argument('--num-classes', type=int, default=2,
                        help='分割类别数')
    parser.add_argument('--pretrained', action='store_true',
                        help='使用预训练权重')
    parser.add_argument('--model-path', type=str, default='',
                        help='加载已有模型权重')
    
    # 优化器配置
    parser.add_argument('--lr', '--learning-rate', dest='lr', type=float, default=1e-4,
                        help='初始学习率（支持--lr或--learning-rate，SegFormer默认1e-4）')
    parser.add_argument('--optimizer', type=str, default='adamw',
                        choices=['sgd', 'adam', 'adamw'],
                        help='优化器类型（SegFormer推荐使用adamw）')
    
    # 损失函数配置（SegFormer 特有）
    parser.add_argument('--dice-loss', action='store_true',
                        help='使用Dice Loss')
    parser.add_argument('--focal-loss', action='store_true',
                        help='使用Focal Loss')
    
    # 新增：输出路径和文件名配置
    parser.add_argument('--model-dir', type=str, default='checkpoints',
                        help='模型保存目录（默认: checkpoints）')
    parser.add_argument('--log-dir', type=str, default='logs',
                        help='训练日志保存目录（默认: logs）')
    parser.add_argument('--model-name', type=str, default=None,
                        help='模型保存文件名（不含扩展名），例如 segformer_b0_exp1')
    parser.add_argument('--log-name', type=str, default=None,
                        help='日志文件名（不含扩展名），例如 segformer_b0_exp1')
    parser.add_argument('--save-interval', type=int, default=0,
                        help='分步保存模型的epoch间隔，0表示不保存中间模型（默认: 0）')
    
    # 其他
    parser.add_argument('--seed', type=int, default=11,
                        help='随机种子')
    parser.add_argument('--no-freeze', action='store_true',
                        help='不进行冻结训练')
    parser.add_argument('--fp16', action='store_true',
                        help='使用混合精度训练')
    parser.add_argument('--workers', type=int, default=4,
                        help='数据加载线程数')
    
    return parser.parse_args()


def save_checkpoint(model, optimizer, epoch, metrics, save_path, is_best=False, is_last=False):
    """保存模型检查点"""
    checkpoint = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    
    if isinstance(metrics, dict):
        checkpoint.update(metrics)
    
    torch.save(checkpoint, save_path)
    
    prefix = ""
    if is_best:
        prefix = "[BEST] "
    elif is_last:
        prefix = "[LAST] "
    
    print(f"{prefix}模型已保存: {save_path}")


if __name__ == "__main__":
    args = parse_args()
    
    Cuda = True
    distributed = False
    sync_bn = False
    fp16 = args.fp16
    
    num_classes = args.num_classes
    phi = args.phi
    pretrained = args.pretrained
    model_path = args.model_path
    input_shape = [args.input_height, args.input_width]
    
    Init_Epoch = 0
    Freeze_Epoch = args.freeze_epochs if not args.no_freeze else 0
    Freeze_batch_size = args.freeze_batch_size
    UnFreeze_Epoch = args.epochs
    Unfreeze_batch_size = args.batch_size
    Freeze_Train = not args.no_freeze
    
    Init_lr = args.lr
    Min_lr = Init_lr * 0.01
    optimizer_type = args.optimizer
    momentum = 0.9
    weight_decay = 1e-2 if optimizer_type == 'adamw' else 1e-4
    lr_decay_type = 'cos'
    
    # 新的保存参数
    model_dir = args.model_dir
    log_dir = args.log_dir
    model_name = args.model_name
    log_name = args.log_name
    save_interval = args.save_interval
    
    # 损失函数配置（SegFormer 特有）
    dice_loss = args.dice_loss
    focal_loss = args.focal_loss
    
    # 关键修复：将 numpy array 转换为 torch tensor
    cls_weights_np = np.ones([num_classes], np.float32)
    cls_weights = torch.from_numpy(cls_weights_np).float()
    
    # 标签格式
    label_ext = args.mask_ext
    label_suffix = args.mask_suffix
    num_workers = args.workers
    
    seed_everything(args.seed)
    
    ngpus_per_node = torch.cuda.device_count()
    if distributed:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        local_rank = 0
        rank = 0

    if local_rank == 0:
        print(f"使用设备: {device}")
        print(f"输入尺寸: {input_shape}")
        print(f"SegFormer 模型: {phi}")
        print(f"学习率: {Init_lr}")
        print(f"优化器: {optimizer_type}")
        print(f"Dice Loss: {dice_loss}, Focal Loss: {focal_loss}")
        
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)

    if pretrained:
        if distributed:
            if local_rank == 0:
                download_weights(phi)  
            dist.barrier()
        else:
            download_weights(phi)

    model = SegFormer(num_classes=num_classes, phi=phi, pretrained=pretrained)
    if not pretrained:
        weights_init(model)
    if model_path != '':
        print(f"加载权重: {model_path}")
        model_dict = model.state_dict()
        pretrained_dict = torch.load(model_path, map_location=device, weights_only=False)
        
        for key in ['epoch', 'optimizer', 'miou', 'metrics']:
            pretrained_dict.pop(key, None)
        
        if 'state_dict' in pretrained_dict:
            pretrained_dict = pretrained_dict['state_dict']
            
        load_key, no_load_key, temp_dict = [], [], {}
        for k, v in pretrained_dict.items():
            if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
                temp_dict[k] = v
                load_key.append(k)
            else:
                no_load_key.append(k)
        model_dict.update(temp_dict)
        model.load_state_dict(model_dict)
        
        if local_rank == 0:
            print(f"成功加载 {len(load_key)} 个参数")
            if len(no_load_key) > 0:
                print(f"未加载 {len(no_load_key)} 个参数")

    if local_rank == 0:
        model_info = get_model_info(model)
        
        info_filename = f'{model_name}_info.txt' if model_name else 'model_info.txt'
        info_path = Path(model_dir) / info_filename
        
        with open(info_path, 'w', encoding='utf-8') as f:
            f.write(f"Model: SegFormer-{phi}\n")
            f.write(f"Total Parameters: {model_info['total_params']:,}\n")
            f.write(f"Trainable Parameters: {model_info['trainable_params']:,}\n")
            f.write(f"Model Size: {model_info['model_size_mb']:.2f} MB\n")
            f.write(f"Input Channels: 3\n")
            f.write(f"Output Channels: {num_classes}\n")
            f.write(f"Input Shape: {input_shape}\n")
            f.write(f"Training Epochs: {UnFreeze_Epoch}\n")
            f.write(f"Batch Size: {Unfreeze_batch_size}\n")
            f.write(f"Learning Rate: {Init_lr}\n")
            f.write(f"Optimizer: {optimizer_type}\n")
            f.write(f"Dice Loss: {dice_loss}\n")
            f.write(f"Focal Loss: {focal_loss}\n")
        print(f"Model info: {model_info['total_params']:,} params, {model_info['model_size_mb']:.2f} MB")

    if local_rank == 0:
        metrics_logger = MetricsLogger(
            log_dir, model, input_shape=input_shape, 
            local_rank=0, log_name=log_name
        )
        
        config = vars(args)
        metrics_logger.save_config(config)
    else:
        metrics_logger = None
        
    if fp16:
        from torch.cuda.amp import GradScaler as GradScaler
        scaler = GradScaler()
    else:
        scaler = None

    model_train = model.train()
    
    if sync_bn and ngpus_per_node > 1 and distributed:
        model_train = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_train)
    elif sync_bn:
        print("Sync_bn is not support in one gpu or not distributed.")

    if Cuda:
        if distributed:
            model_train = model_train.cuda(local_rank)
            model_train = torch.nn.parallel.DistributedDataParallel(model_train, 
                                                                    device_ids=[local_rank], 
                                                                    find_unused_parameters=True)
        else:
            model_train = torch.nn.DataParallel(model)
            cudnn.benchmark = True
            model_train = model_train.cuda()
    
    # 将 cls_weights 移动到正确的设备
    cls_weights = cls_weights.to(device)
    
    # 数据集路径处理
    if args.images is not None and args.masks is not None:
        train_image_dir = args.images
        train_label_dir = args.masks
        
        if args.val_images is not None and args.val_masks is not None:
            val_image_dir = args.val_images
            val_label_dir = args.val_masks
            
            if local_rank == 0:
                print(f"\n{'='*50}")
                print(f"使用独立 Train/Val 路径模式")
                print(f"训练图片: {train_image_dir}")
                print(f"训练标签: {train_label_dir}")
                print(f"验证图片: {val_image_dir}")
                print(f"验证标签: {val_label_dir}")
                print(f"{'='*50}")
            
            train_lines = get_files_from_dir(train_image_dir, train_label_dir, 
                                             label_ext, label_suffix)
            val_lines = get_files_from_dir(val_image_dir, val_label_dir, 
                                           label_ext, label_suffix)
        else:
            if local_rank == 0:
                print(f"\n{'='*50}")
                print(f"从训练集自动划分验证集 (比例: {args.val_split})")
                print(f"图片目录: {train_image_dir}")
                print(f"标签目录: {train_label_dir}")
                print(f"{'='*50}")
            
            train_lines, val_lines = split_dataset(
                train_image_dir,
                val_split=args.val_split,
                seed=args.seed,
                label_dir=train_label_dir,
                label_ext=label_ext,
                label_suffix=label_suffix
            )
            val_image_dir = train_image_dir
            val_label_dir = train_label_dir
    
    elif args.dataset_root is not None:
        dataset_root = args.dataset_root
        image_folder = args.image_folder
        mask_folder = args.mask_folder
        
        train_image_dir = os.path.join(dataset_root, image_folder)
        train_label_dir = os.path.join(dataset_root, mask_folder)
        val_image_dir = train_image_dir
        val_label_dir = train_label_dir
        
        if local_rank == 0:
            print(f"\n{'='*50}")
            print(f"使用数据集根目录模式")
            print(f"根目录: {dataset_root}")
            print(f"自动划分验证集 (比例: {args.val_split})")
            print(f"{'='*50}")
        
        train_lines, val_lines = split_dataset(
            train_image_dir,
            val_split=args.val_split,
            seed=args.seed,
            label_dir=train_label_dir,
            label_ext=label_ext,
            label_suffix=label_suffix
        )
    
    else:
        raise ValueError("必须指定数据路径！请使用以下方式之一：\n"
                        "1. --images 和 --masks（推荐）\n"
                        "2. --dataset-root（旧版方式）")

    num_train = len(train_lines)
    num_val = len(val_lines)

    if local_rank == 0:
        print(f"\n最终数据集:")
        print(f"  训练样本: {num_train}")
        print(f"  验证样本: {num_val}")
        
        print(f"\n保存配置:")
        print(f"  模型目录: {os.path.abspath(model_dir)}")
        print(f"  日志目录: {os.path.abspath(log_dir)}")
        if model_name:
            print(f"  模型前缀: {model_name}")
        if log_name:
            print(f"  日志名称: {log_name}")
        print(f"  分步保存间隔: {save_interval} epochs")
        
        show_config(
            num_classes=num_classes, phi=phi, model_path=model_path, 
            input_shape=input_shape,
            Init_Epoch=Init_Epoch, Freeze_Epoch=Freeze_Epoch, 
            UnFreeze_Epoch=UnFreeze_Epoch, 
            Freeze_batch_size=Freeze_batch_size, 
            Unfreeze_batch_size=Unfreeze_batch_size, 
            Freeze_Train=Freeze_Train,
            Init_lr=Init_lr, Min_lr=Min_lr, 
            optimizer_type=optimizer_type, momentum=momentum, 
            lr_decay_type=lr_decay_type,
            save_period=save_interval, save_dir=model_dir, 
            num_workers=num_workers, num_train=num_train, num_val=num_val
        )
        
        wanted_step = 1.5e4 if optimizer_type in ["adam", "adamw"] else 0.5e4
        total_step = num_train // Unfreeze_batch_size * UnFreeze_Epoch
        if total_step <= wanted_step:
            if num_train // Unfreeze_batch_size == 0:
                raise ValueError('数据集过小，无法进行训练，请扩充数据集。')
            wanted_epoch = wanted_step // (num_train // Unfreeze_batch_size) + 1
            print("\n\033[1;33;44m[Warning] 使用%s优化器时，建议将训练总步长设置到%d以上。\033[0m" % (optimizer_type, wanted_step))
            print("\033[1;33;44m[Warning] 本次运行的总训练数据量为%d，Unfreeze_batch_size为%d，共训练%d个Epoch，计算出总训练步长为%d。\033[0m" % (num_train, Unfreeze_batch_size, UnFreeze_Epoch, total_step))
            print("\033[1;33;44m[Warning] 由于总训练步长为%d，小于建议总步长%d，建议设置总世代为%d。\033[0m" % (total_step, wanted_step, wanted_epoch))
        
    # 冻结训练设置
    if True:
        UnFreeze_flag = False
        if Freeze_Train:
            for param in model.backbone.parameters():
                param.requires_grad = False

        batch_size = Freeze_batch_size if Freeze_Train else Unfreeze_batch_size

        nbs = 16
        lr_limit_max = 1e-4 if optimizer_type in ['adam', 'adamw'] else 5e-2
        lr_limit_min = 3e-5 if optimizer_type in ['adam', 'adamw'] else 5e-4
        Init_lr_fit = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
        Min_lr_fit = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

        optimizer = {
            'adam': optim.Adam(model.parameters(), Init_lr_fit, 
                              betas=(momentum, 0.999), weight_decay=weight_decay),
            'adamw': optim.AdamW(model.parameters(), Init_lr_fit, 
                                betas=(momentum, 0.999), weight_decay=weight_decay),
            'sgd': optim.SGD(model.parameters(), Init_lr_fit, 
                            momentum=momentum, nesterov=True, weight_decay=weight_decay)
        }[optimizer_type]

        lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)
        
        epoch_step = num_train // batch_size
        epoch_step_val = num_val // batch_size
        
        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError("数据集过小，无法继续进行训练，请扩充数据集。")

        # 创建数据集
        train_dataset = SegmentationDataset(
            train_lines, input_shape, num_classes, True, 
            train_image_dir,
            '',
            train_label_dir,
            label_suffix, 
            label_ext, 
            is_2007=False
        )
        val_dataset = SegmentationDataset(
            val_lines, input_shape, num_classes, False, 
            val_image_dir,
            '',
            val_label_dir,
            label_suffix, 
            label_ext, 
            is_2007=False
        )
        
        if distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True,)
            val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False,)
            batch_size = batch_size // ngpus_per_node
            shuffle = False
        else:
            train_sampler = None
            val_sampler = None
            shuffle = True

        gen = DataLoader(train_dataset, shuffle=shuffle, batch_size=batch_size, 
                        num_workers=num_workers, pin_memory=True,
                        drop_last=True, collate_fn=seg_dataset_collate, 
                        sampler=train_sampler, 
                        worker_init_fn=partial(worker_init_fn, rank=rank, seed=args.seed))
        gen_val = DataLoader(val_dataset, shuffle=shuffle, batch_size=batch_size, 
                            num_workers=num_workers, pin_memory=True, 
                            drop_last=True, collate_fn=seg_dataset_collate, 
                            sampler=val_sampler, 
                            worker_init_fn=partial(worker_init_fn, rank=rank, seed=args.seed))

        best_miou = 0.0
        start_epoch = 0
        
        # 恢复训练逻辑
        if model_path and os.path.exists(model_path):
            try:
                checkpoint = torch.load(model_path, map_location=device, weights_only=False)
                if 'epoch' in checkpoint:
                    start_epoch = checkpoint['epoch']
                    print(f"从 epoch {start_epoch} 恢复训练")
            except:
                pass

        for epoch in range(start_epoch, UnFreeze_Epoch):
            
            if epoch >= Freeze_Epoch and not UnFreeze_flag and Freeze_Train:
                batch_size = Unfreeze_batch_size

                nbs = 16
                lr_limit_max = 1e-4 if optimizer_type in ['adam', 'adamw'] else 5e-2
                lr_limit_min = 3e-5 if optimizer_type in ['adam', 'adamw'] else 5e-4
                Init_lr_fit = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
                Min_lr_fit = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)
                
                lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)
                    
                for param in model.backbone.parameters():
                    param.requires_grad = True
                            
                epoch_step = num_train // batch_size
                epoch_step_val = num_val // batch_size

                if epoch_step == 0 or epoch_step_val == 0:
                    raise ValueError("数据集过小，无法继续进行训练，请扩充数据集。")

                if distributed:
                    batch_size = batch_size // ngpus_per_node

                gen = DataLoader(train_dataset, shuffle=shuffle, batch_size=batch_size, 
                                num_workers=num_workers, pin_memory=True,
                                drop_last=True, collate_fn=seg_dataset_collate, 
                                sampler=train_sampler, 
                                worker_init_fn=partial(worker_init_fn, rank=rank, seed=args.seed))
                gen_val = DataLoader(val_dataset, shuffle=shuffle, batch_size=batch_size, 
                                    num_workers=num_workers, pin_memory=True, 
                                    drop_last=True, collate_fn=seg_dataset_collate, 
                                    sampler=val_sampler, 
                                    worker_init_fn=partial(worker_init_fn, rank=rank, seed=args.seed))

                UnFreeze_flag = True

            if distributed:
                train_sampler.set_epoch(epoch)
                
            set_optimizer_lr(optimizer, lr_scheduler_func, epoch)

            if local_rank == 0:
                print(f'\n=== Epoch {epoch+1}/{UnFreeze_Epoch} ===')
            
            # 训练阶段
            model_train.train()
            train_loss = 0
            train_batches = 0
            
            for iteration, batch in enumerate(gen):
                if iteration >= epoch_step:
                    break
                    
                images, targets = batch[0], batch[1]
                with torch.no_grad():
                    images = images.to(device)
                    targets = targets.to(device)
                
                optimizer.zero_grad()
                
                if not fp16:
                    outputs = model_train(images)
                    
                    loss = CE_Loss(outputs, targets, cls_weights, num_classes)
                    if dice_loss:
                        loss += Dice_loss(outputs, targets, cls_weights, num_classes)
                    if focal_loss:
                        loss += Focal_Loss(outputs, targets, cls_weights, num_classes)
                    
                    loss.backward()
                    optimizer.step()
                else:
                    from torch.cuda.amp import autocast
                    with autocast():
                        outputs = model_train(images)
                        
                        loss = CE_Loss(outputs, targets, cls_weights, num_classes)
                        if dice_loss:
                            loss += Dice_loss(outputs, targets, cls_weights, num_classes)
                        if focal_loss:
                            loss += Focal_Loss(outputs, targets, cls_weights, num_classes)
                    
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                
                train_loss += loss.item()
                train_batches += 1
                
                if local_rank == 0:
                    if iteration % 100 == 0:
                        print(f"Train Batch {iteration}/{epoch_step}, Loss: {loss.item():.4f}")
            
            avg_train_loss = train_loss / train_batches if train_batches > 0 else 0
            
            # 验证阶段
            if local_rank == 0:
                print("Evaluating...")
            
            val_metrics = evaluate_metrics(model_train, gen_val, device, num_classes, 
                                          dice_loss=dice_loss, focal_loss=focal_loss, 
                                          cls_weights=cls_weights)
            val_metrics['loss'] = val_metrics.get('loss', 0)
            
            # 计算训练集详细指标
            train_metrics = evaluate_metrics(model_train, gen, device, num_classes,
                                            dice_loss=dice_loss, focal_loss=focal_loss,
                                            cls_weights=cls_weights)
            
            # 记录指标（实时写入CSV）
            if metrics_logger is not None:
                metrics_logger.log_epoch(epoch + 1, train_metrics, val_metrics, 
                                        optimizer.param_groups[0]['lr'])
                metrics_logger.append_loss(epoch + 1, avg_train_loss, val_metrics['loss'])
            
            if local_rank == 0:
                print(f">>> Train Loss: {train_metrics['loss']:.4f}, "
                      f"Train mIoU: {train_metrics['miou']:.4f}, "
                      f"Train FG-Recall: {train_metrics.get('fg_recall', 0):.4f}")
                print(f">>> Val Loss: {val_metrics['loss']:.4f}, "
                      f"Val mIoU: {val_metrics['miou']:.4f}, "
                      f"Val FG-Recall: {val_metrics.get('fg_recall', 0):.4f}, "
                      f"FPS: {val_metrics['fps']:.2f}")
                
                current_miou = val_metrics['miou']
                current_metrics = {
                    'miou': current_miou,
                    'f1': val_metrics['f1'],
                    'precision': val_metrics['precision'],
                    'recall': val_metrics['recall'],
                    'val_loss': val_metrics['loss']
                }
                
                # 保存最佳模型
                if current_miou > best_miou:
                    best_miou = current_miou
                    
                    if model_name:
                        best_path = os.path.join(model_dir, f'{model_name}_best.pth')
                    else:
                        best_path = os.path.join(model_dir, 'checkpoint_best.pth')
                    
                    save_checkpoint(
                        model, optimizer, epoch + 1, 
                        {**current_metrics, 'best_miou': best_miou},
                        best_path, is_best=True
                    )
                
                # 分步保存中间模型
                if save_interval > 0 and (epoch + 1) % save_interval == 0:
                    if model_name:
                        periodic_path = os.path.join(model_dir, f'{model_name}_epoch{epoch+1}.pth')
                    else:
                        periodic_path = os.path.join(model_dir, f'checkpoint_epoch{epoch+1}.pth')
                    
                    save_checkpoint(
                        model, optimizer, epoch + 1,
                        current_metrics,
                        periodic_path
                    )
            
            if distributed:
                dist.barrier()

        # 训练结束，保存最终模型（last）
        if local_rank == 0:
            final_metrics = {
                'miou': val_metrics['miou'],
                'f1': val_metrics['f1'],
                'precision': val_metrics['precision'],
                'recall': val_metrics['recall'],
                'final_epoch': UnFreeze_Epoch
            }
            
            if model_name:
                last_path = os.path.join(model_dir, f'{model_name}_last.pth')
            else:
                last_path = os.path.join(model_dir, 'checkpoint_last.pth')
            
            save_checkpoint(
                model, optimizer, UnFreeze_Epoch,
                final_metrics,
                last_path, is_last=True
            )
            
            print(f"\n{'='*50}")
            print(f"训练完成!")
            print(f"最佳验证 mIoU: {best_miou:.4f}")
            print(f"最终验证 FG-Recall: {val_metrics.get('fg_recall', 0):.4f}")
            if model_name:
                print(f"最佳模型: {model_dir}/{model_name}_best.pth")
                print(f"最终模型: {model_dir}/{model_name}_last.pth")
            else:
                print(f"最佳模型: {model_dir}/checkpoint_best.pth")
                print(f"最终模型: {model_dir}/checkpoint_last.pth")
            print(f"{'='*50}")
            
        if metrics_logger is not None:
            metrics_logger.close()