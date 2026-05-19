import argparse
import os
import time
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from nets.segformer import SegFormer
from utils.dataloader import SegmentationDataset, seg_dataset_collate
from utils.utils import seed_everything


def compute_metrics(pred_mask, true_mask, num_classes):
    """计算分割指标：Precision, Recall, F1, IoU（宏平均）"""
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
            })
    
    if metrics_per_class:
        return {
            'precision': np.mean([m['precision'] for m in metrics_per_class]),
            'recall': np.mean([m['recall'] for m in metrics_per_class]),
            'f1': np.mean([m['f1'] for m in metrics_per_class]),
            'miou': np.mean([m['iou'] for m in metrics_per_class]),
        }
    return {'precision': 0, 'recall': 0, 'f1': 0, 'miou': 0}


@torch.no_grad()
def test_model(model, dataloader, device, num_classes):
    """测试模型，返回各项指标"""
    model.eval()
    
    all_preds = []
    all_targets = []
    total_loss = 0
    num_batches = 0
    inference_times = []
    
    # SegFormer使用CrossEntropyLoss，不需要处理辅助输出
    criterion = nn.CrossEntropyLoss()
    
    for batch in dataloader:
        images, labels = batch[0], batch[1]
        images = images.to(device)
        labels = labels.to(device)
        
        # 测量推理时间
        if device.type == 'cuda':
            torch.cuda.synchronize()
        start = time.time()
        
        outputs = model(images)
        # SegFormer只返回主输出，不需要像PSPNet那样处理辅助输出
            
        if device.type == 'cuda':
            torch.cuda.synchronize()
        inference_times.append(time.time() - start)
        
        # 计算 loss
        loss = criterion(outputs, labels)
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
    metrics['total_images'] = len(all_preds)
    
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


def save_results(metrics, model_info, save_path, args):
    """保存测试结果到 CSV"""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 构建结果字典
    result = {
        'model_path': args.model,
        'test_images': args.images,
        'test_masks': args.masks,
        'phi': args.phi,
        'num_classes': args.num_classes,
        'input_shape': f"{args.input_height}x{args.input_width}",
        'total_params': model_info['total_params'],
        'trainable_params': model_info['trainable_params'],
        'model_size_mb': f"{model_info['model_size_mb']:.2f}",
        'test_loss': f"{metrics['loss']:.6f}",
        'test_precision': f"{metrics['precision']:.6f}",
        'test_recall': f"{metrics['recall']:.6f}",
        'test_f1': f"{metrics['f1']:.6f}",
        'test_miou': f"{metrics['miou']:.6f}",
        'inference_time_ms': f"{metrics['inference_time_ms']:.4f}",
        'fps': f"{metrics['fps']:.2f}",
        'total_images': metrics['total_images'],
    }
    
    # 写入 CSV（追加模式）
    header = list(result.keys())
    file_exists = save_path.exists()
    
    with open(save_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if not file_exists:
            writer.writeheader()
        writer.writerow(result)
    
    print(f"Results saved to {save_path}")


def get_test_files(image_dir, mask_dir, mask_ext='.png', mask_suffix=''):
    """获取测试集文件列表"""
    valid_files = []
    image_exts = ['.jpg', '.jpeg', '.png', '.bmp']
    
    if not os.path.exists(image_dir):
        raise ValueError(f"图片目录不存在: {image_dir}")
    if not os.path.exists(mask_dir):
        raise ValueError(f"标签目录不存在: {mask_dir}")
    
    for fname in sorted(os.listdir(image_dir)):
        is_image = any(fname.lower().endswith(ext) for ext in image_exts)
        if not is_image:
            continue
            
        name_without_ext = os.path.splitext(fname)[0]
        mask_fname = name_without_ext + mask_suffix + mask_ext
        mask_path = os.path.join(mask_dir, mask_fname)
        
        if os.path.exists(mask_path):
            valid_files.append(name_without_ext)
        else:
            print(f"警告: 找不到标签文件 {mask_path}，跳过 {fname}")
    
    if len(valid_files) == 0:
        raise ValueError(f"在 {image_dir} 中没有找到有效的图片-标签对！")
    
    print(f"找到 {len(valid_files)} 个测试样本")
    return valid_files


def main():
    parser = argparse.ArgumentParser(description='Test SegFormer on test set')
    
    parser.add_argument('--model', '-m', type=str, required=True,
                        help='Path to the trained model .pth file')
    parser.add_argument('--images', '-i', type=str, required=True,
                        help='Path to the test images directory')
    parser.add_argument('--masks', type=str, required=True,
                        help='Path to the test masks directory')
    
    # 模型配置
    parser.add_argument('--phi', type=str, default='b0',
                        choices=['b0', 'b1', 'b2', 'b3', 'b4', 'b5'],
                        help='SegFormer model size (b0 to b5)')
    parser.add_argument('--num-classes', type=int, default=2,
                        help='Number of segmentation classes')
    parser.add_argument('--input-height', type=int, default=512,
                        help='Input image height')
    parser.add_argument('--input-width', type=int, default=512,
                        help='Input image width')
    
    # 标签格式
    parser.add_argument('--mask-ext', type=str, default='.png',
                        help='Mask file extension')
    parser.add_argument('--mask-suffix', type=str, default='',
                        help='Mask file suffix')
    
    # 测试配置
    parser.add_argument('--batch-size', type=int, default=4,
                        help='Batch size for testing')
    parser.add_argument('--workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--output', '-o', type=str, default='./logs/test_results.csv',
                        help='Path to save test results CSV')
    parser.add_argument('--seed', type=int, default=11,
                        help='Random seed')
    
    args = parser.parse_args()
    
    # 设置
    seed_everything(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # 加载模型
    print(f'Loading model from {args.model}')
    model = SegFormer(
        num_classes=args.num_classes,
        phi=args.phi,
        pretrained=False
    )
    
    # 加载权重
    checkpoint = torch.load(args.model, map_location=device, weights_only=False)
    
    # 过滤非模型参数
    for key in ['epoch', 'optimizer', 'miou', 'metrics']:
        checkpoint.pop(key, None)
    
    # 处理 state_dict 包装
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        # 直接加载，可能需要过滤不匹配的key
        model_dict = model.state_dict()
        pretrained_dict = checkpoint
        
        # 检查是否有不匹配的key
        load_key, no_load_key, temp_dict = [], [], {}
        for k, v in pretrained_dict.items():
            if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
                temp_dict[k] = v
                load_key.append(k)
            else:
                no_load_key.append(k)
        
        if len(no_load_key) > 0:
            print(f"Warning: {len(no_load_key)} keys not loaded: {str(no_load_key)[:200]}...")
        
        model_dict.update(temp_dict)
        model.load_state_dict(model_dict)
    
    model.to(device)
    
    # 获取模型信息
    model_info = get_model_info(model)
    print(f"Model: SegFormer-{args.phi}")
    print(f"Parameters: {model_info['total_params']:,} (trainable: {model_info['trainable_params']:,})")
    print(f"Model Size: {model_info['model_size_mb']:.2f} MB")
    
    # 准备测试数据
    test_lines = get_test_files(args.images, args.masks, args.mask_ext, args.mask_suffix)
    
    input_shape = [args.input_height, args.input_width]
    
    test_dataset = SegmentationDataset(
        test_lines, input_shape, args.num_classes, False,
        args.images,
        '',
        args.masks,
        args.mask_suffix,
        args.mask_ext,
        is_2007=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=seg_dataset_collate
    )
    
    print(f'Test set size: {len(test_dataset)} images')
    
    # 测试
    print('Starting evaluation...')
    metrics = test_model(model, test_loader, device, args.num_classes)
    
    # 打印结果
    print('=' * 50)
    print('TEST RESULTS')
    print('=' * 50)
    print(f"Loss:           {metrics['loss']:.6f}")
    print(f"Precision:      {metrics['precision']:.6f}")
    print(f"Recall:         {metrics['recall']:.6f}")
    print(f"F1-Score:       {metrics['f1']:.6f}")
    print(f"mIoU:           {metrics['miou']:.6f}")
    print(f"Inference Time: {metrics['inference_time_ms']:.4f} ms")
    print(f"FPS:            {metrics['fps']:.2f}")
    print(f"Total Images:   {metrics['total_images']}")
    print('=' * 50)
    
    # 保存结果
    save_results(metrics, model_info, args.output, args)
    
    # 同时保存详细文本报告
    report_path = Path(args.output).parent / f'test_report_{args.phi}.txt'
    with open(report_path, 'w') as f:
        f.write(f"Model: SegFormer-{args.phi}\n")
        f.write(f"Model Path: {args.model}\n")
        f.write(f"Test Images: {args.images}\n")
        f.write(f"Test Masks: {args.masks}\n\n")
        f.write(f"Total Parameters: {model_info['total_params']:,}\n")
        f.write(f"Trainable Parameters: {model_info['trainable_params']:,}\n")
        f.write(f"Model Size: {model_info['model_size_mb']:.2f} MB\n")
        f.write(f"Input Shape: {args.input_height}x{args.input_width}\n")
        f.write(f"Number of Classes: {args.num_classes}\n\n")
        f.write(f"Test Set Size: {metrics['total_images']} images\n\n")
        f.write(f"Loss:           {metrics['loss']:.6f}\n")
        f.write(f"Precision:      {metrics['precision']:.6f}\n")
        f.write(f"Recall:         {metrics['recall']:.6f}\n")
        f.write(f"F1-Score:       {metrics['f1']:.6f}\n")
        f.write(f"mIoU:           {metrics['miou']:.6f}\n")
        f.write(f"Inference Time: {metrics['inference_time_ms']:.4f} ms\n")
        f.write(f"FPS:            {metrics['fps']:.2f}\n")
    
    print(f"Text report saved to {report_path}")


if __name__ == '__main__':
    main()