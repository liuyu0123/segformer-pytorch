# predict_water_segformer_pro.py
# SegFormer 版本：与 predict_water_best_pro.py 保持相同的命令行接口和评估功能
import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import cv2
from PIL import Image
import csv
from glob import glob
import copy

# SegFormer 模型导入
from nets.segformer import SegFormer
from utils.utils import cvtColor, preprocess_input, resize_image

# ==================== 工具函数（与 PSPNet 版本保持一致） ====================
def get_image_paths(path):
    if os.path.isfile(path):
        return [path]
    elif os.path.isdir(path):
        paths = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff']:
            paths.extend(glob(os.path.join(path, ext)))
            paths.extend(glob(os.path.join(path, ext.upper())))
        return sorted(list(set(paths)))
    else:
        raise ValueError(f"输入路径无效: {path}")

def find_ground_truth(img_name, gt_dir):
    base_name = os.path.splitext(img_name)[0]
    exts = ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '']
    for ext in exts:
        gt_path = os.path.join(gt_dir, base_name + ext)
        if os.path.exists(gt_path):
            return gt_path
    return None

def compute_metrics(pred_mask, gt_mask):
    TP = np.sum((pred_mask == 1) & (gt_mask == 1))
    FP = np.sum((pred_mask == 1) & (gt_mask == 0))
    FN = np.sum((pred_mask == 0) & (gt_mask == 1))
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    miou = TP / (TP + FP + FN) if (TP + FP + FN) > 0 else 0.0
    return {
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'miou': float(miou)
    }

def print_metrics_table(metrics_list):
    print("\n" + "="*80)
    print("SegFormer 分割性能评估结果")
    print("-"*80)
    print(f"{'文件名':<30} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'mIoU':<10}")
    print("-"*80)
    for m in metrics_list:
        print(f"{m['image']:<30} {m['precision']:<10.4f} {m['recall']:<10.4f} "
              f"{m['f1']:<10.4f} {m['miou']:<10.4f}")
    avg_p = np.mean([m['precision'] for m in metrics_list])
    avg_r = np.mean([m['recall'] for m in metrics_list])
    avg_f1 = np.mean([m['f1'] for m in metrics_list])
    avg_iou = np.mean([m['miou'] for m in metrics_list])
    print("-"*80)
    print(f"{'[整体平均]':<30} {avg_p:<10.4f} {avg_r:<10.4f} {avg_f1:<10.4f} {avg_iou:<10.4f}")
    print("="*80)

def save_csv(metrics_list, save_path):
    if not metrics_list:
        return
    avg_metrics = {
        'image': 'AVERAGE',
        'precision': np.mean([m['precision'] for m in metrics_list]),
        'recall': np.mean([m['recall'] for m in metrics_list]),
        'f1': np.mean([m['f1'] for m in metrics_list]),
        'miou': np.mean([m['miou'] for m in metrics_list])
    }
    with open(save_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['image', 'precision', 'recall', 'f1', 'miou'])
        writer.writeheader()
        writer.writerows(metrics_list + [avg_metrics])
    print(f"\n✓ 指标已保存至: {save_path}")

def save_overlay_result(pil_img, pred_mask, save_path, alpha=0.4):
    img_array = np.array(pil_img).astype(np.float32)
    red_overlay = np.zeros_like(img_array)
    red_overlay[pred_mask == 1] = [255, 0, 0]
    mask_3ch = np.stack([pred_mask] * 3, axis=-1)
    bg_weight = 1 - (mask_3ch * alpha)
    red_weight = mask_3ch * alpha
    result = img_array * bg_weight + red_overlay * red_weight
    result = np.clip(result, 0, 255).astype(np.uint8)
    Image.fromarray(result).save(save_path)
    print(f"  ✓ 已保存: {os.path.basename(save_path)}")

# ==================== 主函数 ====================
def main():
    parser = argparse.ArgumentParser(description='SegFormer 水体分割推理工具（自适应修复版）')
    parser.add_argument('--input', '-i', type=str, required=True, help='输入图片路径或文件夹')
    parser.add_argument('--weights', '-w', type=str, required=True, help='模型权重 .pth')
    parser.add_argument('--output', '-o', type=str, default=None, help='输出文件夹')
    parser.add_argument('--ground_truth', '-g', type=str, default=None, help='真值标签文件夹')
    parser.add_argument('--alpha', '-a', type=float, default=0.4, help='红色蒙版透明度')
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--phi', type=str, default='b0', choices=['b0', 'b1', 'b2', 'b3', 'b4', 'b5'], 
                        help='SegFormer 模型大小 (b0最小，b5最大)')
    parser.add_argument('--max_side', type=int, default=1024, help='自适应输入长边最大像素（默认1024，建议512/1024）')
    parser.add_argument('--no_cuda', action='store_true')
    args = parser.parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    # 获取图片
    try:
        input_paths = get_image_paths(args.input)
        print(f"发现 {len(input_paths)} 张待处理图片")
        print(f"模型类型: SegFormer-{args.phi.upper()} | 自适应最大边长: {args.max_side} | 透明度: {args.alpha}\n")
    except ValueError as e:
        print(f"错误: {e}")
        return

    if args.output:
        os.makedirs(args.output, exist_ok=True)
        print(f"输出目录: {args.output}\n")

    # ==================== 模型加载 ====================
    print("加载 SegFormer 模型...")
    net = SegFormer(
        num_classes=args.num_classes,
        phi=args.phi,
        pretrained=False
    )

    # 关键修复：兼容训练脚本保存的字典格式
    checkpoint = torch.load(args.weights, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        net.load_state_dict(checkpoint['state_dict'], strict=True)
        epoch_info = checkpoint.get('epoch', 'unknown')
        miou_info = checkpoint.get('best_miou', checkpoint.get('miou', 'N/A'))
        print(f"  [检测到训练检查点] Epoch: {epoch_info} | mIoU: {miou_info}")
    else:
        # 兼容纯 state_dict 格式
        net.load_state_dict(checkpoint, strict=True)
        
    net = net.eval()
    if use_cuda:
        net = torch.nn.DataParallel(net).cuda()
    print(f"✓ 权重加载成功: {args.weights}")
    print(f"  设备: {device} | 模型: SegFormer-{args.phi.upper()}\n")

    metrics_list = []

    for img_path in input_paths:
        img_name = os.path.basename(img_path)
        try:
            # 1. 读取图片
            image = Image.open(img_path)
            image = cvtColor(image)
            old_img = copy.deepcopy(image)
            orininal_h = np.array(image).shape[0]
            orininal_w = np.array(image).shape[1]

            # 2. 自适应计算输入尺寸 (保持宽高比，长边不超过 max_side，且必须是32的倍数)
            scale = min(args.max_side / max(orininal_w, orininal_h), 1.0)
            nw = int(orininal_w * scale)
            nh = int(orininal_h * scale)
            # 向下对齐到32的倍数（SegFormer 通常需要 32 的倍数）
            nw = max(32, (nw // 32) * 32)
            nh = max(32, (nh // 32) * 32)

            # 3. 预处理（添加灰条实现不失真 resize）
            image_data, nw_real, nh_real = resize_image(image, (nw, nh))
            image_data = np.expand_dims(
                np.transpose(preprocess_input(np.array(image_data, np.float32)), (2, 0, 1)), 
                0
            )

            # 4. 推理
            with torch.no_grad():
                images = torch.from_numpy(image_data).to(device)
                pr = net(images)[0] 
                
                pr = F.softmax(pr.permute(1, 2, 0), dim=-1).cpu().numpy()
                
                # 裁切灰条（与原始逻辑保持一致）
                pr = pr[int((nh - nh_real) // 2) : int((nh - nh_real) // 2 + nh_real), 
                       int((nw - nw_real) // 2) : int((nw - nw_real) // 2 + nw_real)]
                
                # Resize 回原图尺寸
                pr = cv2.resize(pr, (orininal_w, orininal_h), interpolation=cv2.INTER_LINEAR)
                pr = pr.argmax(axis=-1)
                
        except Exception as e:
            print(f"跳过 {img_name}: {e}")
            continue

        # 5. 保存叠加图（红色蒙版）
        if args.output:
            save_path = os.path.join(args.output, img_name)
            save_overlay_result(old_img, pr, save_path, args.alpha)

        # 6. 真值评测（与 PSPNet 版本完全一致）
        if args.ground_truth:
            gt_path = find_ground_truth(img_name, args.ground_truth)
            if gt_path:
                try:
                    gt_img = Image.open(gt_path).convert('L')
                    gt_np = np.array(gt_img)
                    unique_vals = np.unique(gt_np)
                    
                    # 自动检测真值格式并转换
                    if len(unique_vals) <= 2 and unique_vals.max() <= 1:
                        gt_mask = gt_np.astype(np.uint8)
                    elif np.mean(gt_np > 127) > 0.5:
                        gt_mask = (gt_np < 127).astype(np.uint8)
                    else:
                        gt_mask = (gt_np > 127).astype(np.uint8)

                    if gt_mask.shape != pr.shape:
                        gt_mask = cv2.resize(gt_mask, (pr.shape[1], pr.shape[0]), interpolation=cv2.INTER_NEAREST)
                    
                    metrics = compute_metrics(pr, gt_mask)
                    metrics['image'] = img_name
                    metrics_list.append(metrics)
                    print(f"  处理: {img_name} | Precision: {metrics['precision']:.4f} | mIoU: {metrics['miou']:.4f}")
                except Exception as e:
                    print(f"  警告: 真值处理失败 {img_name}: {e}")
            else:
                print(f"  警告: 未找到 {img_name} 的真值文件")

    # 输出结果汇总
    if args.ground_truth and metrics_list:
        print_metrics_table(metrics_list)
        if args.output:
            save_csv(metrics_list, os.path.join(args.output, 'metrics.csv'))
    elif args.ground_truth and not metrics_list:
        print("\n[警告] 未找到任何匹配的真值文件，无法计算指标。")

    print(f"\n✓ 处理完成！共处理 {len(input_paths)} 张图片")

if __name__ == "__main__":
    main()