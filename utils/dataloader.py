# dataloader.py - 修改后的 SegmentationDataset 类
import os
import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data.dataset import Dataset
from utils.utils import preprocess_input, cvtColor


class SegmentationDataset(Dataset):
    def __init__(self, annotation_lines, input_shape, num_classes, train, 
                 dataset_path, image_folder='JPEGImages', label_folder='SegmentationClass', 
                 label_suffix='', label_ext='.png', is_2007=True):
        """
        修改后的数据集类，支持自定义 label 后缀和扩展名，支持非VOC格式数据集
        """
        super(SegmentationDataset, self).__init__()
        self.annotation_lines   = annotation_lines
        self.length             = len(annotation_lines)
        self.input_shape        = input_shape
        self.num_classes        = num_classes
        self.train              = train
        self.dataset_path       = dataset_path
        self.image_folder       = image_folder
        self.label_folder       = label_folder
        self.label_suffix       = label_suffix
        self.label_ext          = label_ext
        
        # 构建完整路径（支持绝对路径）
        if os.path.isabs(image_folder):
            self.images_path = image_folder
        elif is_2007:
            self.images_path = os.path.join(dataset_path, "VOC2007", image_folder)
        else:
            self.images_path = os.path.join(dataset_path, image_folder)
            
        if os.path.isabs(label_folder):
            self.labels_path = label_folder
        elif is_2007:
            self.labels_path = os.path.join(dataset_path, "VOC2007", label_folder)
        else:
            self.labels_path = os.path.join(dataset_path, label_folder)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        name = self.annotation_lines[index].strip()
        
        #-------------------------------#
        #   构建图片路径（尝试多种扩展名）
        #-------------------------------#
        image_path = None
        for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
            tmp_path = os.path.join(self.images_path, name + ext)
            if os.path.exists(tmp_path):
                image_path = tmp_path
                break
        
        if image_path is None:
            raise FileNotFoundError(f"找不到图片文件: {name}.* (在 {self.images_path} 中)")
        
        #-------------------------------#
        #   构建标签路径：name + suffix + label_ext
        #-------------------------------#
        label_name = name + self.label_suffix + self.label_ext
        label_path = os.path.join(self.labels_path, label_name)
        
        if not os.path.exists(label_path):
            raise FileNotFoundError(f"找不到标签文件: {label_name} (在 {self.labels_path} 中)")

        #-------------------------------#
        #   读取图片和标签
        #-------------------------------#
        jpg = Image.open(image_path)
        
        # 读取标签（支持 gif）
        png = Image.open(label_path)
        
        # gif 格式需要特殊处理（取第一帧，转为灰度）
        if label_path.lower().endswith('.gif'):
            png.seek(0)  # 取第一帧
            png = png.convert('L')  # 转为灰度图

        jpg = cvtColor(jpg)
        # 确保 label 是单通道灰度图
        if png.mode != 'L':
            png = png.convert('L')
        
        #-------------------------------#
        #   数据增强
        #-------------------------------#
        jpg, png = self.get_random_data(jpg, png, self.input_shape, random=self.train)

        #-------------------------------#
        #   转换为 Tensor（参考PSPNet的修复）
        #-------------------------------#
        # 处理图片
        jpg_np = preprocess_input(np.array(jpg, np.float64))
        jpg_tensor = torch.from_numpy(np.transpose(jpg_np, [2, 0, 1])).float()

        # 处理标签
        png_np = np.array(png)
        # 确保标签值在合法范围内 [0, num_classes]
        png_np[png_np >= self.num_classes] = self.num_classes
        png_tensor = torch.from_numpy(png_np).long()

        # 生成 one-hot 编码的 seg_labels
        seg_labels_np = np.eye(self.num_classes + 1)[png_np.reshape([-1])]
        seg_labels_np = seg_labels_np.reshape((int(self.input_shape[0]), int(self.input_shape[1]), self.num_classes + 1))
        seg_labels_tensor = torch.from_numpy(seg_labels_np).float()

        return jpg_tensor, png_tensor, seg_labels_tensor

    def rand(self, a=0, b=1):
        return np.random.rand() * (b - a) + a

    def get_random_data(self, image, label, input_shape, jitter=.3, hue=.1, sat=0.7, val=0.3, random=True):
        image = cvtColor(image)
        label = Image.fromarray(np.array(label))
        
        iw, ih = image.size
        h, w = input_shape

        if not random:
            # 非训练模式：使用letterbox方式resize
            scale = min(w/iw, h/ih)
            nw = int(iw*scale)
            nh = int(ih*scale)

            image = image.resize((nw, nh), Image.BICUBIC)
            new_image = Image.new('RGB', [w, h], (128, 128, 128))
            new_image.paste(image, ((w-nw)//2, (h-nh)//2))

            label = label.resize((nw, nh), Image.NEAREST)
            new_label = Image.new('L', [w, h], (0))
            new_label.paste(label, ((w-nw)//2, (h-nh)//2))
            return new_image, new_label

        #------------------------------------------#
        #   训练模式：对图像进行缩放并且进行长和宽的扭曲
        #------------------------------------------#
        new_ar = iw/ih * self.rand(1-jitter, 1+jitter) / self.rand(1-jitter, 1+jitter)
        scale = self.rand(0.5, 2)
        if new_ar < 1:
            nh = int(scale*h)
            nw = int(nh*new_ar)
        else:
            nw = int(scale*w)
            nh = int(nw/new_ar)
        image = image.resize((nw, nh), Image.BICUBIC)
        label = label.resize((nw, nh), Image.NEAREST)
        
        #------------------------------------------#
        #   翻转图像
        #------------------------------------------#
        flip = self.rand() < .5
        if flip: 
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            label = label.transpose(Image.FLIP_LEFT_RIGHT)
        
        #------------------------------------------#
        #   将图像多余的部分加上灰条
        #------------------------------------------#
        dx = int(self.rand(0, w-nw))
        dy = int(self.rand(0, h-nh))
        new_image = Image.new('RGB', (w, h), (128, 128, 128))
        new_label = Image.new('L', (w, h), (0))
        new_image.paste(image, (dx, dy))
        new_label.paste(label, (dx, dy))
        image = new_image
        label = new_label

        image_data = np.array(image, np.uint8)
        
        #------------------------------------------#
        #   高斯模糊
        #------------------------------------------#
        blur = self.rand() < 0.25
        if blur: 
            image_data = cv2.GaussianBlur(image_data, (5, 5), 0)

        #------------------------------------------#
        #   旋转
        #------------------------------------------#
        rotate = self.rand() < 0.25
        if rotate: 
            center = (w // 2, h // 2)
            rotation = np.random.randint(-10, 11)
            M = cv2.getRotationMatrix2D(center, -rotation, scale=1)
            image_data = cv2.warpAffine(image_data, M, (w, h), flags=cv2.INTER_CUBIC, borderValue=(128, 128, 128))
            label = cv2.warpAffine(np.array(label, np.uint8), M, (w, h), flags=cv2.INTER_NEAREST, borderValue=(0))

        #---------------------------------#
        #   对图像进行色域变换
        #---------------------------------#
        r = np.random.uniform(-1, 1, 3) * [hue, sat, val] + 1
        
        hue, sat, val = cv2.split(cv2.cvtColor(image_data, cv2.COLOR_RGB2HSV))
        dtype = image_data.dtype
        
        x = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        image_data = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        image_data = cv2.cvtColor(image_data, cv2.COLOR_HSV2RGB)
        
        return image_data, label


def seg_dataset_collate(batch):
    """
    Collate 函数直接堆叠为 Tensor
    """
    images = []
    pngs = []
    seg_labels = []
    
    for img, png, labels in batch:
        images.append(img)
        pngs.append(png)
        seg_labels.append(labels)
    
    # 使用 torch.stack 直接堆叠
    images = torch.stack(images, 0)
    pngs = torch.stack(pngs, 0)
    seg_labels = torch.stack(seg_labels, 0)
    
    return images, pngs, seg_labels