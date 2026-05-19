import datetime
import os
import random
from functools import partial

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim as optim
from torch.utils.data import DataLoader

from nets.segformer import SegFormer
from nets.segformer_training import (get_lr_scheduler, set_optimizer_lr,
                                     weights_init)
from utils.callbacks import EvalCallback, LossHistory
from utils.dataloader import SegmentationDataset, seg_dataset_collate
from utils.utils import (download_weights, seed_everything, show_config,
                         worker_init_fn)
from utils.utils_fit import fit_one_epoch


def split_dataset(image_dir, val_split=0.1, seed=42, label_dir=None, 
                  image_exts=['.jpg', '.jpeg', '.png', '.bmp'], 
                  label_ext='.png', label_suffix=''): 
    """
    自动分割数据集，支持检查标签文件是否存在
    """
    random.seed(seed)
    
    if label_dir is None:
        label_dir = image_dir
    
    # 获取所有有效图片（同时存在对应标签的图片）
    valid_files = []
    for fname in os.listdir(image_dir):
        # 检查是否是支持的图片格式
        is_image = any(fname.lower().endswith(ext) for ext in image_exts)
        if not is_image:
            continue
            
        # 获取文件名（不含扩展名）
        name_without_ext = os.path.splitext(fname)[0]
        
        # 检查对应的标签文件是否存在
        label_fname = name_without_ext + label_suffix + label_ext
        label_path = os.path.join(label_dir, label_fname)
        
        if os.path.exists(label_path):
            valid_files.append(name_without_ext)
        else:
            print(f"警告: 找不到标签文件 {label_path}，跳过 {fname}")
    
    if len(valid_files) == 0:
        raise ValueError(f"在 {image_dir} 中没有找到有效的图片-标签对！"
                        f"\n请检查: \n  图片目录: {image_dir}\n  标签目录: {label_dir}"
                        f"\n  标签格式: {label_ext}")
    
    valid_files.sort()
    
    # 随机打乱
    random.shuffle(valid_files)
    
    # 分割
    val_num = int(len(valid_files) * val_split)
    val_lines = valid_files[:val_num]
    train_lines = valid_files[val_num:]
    
    print(f"\n{'='*50}")
    print(f"数据集分割完成:")
    print(f"  总样本数: {len(valid_files)}")
    print(f"  训练集: {len(train_lines)} ({len(train_lines)/len(valid_files)*100:.1f}%)")
    print(f"  验证集: {len(val_lines)} ({len(val_lines)/len(valid_files)*100:.1f}%)")
    print(f"{'='*50}\n")
    
    return train_lines, val_lines


if __name__ == "__main__":
    #---------------------------------#
    #   Cuda    是否使用Cuda
    #---------------------------------#
    Cuda            = True
    #----------------------------------------------#
    #   Seed    用于固定随机种子
    #----------------------------------------------#
    seed            = 11
    #---------------------------------------------------------------------#
    #   distributed     用于指定是否使用单机多卡分布式运行
    #---------------------------------------------------------------------#
    distributed     = False
    #---------------------------------------------------------------------#
    #   sync_bn     是否使用sync_bn，DDP模式多卡可用
    #---------------------------------------------------------------------#
    sync_bn         = False
    #---------------------------------------------------------------------#
    #   fp16        是否使用混合精度训练
    #---------------------------------------------------------------------#
    fp16            = False
    #-----------------------------------------------------#
    #   num_classes     类别数（水域分割：背景+水域=2类）
    #-----------------------------------------------------#
    num_classes     = 2  # <-- 修改：水域分割只有2类
    #-------------------------------------------------------------------#
    #   所使用的的主干网络：b0、b1、b2、b3、b4、b5
    #-------------------------------------------------------------------#
    phi             = "b0"
    #---------------------------------------------------------------------#
    #   pretrained      是否使用主干网络的预训练权重
    #---------------------------------------------------------------------#
    pretrained      = False
    #---------------------------------------------------------------------#
    #   model_path      权值文件路径（空字符串表示从头训练）
    #---------------------------------------------------------------------#
    model_path      = ""  # <-- 修改：水域分割不使用VOC预训练权重
    #------------------------------#
    #   输入图片的大小
    #------------------------------#
    input_shape     = [512, 512]  # <-- 可根据需要调整，如 [320, 640]
    
    #------------------------------------------------------------------#
    #   冻结阶段训练参数
    #------------------------------------------------------------------#
    Init_Epoch          = 0
    Freeze_Epoch        = 50
    Freeze_batch_size   = 16
    #------------------------------------------------------------------#
    #   解冻阶段训练参数
    #------------------------------------------------------------------#
    UnFreeze_Epoch      = 100
    Unfreeze_batch_size = 8
    #------------------------------------------------------------------#
    #   Freeze_Train    是否进行冻结训练
    #------------------------------------------------------------------#
    Freeze_Train        = True

    #------------------------------------------------------------------#
    #   其它训练参数：学习率、优化器、学习率下降有关
    #------------------------------------------------------------------#
    Init_lr             = 1e-4
    Min_lr              = Init_lr * 0.01
    optimizer_type      = "adamw"
    momentum            = 0.9
    weight_decay        = 1e-2
    lr_decay_type       = 'cos'
    save_period         = 5
    save_dir            = 'logs'
    eval_flag           = True
    eval_period         = 5

    #==================================================================#
    #   数据集配置（根据 USVInlandDataset 修改）
    #==================================================================#
    #   数据集根目录
    VOCdevkit_path  = r'D:\Files\Data\USVInlandDataset\Water Segmentation\training\training'
    
    #   图片文件夹路径（相对根目录）
    image_folder    = r'640_320_undistorted'
    
    #   标签文件夹路径
    label_folder    = r'640_320_undistorted_pspnet'
    
    #   标签文件后缀（空字符串表示同名：001.jpg -> 001.png）
    label_suffix    = ''
    
    #   标签文件扩展名
    label_ext       = '.png'
    
    #   验证集比例（10%作为验证集）
    val_split       = 0.1
    
    #   启用自动分割（不需要手动准备txt文件）
    auto_split      = True

    #------------------------------------------------------------------#
    #   损失函数设置
    #------------------------------------------------------------------#
    dice_loss       = False
    focal_loss      = False
    cls_weights     = np.ones([num_classes], np.float32)
    num_workers     = 4

    seed_everything(seed)
    
    #------------------------------------------------------#
    #   设置用到的显卡
    #------------------------------------------------------#
    ngpus_per_node  = torch.cuda.device_count()
    if distributed:
        dist.init_process_group(backend="nccl")
        local_rank  = int(os.environ["LOCAL_RANK"])
        rank        = int(os.environ["RANK"])
        device      = torch.device("cuda", local_rank)
        if local_rank == 0:
            print(f"[{os.getpid()}] (rank = {rank}, local_rank = {local_rank}) training...")
            print("Gpu Device Count : ", ngpus_per_node)
    else:
        device          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        local_rank      = 0
        rank            = 0

    #----------------------------------------------------#
    #   下载预训练权重
    #----------------------------------------------------#
    if pretrained:
        if distributed:
            if local_rank == 0:
                download_weights(phi)  
            dist.barrier()
        else:
            download_weights(phi)

    model   = SegFormer(num_classes=num_classes, phi=phi, pretrained=pretrained)
    if not pretrained:
        weights_init(model)
    if model_path != '':
        #------------------------------------------------------#
        #   权值文件请看README，百度网盘下载
        #------------------------------------------------------#
        if local_rank == 0:
            print('Load weights {}.'.format(model_path))
        
        #------------------------------------------------------#
        #   根据预训练权重的Key和模型的Key进行加载
        #------------------------------------------------------#
        model_dict      = model.state_dict()
        pretrained_dict = torch.load(model_path, map_location = device)
        load_key, no_load_key, temp_dict = [], [], {}
        for k, v in pretrained_dict.items():
            if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
                temp_dict[k] = v
                load_key.append(k)
            else:
                no_load_key.append(k)
        model_dict.update(temp_dict)
        model.load_state_dict(model_dict)
        #------------------------------------------------------#
        #   显示没有匹配上的Key
        #------------------------------------------------------#
        if local_rank == 0:
            print("\nSuccessful Load Key:", str(load_key)[:500], "……\nSuccessful Load Key Num:", len(load_key))
            print("\nFail To Load Key:", str(no_load_key)[:500], "……\nFail To Load Key num:", len(no_load_key))
            print("\n\033[1;33;44m温馨提示，head部分没有载入是正常现象，Backbone部分没有载入是错误的。\033[0m")

    #----------------------#
    #   记录Loss
    #----------------------#
    if local_rank == 0:
        time_str        = datetime.datetime.strftime(datetime.datetime.now(),'%Y_%m_%d_%H_%M_%S')
        log_dir         = os.path.join(save_dir, "loss_" + str(time_str))
        loss_history    = LossHistory(log_dir, model, input_shape=input_shape)
    else:
        loss_history    = None
        
    if fp16:
        from torch.cuda.amp import GradScaler as GradScaler
        scaler = GradScaler()
    else:
        scaler = None

    model_train     = model.train()
    
    if sync_bn and ngpus_per_node > 1 and distributed:
        model_train = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_train)
    elif sync_bn:
        print("Sync_bn is not support in one gpu or not distributed.")

    if Cuda:
        if distributed:
            model_train = model_train.cuda(local_rank)
            model_train = torch.nn.parallel.DistributedDataParallel(model_train, device_ids=[local_rank], find_unused_parameters=True)
        else:
            model_train = torch.nn.DataParallel(model)
            cudnn.benchmark = True
            model_train = model_train.cuda()
    
    #---------------------------#
    #   读取数据集（修改部分）
    #---------------------------#
    # 构建完整路径（支持绝对路径）
    image_dir = os.path.join(VOCdevkit_path, image_folder) if not os.path.isabs(image_folder) else image_folder
    label_dir = os.path.join(VOCdevkit_path, label_folder) if not os.path.isabs(label_folder) else label_folder
    
    if auto_split:
        if local_rank == 0:
            print(f"\n{'='*50}")
            print(f"使用自动分割模式")
            print(f"图片目录: {image_dir}")
            print(f"标签目录: {label_dir}")
            print(f"标签格式: [图片名]{label_suffix}{label_ext}")
            print(f"验证集比例: {val_split}")
            print(f"{'='*50}")
        
        train_lines, val_lines = split_dataset(
            image_dir, 
            val_split=val_split, 
            seed=seed,
            label_dir=label_dir,
            label_ext=label_ext,
            label_suffix=label_suffix
        )
    else:
        # 使用原有的txt文件方式
        train_txt_path  = os.path.join(VOCdevkit_path, "VOC2007/ImageSets/Segmentation/train.txt")
        val_txt_path    = os.path.join(VOCdevkit_path, "VOC2007/ImageSets/Segmentation/val.txt")
        with open(train_txt_path, "r") as f:
            train_lines = [line.strip() for line in f.readlines()]
        with open(val_txt_path, "r") as f:
            val_lines = [line.strip() for line in f.readlines()]

    num_train   = len(train_lines)
    num_val     = len(val_lines)

    if local_rank == 0:
        show_config(
            num_classes = num_classes, phi = phi, model_path = model_path, input_shape = input_shape, \
            Init_Epoch = Init_Epoch, Freeze_Epoch = Freeze_Epoch, UnFreeze_Epoch = UnFreeze_Epoch, Freeze_batch_size = Freeze_batch_size, Unfreeze_batch_size = Unfreeze_batch_size, Freeze_Train = Freeze_Train, \
            Init_lr = Init_lr, Min_lr = Min_lr, optimizer_type = optimizer_type, momentum = momentum, lr_decay_type = lr_decay_type, \
            save_period = save_period, save_dir = save_dir, num_workers = num_workers, num_train = num_train, num_val = num_val
        )
        
        wanted_step = 1.5e4 if optimizer_type == "adamw" else 0.5e4
        total_step  = num_train // Unfreeze_batch_size * UnFreeze_Epoch
        if total_step <= wanted_step:
            if num_train // Unfreeze_batch_size == 0:
                raise ValueError('数据集过小，无法进行训练，请扩充数据集。')
            wanted_epoch = wanted_step // (num_train // Unfreeze_batch_size) + 1
            print("\n\033[1;33;44m[Warning] 使用%s优化器时，建议将训练总步长设置到%d以上。\033[0m"%(optimizer_type, wanted_step))
            print("\033[1;33;44m[Warning] 本次运行的总训练数据量为%d，Unfreeze_batch_size为%d，共训练%d个Epoch，计算出总训练步长为%d。\033[0m"%(num_train, Unfreeze_batch_size, UnFreeze_Epoch, total_step))
            print("\033[1;33;44m[Warning] 由于总训练步长为%d，小于建议总步长%d，建议设置总世代为%d。\033[0m"%(total_step, wanted_step, wanted_epoch))

    #------------------------------------------------------#
    #   主干特征提取网络特征通用，冻结训练可以加快训练速度
    #------------------------------------------------------#
    if True:
        UnFreeze_flag = False
        
        if Freeze_Train:
            for param in model.backbone.parameters():
                param.requires_grad = False

        batch_size = Freeze_batch_size if Freeze_Train else Unfreeze_batch_size

        nbs             = 16
        lr_limit_max    = 1e-4 if optimizer_type in ['adam', 'adamw'] else 5e-2
        lr_limit_min    = 3e-5 if optimizer_type in ['adam', 'adamw'] else 5e-4
        Init_lr_fit     = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
        Min_lr_fit      = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

        optimizer = {
            'adam'  : optim.Adam(model.parameters(), Init_lr_fit, betas = (momentum, 0.999), weight_decay = weight_decay),
            'adamw' : optim.AdamW(model.parameters(), Init_lr_fit, betas = (momentum, 0.999), weight_decay = weight_decay),
            'sgd'   : optim.SGD(model.parameters(), Init_lr_fit, momentum = momentum, nesterov=True, weight_decay = weight_decay)
        }[optimizer_type]

        lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)
        
        epoch_step      = num_train // batch_size
        epoch_step_val  = num_val // batch_size
        
        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError("数据集过小，无法继续进行训练，请扩充数据集。")

        # 修改后的数据集类 - 需要自定义支持非VOC格式的Dataset
        # 注意：这里假设您会修改 SegmentationDataset 或创建新的 Dataset 类
        # 如果 SegmentationDataset 不支持自定义路径，需要修改 utils/dataloader.py
        train_dataset   = SegmentationDataset(
            train_lines, input_shape, num_classes, True, 
            VOCdevkit_path, image_folder, label_folder, 
            label_suffix, label_ext, is_2007=False
        )
        val_dataset     = SegmentationDataset(
            val_lines, input_shape, num_classes, False, 
            VOCdevkit_path, image_folder, label_folder, 
            label_suffix, label_ext, is_2007=False
        )
        
        if distributed:
            train_sampler   = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True,)
            val_sampler     = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False,)
            batch_size      = batch_size // ngpus_per_node
            shuffle         = False
        else:
            train_sampler   = None
            val_sampler     = None
            shuffle         = True

        gen             = DataLoader(train_dataset, shuffle = shuffle, batch_size = batch_size, num_workers = num_workers, pin_memory=True,
                                    drop_last = True, collate_fn = seg_dataset_collate, sampler=train_sampler, 
                                    worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))
        gen_val         = DataLoader(val_dataset  , shuffle = shuffle, batch_size = batch_size, num_workers = num_workers, pin_memory=True, 
                                    drop_last = True, collate_fn = seg_dataset_collate, sampler=val_sampler, 
                                    worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))

        if local_rank == 0:
            eval_callback   = EvalCallback(
                model, input_shape, num_classes, val_lines, VOCdevkit_path, log_dir, Cuda, \
                eval_flag=eval_flag, period=eval_period,
                image_folder=image_folder, label_folder=label_folder, 
                label_suffix=label_suffix, label_ext=label_ext, is_2007=False
            )
        else:
            eval_callback   = None
        
        #---------------------------------------#
        #   开始模型训练
        #---------------------------------------#
        for epoch in range(Init_Epoch, UnFreeze_Epoch):
            
            if epoch >= Freeze_Epoch and not UnFreeze_flag and Freeze_Train:
                batch_size = Unfreeze_batch_size

                nbs             = 16
                lr_limit_max    = 1e-4 if optimizer_type in ['adam', 'adamw'] else 5e-2
                lr_limit_min    = 3e-5 if optimizer_type in ['adam', 'adamw'] else 5e-4
                Init_lr_fit     = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
                Min_lr_fit      = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)
                
                lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)
                    
                for param in model.backbone.parameters():
                    param.requires_grad = True
                            
                epoch_step      = num_train // batch_size
                epoch_step_val  = num_val // batch_size

                if epoch_step == 0 or epoch_step_val == 0:
                    raise ValueError("数据集过小，无法继续进行训练，请扩充数据集。")

                if distributed:
                    batch_size = batch_size // ngpus_per_node

                gen             = DataLoader(train_dataset, shuffle = shuffle, batch_size = batch_size, num_workers = num_workers, pin_memory=True,
                                            drop_last = True, collate_fn = seg_dataset_collate, sampler=train_sampler, 
                                            worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))
                gen_val         = DataLoader(val_dataset  , shuffle = shuffle, batch_size = batch_size, num_workers = num_workers, pin_memory=True, 
                                            drop_last = True, collate_fn = seg_dataset_collate, sampler=val_sampler, 
                                            worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))

                UnFreeze_flag = True

            if distributed:
                train_sampler.set_epoch(epoch)
                
            set_optimizer_lr(optimizer, lr_scheduler_func, epoch)

            fit_one_epoch(model_train, model, loss_history, eval_callback, optimizer, epoch, 
                    epoch_step, epoch_step_val, gen, gen_val, UnFreeze_Epoch, Cuda, 
                    dice_loss, focal_loss, cls_weights, num_classes, fp16, scaler, save_period, save_dir, local_rank)
            
            if distributed:
                dist.barrier()

        if local_rank == 0:
            loss_history.writer.close()