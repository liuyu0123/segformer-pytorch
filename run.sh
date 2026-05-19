#训练模型
python train.py
#训练模型（水域分割）
python train_water.py
#训练模型（水域分割，train和val分开）
#用法1：独立 train/val 路径✅
python train_water_val.py `
    --images D:\Files\Data\IRWSB\train\images `
    --masks D:\Files\Data\IRWSB\train\masks_pspnet `
    --val-images D:\Files\Data\IRWSB\val\images `
    --val-masks D:\Files\Data\IRWSB\val\masks_pspnet `
    --epochs 5 `
    --batch-size 4
#用法2：只指定 train，自动划分 val
python train_water_val.py `
    --images D:\Files\Data\IRWSB\images `
    --masks D:\Files\Data\IRWSB\masks `
    --val-split 0.1 `
    --epochs 5
#用法3：其他常用参数
python train_water_val.py `
    --images D:\Files\Data\IRWSB\train\images `
    --masks D:\Files\Data\IRWSB\train\masks_pspnet `
    --val-images D:\Files\Data\IRWSB\val\images `
    --val-masks D:\Files\Data\IRWSB\val\masks_pspnet `
    --epochs 100 `
    --freeze-epochs 10 `
    --batch-size 4 `
    --freeze-batch-size 8 `
    --lr 1e-2 `
    --optimizer sgd `
    --backbone mobilenet `
    --input-height 320 `
    --input-width 640 `
    --save-dir logs_irwsb `
    --mask-ext .png `
    --no-freeze          # 跳过冻结训练阶段

#训练模型（水域分割，train和val分开，记录model和csv路径）
python train_water_val_pro.py `
    --images D:\Files\Data\IRWSB\train\images `
    --masks D:\Files\Data\IRWSB\train\masks_pspnet `
    --val-images D:\Files\Data\IRWSB\val\images `
    --val-masks D:\Files\Data\IRWSB\val\masks_pspnet `
    --epochs 5 `
    --batch-size 4 `
    --learning-rate 1e-4 `
    --model-dir checkpoints/exp_segformer_b0 `
    --model-name segformer_b0_exp01 `
    --log-dir logs/exp_segformer_b0 `
    --log-name segformer_b0_exp01 `
    --save-interval 0


#测试模型（水域分割，train和val分开）
# 基础测试✅
python test_water.py `
    --model logs/checkpoint_best.pth `
    --images D:\Files\Data\IRWSB\test\images `
    --masks D:\Files\Data\IRWSB\test\masks_pspnet
# 指定模型配置和输出路径
python test_water.py `
    --model logs/checkpoint_best.pth `
    --images D:\Files\Data\IRWSB\test\images `
    --masks D:\Files\Data\IRWSB\test\masks_pspnet
    --backbone mobilenet `
    --num-classes 2 `
    --input-height 320 `
    --input-width 640 `
    --batch-size 4 `
    --output results/pspnet_test.csv


#测试模型
python predict.py
#测试模型（水域分割）
python predict_water.py

#测试模型pro（水域分割，生成红色mask蒙版和csv评价指标）
python predict_water_best_pro.py `
    --input "D:\Files\Data\IRWSB\analyse\images" `
    --weights "F:\AAA\5_segformer_best\experiment1\experiment1_last.pth" `
    --output "D:\Files\GitProject\segformer-pytorch-LY\results_segformer" `
    --ground_truth "D:\Files\Data\IRWSB\analyse\masks_pspnet" `
    --alpha 0.5 `
    --phi b0 `
    --max_side 512