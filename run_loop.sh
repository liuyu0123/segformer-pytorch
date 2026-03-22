#!/bin/bash

# 固定基础路径 (注意在Bash中建议使用正斜杠 / 或者双反斜杠 \\)
# 如果你在 Git Bash 中运行，建议把 D:\ 改为 /d/
BASE_DATA_PATH="D:/Files/Data/IRWSB"
TRAIN_IMG="${BASE_DATA_PATH}/train/images"
TRAIN_MASK="${BASE_DATA_PATH}/train/masks_white"
VAL_IMG="${BASE_DATA_PATH}/val/images"
VAL_MASK="${BASE_DATA_PATH}/val/masks_white"

# 默认超参数值 (控制变量法的基准)
DEF_EPOCHS=50
DEF_BS=4
DEF_LR="1e-3"

# 训练函数
run_train() {
    local name=$1
    local epochs=$2
    local bs=$3
    local lr=$4
    
    echo "========================================"
    echo "Start Experiment: $name"
    echo "Settings: Epochs=$epochs, BS=$bs, LR=$lr"
    echo "========================================"
    
    python train_water.py \
        --images "$TRAIN_IMG" \
        --masks "$TRAIN_MASK" \
        --val-images "$VAL_IMG" \
        --val-masks "$VAL_MASK" \
        --epochs "$epochs" \
        --batch-size "$bs" \
        --learning-rate "$lr" \
        --model-dir "checkpoints/$name" \
        --model-name "$name" \
        --log-dir "logs/$name" \
        --log-name "$name" \
        --save-interval 0
}

# 1. 遍历 Epochs (固定 BS=4, LR=1e-3)
# 列表: 1, 5, 25, 50, 100, 150
for e in 1 5 25 50 100 150; do
    exp_name="exp_epoch_${e}"
    # 注意：这里使用了默认的 BS 和 LR
    run_train "$exp_name" "$e" "$DEF_BS" "$DEF_LR"
done

# 2. 遍历 Batch Size (固定 Epochs=50, LR=1e-3)
# 列表: 1, 2, 3, 4, 5, 6
for b in 1 2 3 4 5 6; do
    exp_name="exp_bs_${b}"
    run_train "$exp_name" "$DEF_EPOCHS" "$b" "$DEF_LR"
done

# 3. 遍历 Learning Rate (固定 Epochs=50, BS=4)
# 列表: 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2
for l in 1e-4 5e-4 1e-3 5e-3 1e-2 5e-2; do
    exp_name="exp_lr_${l}"
    run_train "$exp_name" "$DEF_EPOCHS" "$DEF_BS" "$l"
done

echo "All Experiments Completed!"
