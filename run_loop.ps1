# 固定基础路径
$TrainImg = "D:\Files\Data\IRWSB\train\images"
$TrainMask = "D:\Files\Data\IRWSB\train\masks_pspnet"
$ValImg = "D:\Files\Data\IRWSB\val\images"
$ValMask = "D:\Files\Data\IRWSB\val\masks_pspnet"

# 默认超参数 (控制变量法的基准)
$DefEpochs = 50
$DefBs = 4
$DefLr = "1e-3"

# 定义运行训练的函数
function Run-Train {
    param (
        [string]$ExpName,
        [int]$Epochs,
        [int]$Bs,
        [string]$Lr
    )

    Write-Host "========================================" -ForegroundColor Green
    Write-Host "Start Experiment: $ExpName" -ForegroundColor Green
    Write-Host "Settings: Epochs=$Epochs, BS=$Bs, LR=$Lr" -ForegroundColor Green
    Write-Host "========================================" -ForegroundColor Green

    # 构建保存路径
    $ModelDir = "checkpoints\$ExpName"
    $LogDir = "logs\$ExpName"

    # 运行 Python 脚本
    python train_water_val_pro.py `
        --images $TrainImg `
        --masks $TrainMask `
        --val-images $ValImg `
        --val-masks $ValMask `
        --epochs $Epochs `
        --batch-size $Bs `
        --learning-rate $Lr `
        --model-dir $ModelDir `
        --model-name $ExpName `
        --log-dir $LogDir `
        --log-name $ExpName `
        --save-interval 0
}

# 1. 遍历 Epochs (固定 BS=4, LR=1e-3)
# 列表: 1, 5, 25, 50, 100, 150
$EpochsList = @(1, 5, 25, 50, 100, 150)
foreach ($e in $EpochsList) {
    $Name = "exp_epoch_$e"
    Run-Train -ExpName $Name -Epochs $e -Bs $DefBs -Lr $DefLr
}

# 2. 遍历 Batch Size (固定 Epochs=50, LR=1e-3)
# 列表: 1, 2, 3, 4, 5, 6
$BsList = @(1, 2, 3, 4, 5, 6)
foreach ($b in $BsList) {
    $Name = "exp_bs_$b"
    Run-Train -ExpName $Name -Epochs $DefEpochs -Bs $b -Lr $DefLr
}

# 3. 遍历 Learning Rate (固定 Epochs=50, BS=4)
# 列表: 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2
$LrList = @("1e-4", "5e-4", "1e-3", "5e-3", "1e-2", "5e-2")
foreach ($l in $LrList) {
    $Name = "exp_lr_$l"
    Run-Train -ExpName $Name -Epochs $DefEpochs -Bs $DefBs -Lr $l
}

Write-Host "All Experiments Completed!" -ForegroundColor Cyan
