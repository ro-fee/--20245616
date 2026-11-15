# 实验二：六类表情分类

类别顺序：Angry、Fear、Happy、Sad、Surprise、Neutral。原始训练图像为 48×48 灰度图，共 28275 张，按 `data/fer_data/fer_data/train/<类别>/*.jpg` 放置。在本目录运行命令。

```sh
pip install -r requirements.txt
python src/audit.py
python src/train_large.py --model efficientnet_b0 --folds all --epochs 22 --batch-size 128 --lr 0.0003 --patience 6
python src/train_large.py --model convnext_tiny --folds all --epochs 20 --batch-size 96 --lr 0.0002 --patience 6
python src/train_large.py --model convnext_base --folds all --epochs 16 --batch-size 32 --lr 0.0001 --patience 5
python experiments/dinov2_large_v1/src/train_dino.py prepare
python experiments/dinov2_large_v1/src/train_dino.py extract
```

训练预处理只读取训练目录，按完全相同及近重复图像建立分组，再进行分层分组五折；同一组不能跨训练与验证折。CNN 的训练脚本只保存训练折验证预测。DINOv2 固定官方 revision，`extract` 只计算冻结骨干特征；随后在每个 fold=0..4 上依次运行 `selftest`、`head`、`smoke`、`train`，均加 `--fold <fold>`。`head` 使用对应训练折训练分类头 30 轮；LoRA 默认 rank=16、alpha=32、query/value 适配和 10 epochs。

专用模型的预训练来源、固定哈希和许可分别见 `experiments/fer_specialists_v1/download_manifest.json` 与 `third_party/`。从其中官方链接取得 HSEmotion 和 POSTER 的 AffectNet 七类预训练文件后执行：

```sh
python prepare_specialists.py --hsemotion /path/to/enet_b2_7.pt --poster /path/to/poster_affectnet7.pth
python experiments/fer_specialists_v1/src/train.py smoke --model hsemotion
python experiments/fer_specialists_v1/src/train.py smoke --model poster
```

随后在五折上分别执行 `train --model hsemotion --fold <fold> --stop-epoch 10` 与 `train --model poster --fold <fold> --stop-epoch 24`。每个模型仅使用对应训练折与验证折；完整模型训练还会检查显存和磁盘预算。原始官方文件经过固定哈希校验，转换后只保存张量字典。

```sh
python evaluate_oof.py
python export_assets.py --output assets
python predict.py --assets assets --test-dir data/test --template data/sample_submission.csv --output outputs/prediction
```

推理只读取指定测试目录，模板为 `ID,Emotion`（ID 为带扩展名的图像文件名）；Emotion 占位值不参与计算。每家族五折平均，再按固定权重融合：EfficientNet-B0 0.07，ConvNeXt-Tiny 0.105，ConvNeXt-Base 0.075，DINOv2 0.25，HSEmotion 0.25，POSTER 0.25。EfficientNet-B0、ConvNeXt-Tiny、POSTER 使用水平翻转平均。导出工具不会捆绑数据、模板或旧预测。
