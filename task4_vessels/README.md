# 实验四：眼底血管分割

数据目录为 `<data>/train/image/*.jpg` 与 `<data>/train/label/*.jpg`，共 20 张训练图像及其二值掩膜。图像按文件内容解码，支持原始数据扩展名与实际编码不同的情况。标签仅在训练入口读取。测试图像另放一处，由推理命令明确指定。

在本目录运行：

```sh
pip install -r requirements.txt
python train_families.py --data data/segmentation --runs-root runs --mode both --dry-run
python train_families.py --data data/segmentation --runs-root runs --mode both --encoder-weights weights/efficientnet-b3.safetensors
python predict.py --test-dir data/segmentation/test/image --template data/segmentation/sample_submission.csv --runs-root runs --output outputs/submission.csv
```

ResNet34 预训练权重从 PyTorch 官方下载。EfficientNet-B3 使用与 `segmentation_models_pytorch` 编码器匹配的官方 ImageNet 权重；可从 SMP 对应的 Hugging Face 模型 `smp-hub/efficientnet-b3.imagenet` 获取 `model.safetensors`，保存为命令指定文件。对应 revision 为 `074c54a6c473e0d294690d49cedb6cf463e7127d`，文件 SHA-256 为 `d95605b0390229a5aa0431a4f8231f175a31ce7a86b7a75200b8702207245253`，加载前严格检查。不要使用竞赛训练模型替代通用编码器预训练权重。

六个家族配置见 `families.json`。每家族包含五折模型及一个全量训练模型。折划分固定为 seed=0；暖启动仅允许相同训练数据、相同划分和相同 fold/full 类型，不能从全量训练模型初始化验证折。`--family` 可单独运行家族，但须先完成其依赖家族。

发布版在每个家族内部等权平均六个模型及 D4 翻转/转置预测，随后将六个家族的概率图以 PIL 双线性插值对齐至 512×512，固定等权平均并使用 0.5 阈值。该规则不读取历史选参文件。训练指标只计算训练折内部 Dice，推理不提供评分或阈值搜索入口。

提交格式为 `Id,Predicted`，RLE 按行展开、一开始计数；模板只读取 Id，Predicted 占位值被替换。`--dry-run` 只打印训练命令，不启动训练或下载。
