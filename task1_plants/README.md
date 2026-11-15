# 实验一：植物分类

在本目录运行命令。将原始 500 张训练图像按 `train/<类别>/<文件名>` 放置；`original_train_manifest.json` 只记录训练样本身份与哈希，用于检查输入及固定划分。五类顺序定义在 `common.py`。

```sh
pip install -r requirements.txt
python bootstrap.py
python run_pipeline.py
python predict.py --test-dir test --template data/sample_submission.csv
```

`bootstrap.py` 下载官方通用预训练权重，DINOv2 源码位于 `vendor/dinov2`。训练流程在原始训练数据中划出 100 张留出样本，400 张开发数据进行四折交叉验证，确定配置后完成留出评估和 500 张全量训练。测试图像仅在 `predict.py` 中读取，不进行测试标签比对。

模板格式为 `ID,Category`，ID 是相对测试目录的图像文件名；Category 占位值会被预测替换。推理输入需放在当前实验目录以内。正式训练和推理需要 CUDA；输出写入本目录，已加入 Git 忽略规则。
