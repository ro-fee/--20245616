# 实验三：中央凹定位

保留优化版 ResNet18 粗细两阶段流程。80 张有标注训练图像、20 张无标签测试图像按原始 detection 数据结构放入 `data/detection`；训练坐标文件为 `fovea_localization_train_GT.csv`，模板列为 `ImageID,value`。需官方 `resnet18-f37072fd.pth`，来源为 `https://download.pytorch.org/models/resnet18-f37072fd.pth`，加载时验证 SHA-256。

在本目录运行：

```sh
pip install -r requirements.txt
python run.py train --data data/detection --labels data/detection/fovea_localization_train_GT.csv --template data/detection/sample_submission.csv --pretrained weights/resnet18-f37072fd.pth --output-dir runs/fovea
python run.py predict --test-dir data/detection/test --template data/detection/sample_submission.csv --models runs/fovea --output outputs/submission.csv
```

训练使用五个外层分组折；早停在各训练折内部进行。精定位训练中心由内部交叉拟合产生。固定使用 0.20、0.30、0.40 三种局部尺度，并将粗定位与多尺度精定位各以 0.5 权重融合。坐标转换与水平翻转还原在原始像素坐标中完成。

测试帧只传递图像 ID、路径、宽高；`predict` 入口不接受训练坐标或测试答案文件。样例 notebook 仅调用同目录源码，无输出或重复嵌入实现。`run.py predict` 的测试目录应指向图像所在目录，可含子目录。
