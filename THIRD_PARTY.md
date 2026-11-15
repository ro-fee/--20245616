# 第三方来源

保留第三方文件的原始版权及许可；本仓库的整理不改变其授权条件。数据与模型权重不在 Git 中分发，使用时还需遵守各来源的条件。

| 内容 | 上游 | 本地许可 |
|---|---|---|
| DINOv2 官方骨干及所需层 | https://github.com/facebookresearch/dinov2 | `task1_plants/vendor/dinov2/LICENSE`（Apache-2.0） |
| POSTER V2 模型结构 | https://github.com/Talented-Q/POSTER_V2/tree/18de5591c3fa0b7b22bb9fe2d61e7f813e6e3b08 | `task2_fer/third_party/poster/LICENSE`（MIT） |
| HSEmotion / EmotiEffLib 预训练来源 | https://github.com/sb-ai-lab/EmotiEffLib/tree/520a051c64cd191521e5934655314e769a319684 | `task2_fer/third_party/hsemotion/LICENSE`（Apache-2.0） |

DINOv2 保留所需官方 Python 源文件并提供最小 hub 入口；POSTER 运行源码移除了未执行的旧调试路径和过期下载注释。其他第三方组件通过依赖文件安装：PyTorch、torchvision、timm、Transformers、PEFT、SMP、Albumentations 等，遵守各自发行版许可。
