# RT-IceNet

根据 `manuscript.docx` 第 2 节整理的语义分割模型。手稿保留在项目中。

## 论文名称与代码对应

| 文章名称 | 正式类名 | 文件 | 原代码对应 |
| --- | --- | --- | --- |
| RT-IceNet | `RTIceNet` | `rt_icenet.py` | `DeepLab` |
| High-frequency Feature Extraction Module | `HFEM` | `hfem.py` | `SFF`，已加入 SKFM |
| Selective Kernel Fusion Module | `SKFM` | `skfm.py` | 原代码缺失，本次补齐 |
| Semantic Information Enhancement Module | `SIEM` | `siem.py` | `PAPPM` + `EfficientAttention` 的组合 |
| Parallel Attention Feature Fusion Module | `PAFM` | `pafm.py` | `CGAFusion` |
| MobileNetV2-derived backbone | `MobileNetV2Backbone` | `backbone.py` | 原主模型里的 `MobileNetV2` 包装类 |

`mobilenetv2.py` 中的分类网络仍叫 `MobileNetV2`，其倒残差块仍叫 `InvertedResidual`。金字塔池化和局部/全局注意力的全部实现已整合到 `siem.py`，对外仅提供 `SIEM`；原 `pappm.py`、`CloAttention.py` 已删除。空间/通道/像素注意力仍是 `pafm.py` 的内部组件。

## 网络流程

```text
输入 → MobileNetV2Backbone（一次前向）
        ├─ 1/2、1/4、1/8 特征 → HFEM（两个 SKFM）→ 1×1 投影 ─┐
        └─ 深层特征 → SIEM（PAPPM → 局部/全局注意力）→ 上采样 ─┤
                                                             PAFM
                                                               ↓
                                                     分类卷积 → 恢复输入尺寸
```

默认四分类、骨干输出步长为 8，对应手稿中三次下采样与膨胀率 2、4 的配置。可显式使用 `downsample_factor=16` 做对比。类别顺序需要由数据集定义，手稿片段未给出标签编号。

HFEM 通道选择：1/8 特征由 32 投影到 64 通道；第一次上采样得到 32 通道，与 1/4 跳接的 32 通道进行 SKFM 融合；第二次上采样得到 16 通道，与 1/2 跳接的 16 通道融合；最后输出 24 通道。对于奇数尺寸，上采样结果对齐到跳接特征尺寸。

SKFM 依照第 2.1 节：两路求和、全局平均池化、MLP、沿两路分支做 softmax、加权求和。手稿文字和公式没有给出 MLP 隐藏宽度，代码采用 `max(channels // 8, 4)`，可配置；上述通道配置属于实现选择，不能视为已核实的原始实验配置。

## 使用

安装依赖：`pip install -r requirements.txt`。

```python
import torch
from rt_icenet import RTIceNet

model = RTIceNet(num_classes=4, pretrained=False).eval()
x = torch.randn(1, 3, 512, 512)
with torch.no_grad():
    logits = model(x)                    # [1, 4, 512, 512]
    prediction = logits.argmax(dim=1)    # [1, 512, 512]
```

输出是 logits，可直接用于交叉熵损失。`pretrained=True`（保留原默认值）会在本地缓存不存在时下载 MobileNetV2 权重；以上示例和测试均关闭下载。训练模式下，SIEM 的全局池化分支含 BatchNorm，单设备批量需大于 1；单张推理使用 `eval()`。

## 结构整理与兼容性

SIEM 本次文件整合保留参数名称、形状和计算逻辑，整合前的 SIEM `state_dict` 可以严格加载；推理和训练模式下的输出、梯度及状态已验证完全一致。下述旧权重限制针对最初的网络结构调整。

- 已删除旧兼容入口、重复演示、旧名称别名和备份包；统一从 `rt_icenet.py` 及各正式模块导入。
- 删除从未参与前向计算的 `cat_conv`；四次重复骨干调用合并为一次，因此训练时 BatchNorm 更新次数也随之改变。
- 依据手稿公式（14），PAFM 的门控只执行一次 sigmoid，移除原来的重复 sigmoid。
- 全局注意力用实际池化后的高宽计算 token 数，支持测试中的非整倍数输入尺寸。过小输入仍受注意力池化窗口及反射填充限制，不承诺任意尺寸。
- 移除没有提供实现、也不符合此 HFEM 通道接口的 Xception 分支，选择其他骨干会明确报错。
- `deeplabv3_training.py` 的损失和学习率工具未改变；其中损失函数默认 `num_classes=21`，四分类训练调用 CE/Focal 时应显式传入 `num_classes=4`。

**旧完整模型权重不能直接严格加载。** 名称前缀由 `pappm.*` / `cloattention.*` / `sff.*` / `block.*` 分别变为 `siem.pappm.*` / `siem.attention.*` / `hfem.*` / `pafm.*`；HFEM 部分参数形状也发生变化且新增 SKFM。`strict=False` 不会跳过同名但形状不同的参数。新结构应重新训练，或自行按名称和形状筛选迁移权重。

## 验证

```powershell
python -m unittest test_rt_icenet -v
```

覆盖 SKFM 等权平均、相同输入保持、输入梯度、PAFM 单次 sigmoid 的等权门控、8/16 倍骨干的规则及非整倍数输入、一次骨干调用、所有可训练参数的有限梯度。

这些是结构及运行验证，不代表重新训练后的准确率、实时速度或文章指标已经复现。
