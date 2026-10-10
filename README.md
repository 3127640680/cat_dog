# 猫狗图像分类系统（Cat-Dog Image Classification）

> 基于 PyTorch + 迁移学习的猫狗图像二分类项目，从零搭建模型到部署上线，共六步渐进式实验。

---

## 项目简介

本项目是一个完整的深度学习图像分类实践，以猫狗二分类为任务，按照"从零训练 → 迁移学习 → 数据增强 → 多模型对比 → 模型保存 → 部署上线"六个步骤逐步迭代，完整覆盖了从数据处理、模型训练到工程部署的全流程。适合作为深度学习入门与课程实验参考。

### 核心特点

- **渐进式实验设计**：六步实验层层递进，每步只改一个变量，公平对比
- **迁移学习实践**：使用 ImageNet 预训练权重，小数据集也能达到高准确率
- **多模型横向对比**：ResNet18 / MobileNetV3-Small / ShuffleNetV2 同台竞技
- **端到端部署**：训练完成后用 Gradio 一键封装为可拍照识别的 Web 应用
- **控制变量法**：固定数据划分（seed=42）与统一评估指标，确保实验可比性

---

## 项目结构

```
.
├── common.py                 # 公共代码：数据加载 + 训练评估函数
├── step1_scratch.py          # 第一步：手动搭建 AlexNet，从零训练
├── step2_pretrained.py      # 第二步：加载 ImageNet 预训练参数 + 冻结
├── step3_augmentation.py     # 第三步：加入数据增强
├── step4_compare.py          # 第四步：多模型横向对比
├── step5_save.py             # 第五步：训练时保存最佳模型
├── step5_test.py             # 第五步：单张图片推理测试
├── step6_app.py              # 第六步：Gradio 拍照分类小程序
├── create_test_data.py       # 生成测试数据集（占位图）
├── datasets/                 # 数据集（cat.*.jpg / dog.*.jpg）
├── 实验结果/                  # 实验结果截图
├── best_model.pth            # 最佳模型权重（ResNet18）
├── model_resnet18.pth        # ResNet18 权重
├── model_mobilenet_v3_small.pth   # MobileNetV3-Small 权重
├── model_shufflenet_v2_x1_0.pth   # ShuffleNetV2 权重
├── 猫狗分类实验指导书.md        # 详细实验指导书
└── README.md
```

---

## 技术栈

| 类别         | 技术                                                  |
| ------------ | ----------------------------------------------------- |
| 深度学习框架 | PyTorch                                               |
| 计算机视觉库 | torchvision                                           |
| 模型架构     | AlexNet、ResNet18、MobileNetV3-Small、ShuffleNetV2    |
| 评估指标     | Accuracy、Precision、Recall、F1-Score（scikit-learn） |
| Web 部署     | Gradio                                                |
| 图像处理     | Pillow (PIL)                                          |

---

## 环境准备

### 依赖安装

```bash
pip install torch torchvision
pip install scikit-learn
pip install pillow
pip install gradio
```

### 数据集

数据集位于 `./datasets` 目录，约 100 张图片，文件名格式为 `cat.1.jpg` / `dog.1.jpg`（标签编码在文件名中：cat → 0，dog → 1）。

如需生成测试占位图：

```bash
python create_test_data.py
```

---

## 六步实验说明

### 第一步：手动搭建模型，从零训练（AlexNet）

从零搭建一个精简版 AlexNet，使用随机初始化的权重在猫狗数据集上训练。

- **模型**：自定义 AlexNet（精简版）
- **初始化**：Xavier 均匀初始化
- **学习率**：0.0001
- **预期准确率**：0.65 ~ 0.80
- **核心结论**：小数据集从零训练难以学到通用视觉特征

```bash
python step1_scratch.py
```

### 第二步：加载 ImageNet 预训练参数（迁移学习）

在手动搭建的 AlexNet 基础上，加载 torchvision 官方 AlexNet 的 ImageNet 预训练权重，冻结卷积层，只训练分类头。

- **迁移方式**：冻结 features，仅训练 classifier
- **核心结论**：迁移学习是小数据集的最优解

```bash
python step2_pretrained.py
```

### 第三步：加入数据增强

在第二步的基础上，对训练集应用数据增强（随机裁剪、水平翻转、色彩抖动、随机旋转），验证集保持干净变换。

| 增强方式             | 模拟的现实变化    |
| -------------------- | ----------------- |
| RandomHorizontalFlip | 猫狗朝左还是朝右  |
| RandomResizedCrop    | 拍摄距离/构图变化 |
| ColorJitter          | 光线、白平衡变化  |
| RandomRotation       | 相机角度轻微倾斜  |

```bash
python step3_augmentation.py
```

### 第四步：多模型横向对比

使用 `torchvision.models` 封装好的预训练模型进行统一对比，冻结特征提取层，仅训练分类头。

| 模型              | 参数量 | 特点                         |
| ----------------- | ------ | ---------------------------- |
| ResNet18          | 11M    | 结构规整，经典入门模型       |
| MobileNetV3-Small | 2.5M   | 深度可分离卷积，为手机端设计 |
| ShuffleNetV2-x1_0 | 2.3M   | 通道混洗，极致轻量           |

```bash
python step4_compare.py
```

### 第五步：模型保存与单张图片推理

训练过程中记录验证集最佳准确率，仅保存 best 模型（早停思想）。支持单张图片推理。

```bash
# 训练并保存最佳模型
python step5_save.py

# 单张图片推理
python step5_test.py best_model.pth ./datasets/cat.1.jpg
```

### 第六步：Gradio 拍照分类小程序

将训练好的模型封装为 Web 应用，支持摄像头拍照和图片上传两种输入方式。

```bash
python step6_app.py
```

启动后：

- 本机访问：`http://127.0.0.1:7860`
- 手机（同一局域网）访问：`http://<电脑IP>:7860`

---

## 实验对比总结

| 步骤          | 核心改动                          | 预期准确率     | 核心结论                           |
| ------------- | --------------------------------- | -------------- | ---------------------------------- |
| 1. 从零训练   | 随机权重                          | 0.65 ~ 0.80    | 小数据集从零训练学不出通用视觉特征 |
| 2. +预训练    | ImageNet 权重 + 冻结              | 0.90 ~ 1.00    | 迁移学习是小数据集的最优解         |
| 3. +数据增强  | 随机翻转/裁剪/色彩                | ≈ 步骤2 或更稳 | 增强提升泛化，数据越多越明显       |
| 4. 多模型对比 | ResNet18 / MobileNet / ShuffleNet | 都在 0.9+      | 小数据下差距小，选型看速度与体积   |
| 5. 保存与推理 | state_dict 保存 + 单图测试        | —              | 训练与推理分离是工程标配           |
| 6. 部署上线   | Gradio 摄像头界面                 | —              | 模型只有被用起来才有价值           |

---

## 核心代码说明

### 数据加载（`common.py`）

- `FileNameDataset`：从文件名解析标签（cat → 0，dog → 1）
- `get_loaders()`：统一划分 80% 训练 / 20% 验证，固定 seed=42 保证各实验可比
- `train_and_eval()`：统一的训练 + 评估函数，输出 Accuracy / Precision / Recall / F1

### 控制变量原则

全实验过程保持：

1. 同一份数据划分（seed=42）
2. 相同的评价指标
3. 每次只改一个变量（网络结构 / 预训练 / 数据增强）

---

## 扩展方向

1. 把冻结改为全模型微调（lr=1e-5），观察准确率和训练时间变化
2. 加入更强的数据增强（如 `RandomErasing`），观察置信度分布变化
3. 用 FastAPI 将 predict 封装为 HTTP 接口，对接微信小程序前端
