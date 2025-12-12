# PyTorch深度学习

PyTorch是当前最流行的深度学习框架之一，以其动态计算图、易用性和强大的生态系统而受到广泛欢迎。本板块包含25篇PyTorch核心技术详解，涵盖从基础到进阶的完整内容。

## � PyTorch核心基础

### 🔥 张量与计算图
- [PyTorch框架Tensor张量基础](./深度学习Pytorch框架Tensor张量.md)
  - Tensor基本概念与数据类型
  - 张量创建与操作
  - 设备管理与数据迁移

- [Tensor属性与算术运算](./深度学习Pytorch-Tensor的属性、算术运算.md)
  - 张量属性与维度操作
  - 算术运算与数学函数
  - 广播机制与索引

- [Tensor函数详解](./深度学习Pytorch-Tensor函数.md)
  - 张量变换函数
  - 数学运算函数
  - 统计与比较函数

### 🧠 自动求导机制
- [PyTorch与autograd自动求导](./Pytorch与autograd自动求导.md)
  - 计算图与自动求导原理
  - 梯度计算与累积
  - 自定义梯度与求导技巧

### 📦 核心模块
- [PyTorch核心模块详解](./Pytorch详解-Pytorch核心模块.md)
  - nn.Module基类
  - 常用网络层
  - 模型构建流程

- [PyTorch数据模块](./Pytorch详解-数据模块.md)
  - Dataset与DataLoader
  - 数据预处理与增强
  - 自定义数据集

- [nn库与nn.functional区别](./Pytorch torch.nn库以及nn与nn.functional有什么区别？.md)
  - nn模块与函数式API对比
  - 使用场景与最佳实践
  - 性能优化建议

## 🚀 模型构建与训练

### 🏗️ 模型模块
- [模型模块详解(RNN,CNN,FNN,LSTM,GRU,TCN,Transformer)](./Pytorch详解-模型模块(RNN,CNN,FNN,LSTM,GRU,TCN,Transformer).md)
  - 前馈神经网络
  - 卷积神经网络
  - 循环神经网络与变体
  - Transformer架构

### ⚙️ 优化与训练
- [PyTorch优化模块](./PyTorch详解-优化模块.md)
  - 优化器原理与实现
  - 学习率调度器
  - 训练策略与技巧

- [深度学习优化方法](./深度学习面试笔试之深度学习的优化方法.md)
  - 梯度下降变体
  - 动量与自适应学习率
  - 正则化与泛化

### � 模型保存与部署
- [模型保存、加载与分布式](./Pytorch_模型的保存_加载、并行化、分布式.md)
  - 模型保存与加载策略
  - 多GPU并行训练
  - 分布式训练框架

- [模型保存与加载进阶](./Pytorch详解-模型保存与加载、Finetune 模型微调、GPU使用、nvidia-smi详解、TorchEnsemble 模型集成库、torchmetrics 模型评估指标库.md)
  - 模型微调与迁移学习
  - GPU使用与监控
  - 模型集成与评估

## 🎨 可视化与调试

### � 可视化工具
- [PyTorch可视化工具](./Pytorch可视化Visdom、tensorboardX和Torchvision.md)
  - Visdom交互式可视化
  - TensorBoardX集成
  - TorchVision应用

- [PyTorch可视化模块](./PyTorch详解-可视化模块.md)
  - 模型结构可视化
  - 训练过程可视化
  - 结果分析与调试

## 🔬 网络架构实战

### 🖼️ 经典CNN模型
- [从零搭建经典CNN模型](./从零搭建GoogLeNet，ResNet18，ResNet50，vgg、mobilenetv1、mobilenetv2、shufflenetv1、shufflenetv2模型（Pytorch代码示例）.md)
  - GoogLeNet
  - ResNet系列
  - VGG、MobileNet、ShuffleNet

### 🌟 注意力与Transformer
- [从零搭建Attention模型](./从零搭建CBAM、SENet、STN、transformer、mobile_vit、simple_vit、vit模型（Pytorch代码示例）.md)
  - CBAM、SENet注意力机制
  - STN空间变换网络
  - Transformer、ViT、MobileViT

## 🎯 计算机视觉应用

### � 图像识别
- [PyTorch与卷积神经网络(OpenCV)](./Pytorch与卷积神经网络(OpenCV).md)
  - CNN与OpenCV结合
  - 图像处理流程
  - 计算机视觉应用

- [视觉识别技术](./视觉识别：ffmpeg-python、ultralytics.YOLO、OpenCV-Python、标准RTSP地址格式.md)
  - YOLO目标检测
  - OpenCV图像处理
  - RTSP视频流处理

### 👤 人脸识别
- [人脸识别：face_recognition参数详解](./人脸识别：face_recognition参数详解.md)
  - face_recognition库使用
  - 人脸识别算法
  - 人脸特征提取

## 🎙️ 语音处理

### � 语音识别技术
- [语音识别：PyAudio、SoundDevice、Vosk、openai-whisper](./语音识别：PyAudio、SoundDevice、Vosk、openai-whisper、Argos-Translate、FunASR（Python）.md)
  - 音频采集与处理
  - 语音识别模型
  - 多语言翻译

## 📚 深度学习理论

### 🧠 神经网络基础
- [前向神经网络与反向传播](./深度学习面试笔试之前向神经网络-多层感知器、损失函数、反向传播.md)
  - 多层感知器原理
  - 损失函数设计
  - 反向传播算法推导

### � 卷积神经网络
- [深度学习面试笔试之卷积神经网络(CNN)](./深度学习面试笔试之卷积神经网络(CNN).md)
  - CNN架构与原理
  - 卷积操作与池化
  - 经典CNN模型分析

### � 循环神经网络
- [循环神经网络、GRU与LSTM](./深度学习面试笔试之循环神经网络(RNN)、门控循环单元（GRU）、长短期记忆(LSTM).md)
  - RNN基本原理
  - GRU与LSTM门控机制
  - 序列模型应用

### 🔀 迁移与强化学习
- [迁移学习、强化学习与多任务](./深度学习面试笔试之迁移学习(Transfer)、强化学习(Reinforcement) & 多任务.md)
  - 迁移学习策略
  - 强化学习基础
  - 多任务学习框架

## 🌟 图深度学习

### 📊 图神经网络
- [图深度学习、A*算法、EMD和VMD](./图深度学习、A_（A-Star）算法、EMD和VMD详解.md)
  - 图神经网络基础
  - A*路径规划算法
  - 经验模态分解技术

---

*从Tensor基础到Transformer模型，掌握PyTorch深度学习全栈技术！*