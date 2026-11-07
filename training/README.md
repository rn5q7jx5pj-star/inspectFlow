+ 当前项目目的
  + 跑通第一个程序 全流程熟悉+基础知识复盘
+ 环境创建方法
  + 使用uv创建环境 主要是python版本和一些库函数
+ 基线实验命令 略
+ 输出目录说明
  + outputs：模型-数据集-数据集名字-v1
  + 有训练识别的图片images
  + .ckpt文件有训练的结果模型权重weights，==文件较大==，通常可能包含：
    - 模型参数，也就是神经网络的权重。
    - 模型结构所需的部分配置信息。
    - 当前Epoch和训练步数。
    - 优化器状态。
    - 学习率调度器状态。
    - 训练过程中记录的指标
    - 用途
      + 训练中断后继续
      + 使用训练好的模型推理
      + 导出成其他模型格式
  + .yaml有保存配置数据的文本格式,把容易修改的配置从Python代码中分离出来，==相当于输入保存下来==

```bash
# 内容类似为
model:
  class_path: anomalib.models.Patchcore
  init_args:
    backbone: wide_resnet50_2
    coreset_sampling_ratio: 0.1

data:
  class_path: anomalib.data.MVTecAD
  init_args:
    root: ./datasets/MVTecAD
    category: bottle
    train_batch_size: 8
    eval_batch_size: 8

trainer:
  accelerator: cpu
  devices: 1
  default_root_dir: ./outputs/patchcore_bottle
  
# 然后通过配置文件运行：不需要每次输入一长串命令，也方便记录和复现实验
uv run anomalib train --config configs/patchcore_bottle.yaml
```

+ 当前进度
  + 完成了bottle数据集coreset_sampling_ratio=0.1、0.05、0.01的对比实验
  + 结论是三者相比性能指标相差不大 但运行载荷显著降低
+ 数据集说明
  + 从网站下载的MVTecAD数据集
  + 目前只跑了bottle的209张图 reload_dataloaders_every_n_epochs：0不按周期重建DataLoader，训练轮数由其他配置决定。
+ 基线命令在上面
+ 单图推理

```bash
uv run python scripts/predict_single.py \
  --image test.png \
  --checkpoint model.ckpt \
  --output-dir outputs/result
```

+ 目录结构 略
+ 已有实验结果（见当前进度）

#### 一次推理多张图片 并以csv保存

```python
# 代码讲解
#1 Anomalib内部使用了：path.glob("**/*") 递归查找子目录，再筛选支持的图片文件。 因此只要出入图片的根目录就可以执行推理
input_dir = "/Users/yangfan/CS/InspectFlow/training/datasets/MVTecAD/bottle/test"
#2 解释推理模型的参数
pre_trained=False,# 预训练模型 这里指创建模型时不加载那套通用预训练权重 而是从checkpoint还原模型权重参数
num_neighbors=9, # 模型找到图片中最异常的局部区域后，会参考正常特征库中的邻域信息，对整张图片的异常分数进行加权。 相当于在特定部分采用9临域加权
visualizer=False,# 关闭Anomalib自动生成、保存可视化图片的功能。预测仍然会产生异常分数、热力图数值和预测掩码，只是不自动把它们画成图片保存。这里只要保存为csv格式
```

+ 代码在/Users/yangfan/CS/InspectFlow/training/scripts/predict_batch.py
+ 完成输出csv的分析处理
+ 完成推理过程的时间、数值分析
+ 完成同一代码实现两种模型的批推理，并实现独立分析
+ 完成输入、预处理、输出、后处理等变量属性分析

+ 第五周
  + 完成训练集、验证集、测试集的划分和验证
  + 完成数据集分割后的模型训练
  + 完成onnx的验证（与torch推理的输出相同）

+ 第六周
  + 完成C++/onnx的输入输出集的提取
  + 完成C++工程的搭建