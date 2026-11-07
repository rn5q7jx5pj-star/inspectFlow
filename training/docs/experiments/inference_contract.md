+ 输入
  + 图片格式png
    + torch.Size([3, 900, 900])
  + RGB通道
    + 读取函数放在：/Users/yangfan/CS/InspectFlow/training/.venv/lib/python3.12/site-packages/anomalib/data/utils/image.py
  + 读取后dataset的dtype为<class 'anomalib.data.predict.PredictDataset'>
    + 其中dataset[0]的dtype为<class 'anomalib.data.dataclasses.torch.image.ImageItem'>
    + 数据类型：torch.float32
  + 数值范围为0～1
  
+ 预处理
  
  + 路径位置：/Users/yangfan/CS/InspectFlow/training/.venv/lib/python3.12/site-packages/anomalib/pre_processing/pre_processor.py
  
  + 在on_predict_batch_start函数中进行缩放、标准化
  + 最终尺寸是1 *3 *256 *256
  
+ 输出：
  
  + pred_score：<class 'torch.Tensor'> tensor([[0.6846]])
  + pred_label：<class 'torch.Tensor'> tensor([True]) torch.bool
  + anomaly_map：<class 'torchvision.tv_tensors._mask.Mask'>：torch.Size([1, 256, 256]) 数值0～1
  + pred_mask：<class 'torchvision.tv_tensors._mask.Mask'> torch.Size([1, 256, 256])
  + shape为[n,1]

+ 后处理：
  + 阈值从哪里来
    + 阈值在后处理对象中model.post_processor中
    + 原始分数尺度下的阈值：
      + 图片：49.429378509521484
      + 像素：33.10508728027344
    + 你当前后处理会将分数归一化。默认敏感度下，归一化后的决策阈值是0.5
      + 图片：0.5
      + 像素：0.5
  + 热力图颜色范围
    + 使用jet色表，每张图片独立进行min-max归一化，颜色表示本图内部的相对异常程度。
    + 固定色标版本：使用jet色表，直接映射后处理后的0～1异常分数。

+ 计时：

| 包含                             | 不包含                 |
| -------------------------------- | ---------------------- |
| `Engine.predict()`内部的预测准备 | Python启动、导入依赖   |
| 从checkpoint恢复模型状态         | 创建模型、Engine对象   |
| 读取图片、组成Batch              | 前面收集、排序图片路径 |
| 图片预处理                       | 后面整理CSV行          |
| 模型计算与后处理                 | 保存CSV和JSON          |
| 整理并返回预测结果               | 后续打印、评测         |
