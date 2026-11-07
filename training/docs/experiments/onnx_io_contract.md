# 模型调用书名书

+ 输入

输入名称：从模型读取
输入形状：[1, 3, 256, 256]
输入类型：float32
通道顺序：RGB
内存排列：NCHW
输入数值范围：0～1
外部缩放方式：Pillow BILINEAR
ImageNet标准化：已包含在模型中，外部不能重复执行

+ 输出

pred_score：图片异常分数，具体shape和dtype
pred_label：正常/异常标签
anomaly_map：像素异常分数
pred_mask：二值异常区域