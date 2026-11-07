#用于记录onnx导出是否改变结果

+ 对应checkpoint文件与onnx文件

  + outputs/patchcore_bottle001_split/Patchcore/MVTecAD/bottle/v1/weights/lightning/model.ckpt

  + outputs/onnx/patchcore_bottle_r001_split_v1

+ 软件版本

  + PyTorch版本：2.13.0
  + ONNX Runtime版本：1.22.0
  + 运行设备：CPU

+ 测试图片

  + datasets/MVTecAD/bottle/test/broken_small/000.png

+ 输入完全相同

+ 异常分数误差：

  + 平均值误差：2.384185791015625e-07
  + 最大误差：2.384185791015625e-07

+ 热力图误差：

  + 平均误差：5.071824489277788e-07
  + 最大误差：5.27501106262207e-06

+ 标签、掩码一致

+ 误差容限

  + "rtol": 0.001,

      "atol": 0.0001