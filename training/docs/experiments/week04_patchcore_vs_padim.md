## 本周总结

+ 周一：修改之前的代码
+ 周三：跑通Padim基线

```python
# 以文件配置的方式运行
/usr/bin/time -l uv run anomalib train \
  --config configs/padim_bottle_resnet18.yaml
```

+ 周四：推理批处理
  + 通过模型选择参数 进行对应模型的批推理

```python
# 调用方式
uv run python scripts/predict_batch_C_D.py \
--model Padim \
--input-dir datasets/MVTecAD/bottle/test \
--checkpoint outputs/padim_bottle_baseline/Padim/MVTecAD/bottle/v0/weights/lightning/model.ckpt \
--output-dir outputs/debug_padim

# 对输出的csv进行分析
uv run python scripts/evaluate_csv.py \
--csv-dir xxxx \
--count xxx
```

+ 周五结果分析

| 项目             | PatchCore / WideResNet50-2 / Ratio 0.01 | PaDiM / ResNet18 / 100维 |
| ---------------- | --------------------------------------- | ------------------------ |
| Image AUROC      | 1.0                                     | 0.9968                   |
| Image F1         | 0.9919999999999999                      | 0.976                    |
| Pixel AUROC      | 0.98534                                 | 0.98261                  |
| Pixel F1         | 0.72424                                 | 0.7144                   |
| checkpoint大小   | 112.9                                   | 176.7                    |
| 批量预测调用耗时 | 4.645                                   | 3.264                    |
| 误报数、漏检数   | FP=0 ｜ FN=1                            | FP = 1 ｜FN = 2          |

+ 周六详细剖析输入、预处理、输出、后处理等相关值的类型、数值、范围等

