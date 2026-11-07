# PatchCore配置文件重放实验

## 实验目的

验证保存的YAML配置能否独立复现第一次PatchCore基线。

## 使用配置

configs/patchcore_bottle_r010.yaml

## 输出目录

outputs/patchcore_bottle_r010_replay/Patchcore/MVTecAD/bottle/v2

## 测试指标

- Image AUROC：1.0
- Image F1：0.992
- Pixel AUROC：0.986
- Pixel F1：0.726

## 性能

- 总运行时间：约5min
- 峰值常驻内存：
- checkpoint大小：231MB

## 与第一次基线比较

| 指标        | 第一次基线 | 配置重放 | 差值 |
| ----------- | ---------: | -------: | ---: |
| Image AUROC |     1.0000 |          |      |
| Image F1    |     0.9920 |          |      |
| Pixel AUROC |     0.9856 |          |      |
| Pixel F1    |     0.7272 |          |      |

## 结论

配置文件**能够**/不能够独立运行。

本轮输出**没有**覆盖第一次基线。

两次结果**是**否接近：

可能产生差异的原因：