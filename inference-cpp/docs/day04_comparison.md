# 第4天：直接比较Python与C++结果目录

在 `/Users/yangfan/CS/InspectFlow/inference-cpp` 中执行。使用训练环境的Python，因为该环境已经安装NumPy；不需要安装额外库，也不执行模型推理。

```bash
../training/.venv/bin/python scripts/compare_result_dirs.py \
  --python-dir ../training/outputs/week06_io/good015 \
  --cpp-dir outputs/day03_good015 \
  --report outputs/comparisons/good015.json
```

异常样例：

```bash
../training/.venv/bin/python scripts/compare_result_dirs.py \
  --python-dir ../training/outputs/week06_io/broken_small015 \
  --cpp-dir outputs/day03_broken_small015 \
  --report outputs/comparisons/broken_small015.json
```

两个参数均指向单个样例的结果文件夹，脚本读取其中的 `prediction.json`、`anomaly_map.f32`、`pred_mask.u8`。Python目录还要有其input_format.file指定的输入文件；C++记录的input_path必须可访问。模型路径从两端JSON获取。现有数据使用绝对模型路径；相对记录路径从当前目录或结果目录解析，有歧义则报错。

`--report`可省略，此时只打印检查结果，不写文件。指定时会创建父目录，报告已存在则拒绝覆盖。默认 `--rtol 1e-3 --atol 1e-4`。

## 比较顺序

1. 对主ONNX文件及两份输入文件计算SHA-256，判断当前内容是否相同。
2. 核对float32、小端、RGB、NCHW和输入shape，并检查字节数及数值范围。
3. 只将 `[1,1,H,W]` 与 `[H,W]` 规范为同一种单图布局，其他shape不静默调整。
4. 浮点输出使用 `abs(cpp-python) <= atol + rtol*abs(python)`，打印平均、最大误差和超容差元素数。
5. 标签要求一致，掩码要求逐像素一致；掩码只能含0或1。
6. 对照掩码实际统计值检查JSON里的异常像素数。

返回码：0全部通过；1发现内容/数值不一致；2缺少文件、协议错误、报告已存在等导致无法正常比较。

## 限制

- 旧结果没有推理时的文件哈希。本次SHA-256只能核验当前文件，无法证明它在推理后从未变更。
- 裸二进制本身没有dtype/shape信息，必须依赖双方写入协议、JSON和字节数。JSON的分数/标签也不保留原始Tensor dtype，脚本不会声称独立恢复了这种信息。
- 比较主ONNX文件，不包含外部权重文件。当前模型之外若使用外部数据，需要额外核验。
- 比较结果一致性，不重新评估准确率，也不验证原始图片的预处理一致性。

## 测试

```bash
../training/.venv/bin/python -m unittest discover -s tests -p 'test_compare_result_dirs.py' -v
```

测试使用临时构造的数据，不修改真实结果目录。覆盖shape规范化、不同模型/输入、热力图误差、掩码差异、截断文件、非法标签、NaN和广播shape错误。

2026-09-24实测：good015和broken_small015两组结果全部通过，分数误差和热力图最大误差均为0，掩码不一致像素数为0。
