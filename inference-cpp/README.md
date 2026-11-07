# InspectFlow C++ ONNX推理：第3天

支持打印模型信息，以及读取Python准备的输入Tensor完成CPU推理。

## 构建

在本目录运行，使用现有macOS ARM64 ONNX Runtime 1.30.0 SDK：

```bash
cmake -S . -B build
cmake --build build -j 2
```

依赖不在默认位置时，配置命令增加 `-DONNXRUNTIME_ROOT=/实际SDK目录`。

## 正常样例

```bash
./build/inspectflow_infer \
  --model ../training/outputs/onnx/patchcore_bottle_r001_split_v1/weights/onnx/patchcore_bottle.onnx \
  --input-bin ../training/outputs/week06_io/good015/input.f32 \
  --output-dir outputs/day03_good015
```

## 异常样例

```bash
./build/inspectflow_infer \
  --model ../training/outputs/onnx/patchcore_bottle_r001_split_v1/weights/onnx/patchcore_bottle.onnx \
  --input-bin ../training/outputs/week06_io/broken_small015/input.f32 \
  --output-dir outputs/day03_broken_small015
```

输出目录必须不存在。重复运行请选择新目录；程序不会覆盖已有结果。仅传 `--model` 仍然只打印模型元信息。

## 输入与输出

输入是无文件头的小端IEEE 754 float32，RGB、NCHW、shape `[1,3,256,256]`，值在0～1范围内。总共196608个数、786432字节。不能直接传PNG或NPY文件。本程序不执行读图、缩放或重复标准化；这些输入已经由Python准备好。

每次推理输出：

- `prediction.json`：模型和输入路径、ORT版本、分数、标签、热力图范围、异常像素数及首次Run耗时。
- `anomaly_map.f32`：65536个小端float32，按行排列。完整输出shape是 `[1,1,256,256]`。
- `pred_mask.u8`：65536个uint8，按行排列，每个值为0或1。

分数/标签写在JSON里，热力图和掩码保存原始数值，不生成彩色PNG。`session_run_ms`只计首次 `Session::Run()`，不含模型加载、读写文件，不是稳定性能基准。

## 建议阅读顺序

1. `src/main.cpp`：沿用昨天的模型检查入口，新增参数解析和推理调用。
2. `src/binary_inference.hpp` 的 `read_input()`：检查字节数，读入 `std::vector<float>`，验证数值。
3. `run_binary_inference()`：检查模型协议、创建Tensor、按输出名称调用Run。
4. `require_tensor()`：先确认dtype和shape，再用对应C++类型读取。
5. `write_bytes()`及报告构造：将输出保存为可与Python比较的文件。

`CreateTensor<float>()`借用vector内存。在Run完成之前不能销毁或改变vector容量。`GetTensorData<T>()`返回的指针也只是借用输出Ort::Value中的内存，在写文件完成之前输出对象必须存活。布尔输出用 `GetTensorData<bool>()`，保存时显式转成uint8，不能用float指针解释。

头文件中的函数使用inline，方便当前教学工程直接包含。需要扩展工程时再拆成声明和cpp实现。

## 已验证（2026-09-23）

使用当前ONNX及Python参考数据，在CPU/ORT 1.30.0上运行6个样例：good015、broken_small015、broken_small017、broken_large015、broken_large017、contamination015。1张正常、5张异常，分数与参考值在rtol=1e-3、atol=1e-4内一致，热力图最大绝对误差均为0，掩码逐字节一致。这是相同输入Tensor的跨语言验证，不是独立准确率评测。

同时检查了：仅查看模型、帮助命令、缺失配对参数、未知参数、错误文件长度、NaN输入、已有输出目录。错误返回非零退出码。写文件中途失败可能留下不完整目录，应查看错误并换新目录重试。
