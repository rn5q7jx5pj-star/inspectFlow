# InspectFlow C++ 推理与 HTTP 服务

使用 ONNX Runtime 运行固定输入为 `float32 [1,3,256,256]` 的异常检测模型。
命令行程序支持四种用法：只查看模型输入输出、单张图片推理、图片清单批量推理、
输入已预处理的 `.f32` 张量推理。HTTP 服务复用同一套预处理与推理模块，
可通过上传图片获得异常分数和正常/异常判断。

## 构建

需要 CMake 3.20+、C++17、OpenCV（`core`、`imgproc`、`imgcodecs`）、
`third_party/onnxruntime-osx-arm64-1.30.0` 中的 ONNX Runtime SDK，以及
`third_party/cpp-httplib/httplib.h`。后两个依赖目录不随 Git 仓库提交。
在本机 Homebrew OpenCV 5 环境下：

```sh
cmake -S . -B build-local -DCMAKE_BUILD_TYPE=Release \
  -DOpenCV_DIR=/opt/homebrew/opt/opencv/lib/cmake/opencv5
cmake --build build-local -j 4
ctest --test-dir build-local --output-on-failure
```

如果 `build/` 是从其他项目复制来的，里面的 `CMakeCache.txt` 可能记录了旧源码路径；
此时换一个新的构建目录，不要复用旧缓存。如果 ONNX Runtime SDK 放在别处，
增加 `-DONNXRUNTIME_ROOT=/实际/SDK/目录`；该目录下须有
`include/onnxruntime_cxx_api.h` 和 `lib/libonnxruntime.dylib`。

## 命令行运行

```sh
./build-local/inspectflow_infer --model /path/to/model.onnx
./build-local/inspectflow_infer --model /path/to/model.onnx --image /path/to/image.png --output-dir /path/to/new-output
./build-local/inspectflow_infer --model /path/to/model.onnx --image-list /path/to/images.txt --output-dir /path/to/new-batch-output
./build-local/inspectflow_infer --model /path/to/model.onnx --input-bin /path/to/input.f32 --output-dir /path/to/new-output
```

`--image`、`--image-list` 和 `--input-bin` 三选一。推理时必须提供一个**尚不存在**的输出目录；
程序会创建它。图片会转换为 RGB、缩放到 256×256，再除以 255 并按 NCHW 排列。
二进制输入必须是按相同顺序保存的 786432 字节（`3×256×256` 个 float32）。

一次推理会保存 `input.f32`、`input_metadata.json`、`heatmap.png`、
`mask.png` 和 `prediction.json`。热力图按当前图片的最小值和最大值归一化，
仅用于可视化，不适合直接比较不同图片的颜色深浅。

批量清单每行一个图片的**绝对路径**。程序只加载一次模型，逐张运行固定 Batch=1 的模型。
每张图片放在类别名与文件名拼接的目录下，例如 `good/015.png` 对应 `good015/`。
批量根目录另外生成 `predictions.csv` 和 `run_info.json`。坏图片会在 CSV 中记录错误，
程序继续处理下一张；如果至少一张失败，最终退出码为 1。`run_info.json` 记录模型文件与
ONNX 元数据版本、预处理协议、成功/失败数、ORT/OpenCV 版本和计时范围。若模型没有可用的
ONNX 元数据版本，该字段为 `null`。

## 启动 HTTP 服务

构建后还会生成 `build-local/inspectflow_server`。它接收两个位置参数：
ONNX 模型路径和监听端口。在项目根目录运行：

```sh
./build-local/inspectflow_server \
  /Users/yangfan/CS/InspectFlow/training/outputs/onnx/patchcore_bottle_r001_split_v1/weights/onnx/patchcore_bottle.onnx \
  8080
```

这是本机现有模型的示例路径，换设备时要替换为实际路径。服务启动时加载模型一次，
然后监听 `127.0.0.1:8080`。运行期间启动终端会一直占用；下面的请求请在另一个终端执行。
模型不存在、无法加载或端口被占用时，服务会打印原因并退出。在启动终端按 `Ctrl+C` 可停止服务。

### 1. 健康检查

```sh
curl -i http://127.0.0.1:8080/health
```

成功时状态为 HTTP 200，响应体为：

```json
{"status":"ok","model_loaded":true}
```

### 2. 查看当前模型

```sh
curl -i http://127.0.0.1:8080/v1/models/current
```

响应包含 `model_name`、`model_version`、`input_name`、`input_shape`、
`input_dtype`、`provider` 和 `onnxruntime_version`。`model_version` 读取 ONNX
模型元数据中的版本：有效正整数原样返回；非正值或 ORT 的
`INT64_MAX` 占位值按未设置处理并返回 JSON `null`，不再使用固定占位值。
该字段不是从模型文件名或训练目录名推断出的训练版本。

### 3. 上传图片进行检测

请求方法为 `POST`，路径为 `/v1/detections`。使用 `multipart/form-data`
上传图片，不接受直接发送 JSON 或 Base64 字符串代替文件上传。

| 请求项 | 约定 |
|---|---|
| 文件字段名 | `image`，必填，内容为图片文件的原始字节 |
| 图片格式 | 必须能被当前 OpenCV 环境解码，例如 PNG、JPEG；仅修改文件扩展名不能使无效文件变为图片 |
| 单个图片文件大小 | 大于 0 字节且不超过 5 MiB（5,242,880 字节） |
| 整个 HTTP 请求体大小 | 不超过 6 MiB（6,291,456 字节），包含文件和 multipart 边界等开销 |

使用 `curl -F` 时由 curl 自动生成 `Content-Type` 和 multipart boundary，
无需手动设置该请求头：

```sh
curl -i -F 'image=@/Users/yangfan/CS/InspectFlow/training/datasets/MVTecAD/bottle/test/broken_small/015.png' \
  http://127.0.0.1:8080/v1/detections
```

成功时状态为 HTTP 200，响应体的结构如下；分数和耗时由实际推理产生：

```json
{
  "request_id": "req-1",
  "pred_score": 0.66,
  "pred_label": true,
  "inference_ms": 35.2
}
```

成功响应的 `Content-Type` 为 `application/json`，字段约定如下：

| 字段 | JSON 类型 | 含义 |
|---|---|---|
| `request_id` | string | 服务生成的请求标识，例如 `req-1`；进程内递增，重启后重新计数，不是跨进程全局唯一 ID |
| `pred_score` | number | 模型输出的图像级异常分数；不是经过概率校准的异常概率，不应直接换算为百分比，也不保证始终处于 `[0,1]` |
| `pred_label` | boolean | `false` 表示模型判断为正常，`true` 表示模型判断为异常 |
| `inference_ms` | number | ONNX Runtime `Session::Run` 的耗时，单位为毫秒；不含上传、图片解码、预处理和结果序列化 |

注意：HTTP 接口实际返回的是 JSON 布尔值 `true` / `false`，不是数值 `1` / `0`，
也不是字符串 `"true"` / `"false"`。若客户端为了展示将标签转换为数值，
可约定 `0=正常`、`1=异常`，但这属于客户端转换，不是当前接口的原始响应格式。
服务直接读取 ONNX 模型输出的 `pred_label`，没有在 HTTP 层额外设置分数阈值；
不要自行用示例分数或固定的 `0.5` 阈值替代模型判定。

响应头 `X-Request-ID` 与响应体中的 `request_id` 对应，可用于关联记录；
推理失败时，服务器终端会记录带有该 ID 的错误。当前 HTTP 接口尚未返回热力图。

## HTTP 错误响应

| 情况 | 状态码 | 错误码 |
|---|---:|---|
| 缺少 `image` 文件字段 | 400 | `MISSING_IMAGE` |
| 图片文件是空的 | 400 | `IMAGE_IS_EMPTY` |
| 图片无法解码 | 400 | `INVALID_IMAGE` |
| 单个图片文件超过 5 MiB | 413 | `IMAGE_TOO_LARGE` |
| 推理或服务内部失败 | 500 | `INFERENCE_FAILED` 或 `INTERNAL_ERROR` |

通常的错误响应格式如下：

```json
{
  "request_id": "req-2",
  "error": {
    "code": "MISSING_IMAGE",
    "message": "请求中缺少image文件"
  }
}
```

上述业务错误响应的 `Content-Type` 为 `application/json`，字段含义如下：

| 字段 | JSON 类型 | 含义 |
|---|---|---|
| `request_id` | string | 本次请求的标识，与响应头 `X-Request-ID` 对应 |
| `error` | object | 错误详情；业务错误响应不包含成功时的检测结果字段 |
| `error.code` | string | 供程序判断错误类别的稳定错误码，取值见上表 |
| `error.message` | string | 面向使用者的简短错误说明；客户端应按状态码和错误码判断，不依赖中文文本匹配 |

HTTP 请求体总大小限制为 6 MiB，给图片之外的 multipart 数据留出空间。
如果请求在进入检测处理函数之前就超过该限制，HTTP 库可能直接拒绝，响应未必采用上面的格式。
OpenCV 解码抛出异常与解码后得到空图片这两种情况，均返回 HTTP 400、
`INVALID_IMAGE`，并使用上面的业务错误格式。

可以用以下请求检查常见错误，再访问 `/health` 确认服务仍在运行：

```sh
curl -i -X POST http://127.0.0.1:8080/v1/detections
curl -i -F 'image=@README.md' http://127.0.0.1:8080/v1/detections
curl -i http://127.0.0.1:8080/health
```

## 对照命令行结果

HTTP 服务与 `inspectflow_infer` 共用 `InferenceEngine` 和预处理逻辑。
对同一张图片分别调用 HTTP 和命令行，比较返回的 `pred_score`、`pred_label`：

```sh
./build-local/inspectflow_infer \
  --model /Users/yangfan/CS/InspectFlow/training/outputs/onnx/patchcore_bottle_r001_split_v1/weights/onnx/patchcore_bottle.onnx \
  --image /Users/yangfan/CS/InspectFlow/training/datasets/MVTecAD/bottle/test/broken_small/015.png \
  --output-dir outputs/cli-broken-small-015
```

输出目录必须事先不存在，查看其中的 `prediction.json`。
目前服务只绑定本机地址，没有认证、历史检测记录或热力图下载功能；
`request_id` 是进程内递增编号，服务重启后重新计数。

## 测试

```sh
cmake --build build-local -j 4
ctest --test-dir build-local --output-on-failure
```

CTest 当前主要覆盖颜色通道与 HWC→CHW 排列、非法二进制输入和无法读取的图片。
HTTP 接口可按本 README 的 `curl` 命令手工验证；仓库目前没有自动化 HTTP 接口测试。
