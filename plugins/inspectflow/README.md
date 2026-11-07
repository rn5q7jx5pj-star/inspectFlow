## inspectflow

**Author:** godfan
**Version:** 0.0.1
**Type:** tool

### Description

Industrial image anomaly detection tools

> Replace the placeholder sections below with real content before publishing to the Dify Marketplace, then remove this note.
> Keep this file in English only — a README.md containing Chinese characters will be rejected during review.
> For translated documentation, add localized files such as `README_zh_Hans.md` instead. See the [multilingual README guide](https://docs.dify.ai/en/develop-plugin/features-and-specs/plugin-types/multilingual-readme).

### Setup

Describe how to configure the plugin: required credentials or API keys, where to obtain them, and any prerequisites.

### Usage

Describe how to use the plugin in Dify, with examples if possible.



+ 启动和停止

```powershell
cd /Users/yangfan/CS/dify-local/docker #启动docker以及dify服务
docker compose up -d

#启动插件
cd /Users/yangfan/CS/InspectFlow/plugins/inspectflow
source .venv/bin/activate
python -m main

#访问http://127.0.0.1或者直接http://localhost 即可进入平台使用插件搭建工作流

#启动服务
cd /Users/yangfan/CS/InspectFlow/inference-cpp-split
./build-http/inspectflow_server \
../training/outputs/onnx/patchcore_bottle_r001_split_v1/weights/onnx/patchcore_bottle.onnx \
  8080

#停止
#Ctrl C停止插件
deactivate #退出虚拟环境
cd /Users/yangfan/CS/dify-local/docker
docker compose stop
docker compose ps -a

#有时需要更新调试密钥
#打开/Users/yangfan/CS/InspectFlow/plugins/inspectflow/.env
REMOTE_INSTALL_KEY=替换
```

### 推理服务地址如何配置

`base_url` 是插件访问 C++ 推理服务的基础地址，在 Dify 的插件配置页面填写，不是在插件的 `.env` 中填写。

#### 当前本地开发配置

当前 Python 插件与 C++ 服务都直接运行在 Mac 上，Dify 平台运行在 Docker 中。C++ 服务使用端口 `8080` 时，填写：

```text
http://127.0.0.1:8080
```

1. 按上面的启动命令运行 C++ 服务，确认模型加载和服务启动成功，并保持该终端运行。
2. 打开本地 Dify，在插件或集成页面找到 `inspectflow`，进入其授权/配置入口。
3. 在“推理服务地址 / Inference Service URL”字段中填写上述地址并保存；若工作流需要选择凭据，选择这份配置。
4. 重新运行工作流，上传图片进行检测。

地址必须包含 `http://` 或 `https://`，端口必须与 C++ 服务启动参数一致。这里只填写基础地址，不要附加 `/v1/detections`。插件会自动拼接检测接口路径：

```python
base_url = self.runtime.credentials["base_url"].rstrip("/")
url = f"{base_url}/v1/detections"
```

例如，配置 `http://127.0.0.1:8080` 后，实际请求地址为 `http://127.0.0.1:8080/v1/detections`。如果服务改用其他端口，也要同步修改 Dify 中的配置，无需修改 Python 代码。

#### 检查地址是否可用

在 Mac 的另一个终端执行：

```bash
curl --max-time 5 http://127.0.0.1:8080/health
```

正常应返回：

```json
{"status":"ok","model_loaded":true}
```

若连接失败，先检查 C++ 服务是否仍在运行，以及启动端口是否与配置一致。当前提供者配置只校验 URL 格式，保存成功不代表服务已经可连接。

#### 不要混淆这几个地址

| 地址或配置 | 用途 |
| --- | --- |
| `http://localhost` | 浏览器访问 Dify 平台 |
| `REMOTE_INSTALL_URL=127.0.0.1:5003` | Python 插件连接 Dify 调试服务，配置在插件 `.env` 中 |
| `base_url=http://127.0.0.1:8080` | 插件调用 C++ 推理服务，在 Dify 插件配置页面填写 |
| `FILES_URL=http://127.0.0.1` | Dify 生成文件下载地址，配置在 Dify 部署目录的 `.env` 中 |

以上地址适用于当前本机调试方式。如果以后把 Python 插件部署到容器或其他机器，`127.0.0.1` 就会指向插件所在的环境，而不是这台 Mac；届时需要使用插件可访问的服务地址，并相应检查 C++ 服务的监听范围。当前 C++ 服务仅监听本机 `127.0.0.1`，不要直接将无鉴权的服务暴露到公网。

