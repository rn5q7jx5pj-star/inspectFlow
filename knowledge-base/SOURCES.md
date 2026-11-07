# 来源与核对记录

核对日期：2026-11-01。以下路径均相对于本目录的父目录 InspectFlow；这是本地资料索引，不是公开标准清单。本版不引用外部网页，也不包含未经验证的行业操作要求。

| 来源 ID | 路径 | 核对内容 |
| --- | --- | --- |
| SRC-API | inference-cpp-split/src/server_main.cpp | 三个 HTTP 路由、上传限制、错误处理、响应字段、请求编号、模型版本处理 |
| SRC-ENGINE | inference-cpp-split/src/inference_engine.cpp | 固定张量协议、直接读取模型标签、Session::Run 计时、内部异常图与掩码 |
| SRC-PREPROCESS | inference-cpp-split/src/preprocess.cpp | BGR 转 RGB、线性缩放、除以 255、CHW 排列 |
| SRC-API-DOC | inference-cpp-split/README.md | 标签语义、分数与耗时解释、上传与错误响应约定 |
| SRC-MODEL-CONFIG | training/configs/patchcore_bottle_r001_split.yaml | Patchcore、MVTecAD bottle 类别、启用后处理的训练配置 |
| SRC-PLUGIN | plugins/inspectflow/tools/inspectflow.py | Dify File 读取、multipart 调用、请求超时、异常处理、响应必要字段检查 |
| DEMO-DESIGN | 本次为学习目的编写的演示流程 | 人工复核建议、记录格式、保守报告策略；无企业或行业效力 |

## 事实与推断的界限

- SRC-MODEL-CONFIG 说明该配置面向 bottle，不证明当前运行服务一定加载了与它对应的产物。模型由启动参数决定，部署时仍需核对实际文件。
- 标签 true/false 的业务含义依据 SRC-API-DOC；SRC-ENGINE 证明服务直接读取 ONNX 的布尔输出，不在 HTTP 层重新设置阈值。
- 预处理描述只涉及 C++ 服务在模型外执行的处理，不推断 ONNX 图内部是否包含额外变换。
- 热力图的限制针对当前 HTTP 检测响应：模型及推理引擎内部已有异常图和掩码，不等于 Dify 已收到这些数据。
- 未重新运行训练、评测、HTTP 服务或 Dify。本文不是测试通过报告；未记录任何虚构准确率、吞吐量、成本或检索指标。

## 条目来源映射

| 条目 | 主要来源 |
| --- | --- |
| RESULT-01、02 | SRC-API、SRC-ENGINE、SRC-API-DOC |
| RESULT-03、04 | SRC-API、SRC-ENGINE、SRC-API-DOC |
| INPUT-01 | SRC-API |
| INPUT-02 | SRC-PREPROCESS、SRC-ENGINE |
| INPUT-03 | SRC-MODEL-CONFIG、DEMO-DESIGN |
| SOP-REVIEW-01～04 | DEMO-DESIGN；所用接口事实另引 SRC-API / SRC-API-DOC / SRC-PLUGIN |
| LIMIT-01 | SRC-MODEL-CONFIG、SRC-API、DEMO-DESIGN |
| LIMIT-02 | SRC-API、SRC-ENGINE |
| LIMIT-03 | SRC-API、SRC-ENGINE |
| LIMIT-04 | DEMO-DESIGN、SRC-API-DOC |
