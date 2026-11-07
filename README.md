# InspectFlow

工业图像异常检测学习项目：Python 训练与 ONNX 导出、C++ 推理及 HTTP 服务、Dify 插件与知识库。

## 目录

- `training/`：训练、模型导出、评测及实验记录。
- `inference-cpp-split/`：当前 C++ 主线，构建和运行方式见该目录 README。
- `plugins/inspectflow/`：Dify 检测插件。
- `knowledge-base/`：检测报告相关知识资料。
- `inference-cpp/`、`inference-cpp-all/`：历史学习实现。
- `opencv-practice/`：OpenCV 练习。

## 当前状态

首次导入为当前本地文件快照，未导入原子仓库的提交历史。SQLite 存储代码尚未接入 CMake，属于开发中功能。本次整理未执行构建或功能测试。

数据集、模型权重、第三方 SDK、虚拟环境和运行输出不随仓库提供。请参考各模块 README 配置依赖与模型路径；部分示例含原开发机器的绝对路径，需要按本机修改。
