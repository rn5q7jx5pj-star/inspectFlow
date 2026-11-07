from collections.abc import Generator
from typing import Any

import requests
from dify_plugin import Tool
from dify_plugin.entities.tool import ToolInvokeMessage


class GetDetectionResultTool(Tool):
    def _invoke(
        self,
        tool_parameters: dict[str, Any],
    ) -> Generator[ToolInvokeMessage]:

        # 1. 从 Dify 传入的参数中读取编号
        detection_id = str(
            tool_parameters.get("detection_id", "")
        ).strip()

        # 2. 这里按“正整数编号”的接口约定检查输入
        if not detection_id.isascii() or not detection_id.isdecimal():
            raise ValueError("检测编号必须是正整数")

        if int(detection_id) <= 0:
            raise ValueError("检测编号必须大于 0")

        # 3. 复用现有插件配置的 C++ 服务地址
        base_url = self.runtime.credentials["base_url"].rstrip("/")
        url = f"{base_url}/v1/detections/{detection_id}"

        # 4. 向 C++ 服务发送查询请求
        try:
            response = requests.get(url, timeout=10)
        except requests.exceptions.Timeout as exc:
            raise RuntimeError("查询请求超时") from exc
        except requests.exceptions.ConnectionError as exc:
            raise RuntimeError(
                "无法连接推理服务，请检查服务是否启动以及地址是否正确"
            ) from exc
        except requests.exceptions.RequestException as exc:
            raise RuntimeError("查询检测结果时发生网络错误") from exc

        # 5. 根据状态码处理常见情况
        if response.status_code == 404:
            raise ValueError(f"未找到检测编号为 {detection_id} 的记录")

        if response.status_code != 200:
            raise RuntimeError(
                f"查询失败，服务返回 HTTP {response.status_code}"
            )

        # 6. 解析并返回服务的 JSON
        try:
            result = response.json()
        except requests.exceptions.JSONDecodeError as exc:
            raise RuntimeError("服务返回的不是有效 JSON") from exc

        if not isinstance(result, dict):
            raise RuntimeError("服务返回格式错误：预期为 JSON 对象")

        yield self.create_json_message(result)