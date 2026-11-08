from collections.abc import Generator
from typing import Any

from dify_plugin import Tool
from dify_plugin.entities.tool import ToolInvokeMessage

import httpx
from dify_plugin.file.file import File
import requests

import base64
import binascii

class InspectflowTool(Tool):
    def _invoke(
        self,
        tool_parameters: dict[str, Any],
    ) -> Generator[ToolInvokeMessage]:
        image = tool_parameters.get("image")

        if image is None:
            raise ValueError("请上传一张待检测图片")

        if not isinstance(image, File):
            raise ValueError("image 参数必须是 Dify 文件，请检查工作流的变量绑定")

        try:
            image_bytes = image.blob #读取为二进制文件
        except (httpx.HTTPError, ValueError) as exc:
            raise ValueError("无法读取上传文件，请检查文件访问地址和网络连接") from exc

        if not image_bytes:
            raise ValueError("上传的文件内容为空")

        # self.runtime.credentials：Dify SDK 提供的配置字典，里面保存了你在插件配置页面填写的内容
        base_url = self.runtime.credentials["base_url"].rstrip("/")
        url = f"{base_url}/v1/detections"

        files = {
            "image": (
                image.filename or "upload.bin",
                image_bytes,
                image.mime_type or "application/octet-stream",
            )
        }

        try:
            resp = requests.post(
                url,
                files=files,
                data={"include_visualization": "true"},
                timeout=10,
            )

        except requests.exceptions.Timeout as exc:
            raise RuntimeError("推理服务请求超时") from exc

        except requests.exceptions.ConnectionError as exc:
            raise RuntimeError(
                "无法连接推理服务，请确认 C++ 服务已启动且地址正确"
            ) from exc

        except requests.exceptions.RequestException as exc:
            raise RuntimeError("请求推理服务时发生其他 HTTP 请求异常") from exc

        if resp.status_code != 200:
            # 先准备兜底信息
            request_id = resp.headers.get("X-Request-ID", "未知")
            error_code = "UNKNOWN_ERROR"
            error_message = "服务未提供可用的错误说明"

            # 尝试解析错误响应，但不能假设它一定是 JSON
            try:
                body = resp.json()
            except requests.exceptions.JSONDecodeError:
                body = {}

            if isinstance(body, dict):
                request_id = body.get("request_id") or request_id

                error = body.get("error")

                if isinstance(error, dict):
                    error_code = error.get("code") or error_code
                    error_message = error.get("message") or error_message

            raise RuntimeError(
                f"检测请求失败，HTTP 状态码：{resp.status_code}，"
                f"请求 ID：{request_id}，"
                f"错误代码：{error_code}，"
                f"错误原因：{error_message}"
            )
        try:
            result = resp.json()
        except requests.exceptions.JSONDecodeError as exc:
            raise RuntimeError("推理服务返回的内容不是有效 JSON") from exc

        if not isinstance(result, dict):
            raise RuntimeError("推理服务响应格式错误：预期为 JSON 对象")

        required_fields = (
            "request_id",
            "pred_score",
            "pred_label",
            "inference_ms",
        )

        for field in required_fields:
            if field not in result:
                raise RuntimeError(f"推理服务响应缺少字段：{field}")

        # 将图片数据从检测摘要中拆出，避免 Base64 出现在 JSON 输出里。
        visualization = result.pop("visualization", None)
        png_bytes = None
        visualization_warning = None

        if visualization is None:
            visualization_warning = "检测成功，但服务没有返回可视化图片。"
        elif not isinstance(visualization, dict):
            visualization_warning = "检测成功，但可视化数据格式不正确。"
        elif (
            visualization.get("mime_type") != "image/png"
            or visualization.get("encoding") != "base64"
        ):
            visualization_warning = "检测成功，但图片类型或编码方式不符合约定。"
        else:
            overlay_base64 = visualization.get("overlay")
            if not isinstance(overlay_base64, str) or not overlay_base64:
                visualization_warning = "检测成功，但图片数据为空或格式不正确。"
            else:
                try:
                    png_bytes = base64.b64decode(overlay_base64, validate=True)
                except (binascii.Error, ValueError):
                    visualization_warning = "检测成功，但图片数据解码失败。"

        if png_bytes == b"":
            png_bytes = None
            visualization_warning = "检测成功，但解码后的图片为空。"

        if visualization_warning is not None:
            result["visualization_warning"] = visualization_warning

        # 输出一：不包含 Base64 的检测摘要。
        yield self.create_json_message(result)

        # 输出二：如果图片可用，再输出 PNG 文件。
        if png_bytes is not None:
            yield self.create_blob_message(
                blob=png_bytes,
                meta={"mime_type": "image/png"},
            )

        # 图片不可用时，再返回一条易读提示。
        if visualization_warning is not None:
            yield self.create_text_message(visualization_warning)
