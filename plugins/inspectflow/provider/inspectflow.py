from typing import Any
from urllib.parse import urlsplit

from dify_plugin import ToolProvider
from dify_plugin.errors.tool import ToolProviderCredentialValidationError


class InspectflowProvider(ToolProvider):
    def _validate_credentials(
        self,
        credentials: dict[str, Any],
    ) -> None:
        base_url = str(credentials.get("base_url", "")).strip()
        parsed = urlsplit(base_url)

        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ToolProviderCredentialValidationError(
                "请填写有效地址，例如 http://127.0.0.1:8080"
            )