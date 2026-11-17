"""OCR for scanned PDF pages through an OpenAI-compatible vision model."""

from __future__ import annotations

import base64
import logging

import httpx

from fastrag.config import Settings
from fastrag.providers.http import bearer, post_json

logger = logging.getLogger(__name__)

_PROMPT = (
    "Chép lại chính xác toàn bộ văn bản tiếng Việt trong ảnh, giữ nguyên dấu, xuống dòng và bảng "
    "(dùng | giữa các cột). Không thêm lời giải thích."
)


class VlmOcr:
    def __init__(self, s: Settings):
        assert s.ocr_url and s.ocr_model
        base = s.ocr_url.rstrip("/")
        self.url = base if base.endswith("/chat/completions") else f"{base}/v1/chat/completions"
        self.model = s.ocr_model
        self.headers = bearer(s.ocr_api_key.get_secret_value() if s.ocr_api_key else None)

    async def page_text(self, images: list[bytes]) -> str:
        content: list[dict[str, object]] = [{"type": "text", "text": _PROMPT}]
        for img in images:
            mime = "image/png" if img[:4] == b"\x89PNG" else "image/jpeg"
            b64 = base64.b64encode(img).decode()
            content.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}})
        async with httpx.AsyncClient(timeout=120) as client:
            data = await post_json(
                client,
                self.url,
                {"model": self.model, "messages": [{"role": "user", "content": content}], "temperature": 0},
                headers=self.headers,
            )
        return str(data["choices"][0]["message"]["content"])


def build_ocr(s: Settings) -> VlmOcr | None:
    if s.ocr_provider == "vlm" and s.ocr_url and s.ocr_model:
        return VlmOcr(s)
    return None
