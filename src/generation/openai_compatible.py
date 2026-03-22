"""OpenAI-compatible chat completion client using the Python standard library."""

from __future__ import annotations

import json
from dataclasses import dataclass
from urllib import error, request


class LLMClientError(RuntimeError):
    """Raised when a hosted LLM call fails."""


@dataclass
class OpenAICompatibleClient:
    api_key: str
    base_url: str
    model: str
    timeout_seconds: int = 30
    temperature: float = 0.0
    max_tokens: int = 120

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        endpoint = f"{self.base_url.rstrip('/')}/chat/completions"
        payload = {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        body = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        req = request.Request(endpoint, data=body, headers=headers, method="POST")

        try:
            with request.urlopen(req, timeout=self.timeout_seconds) as response:
                raw = response.read().decode("utf-8")
        except error.HTTPError as exc:  # pragma: no cover - network dependent
            details = exc.read().decode("utf-8", errors="replace")
            raise LLMClientError(f"LLM request failed with HTTP {exc.code}: {details}") from exc
        except error.URLError as exc:  # pragma: no cover - network dependent
            raise LLMClientError(f"LLM request failed: {exc.reason}") from exc

        data = json.loads(raw)
        choices = data.get("choices", [])
        if not choices:
            raise LLMClientError("LLM response did not contain any choices.")
        message = choices[0].get("message", {})
        content = message.get("content", "")
        if isinstance(content, list):
            text_parts = [part.get("text", "") for part in content if isinstance(part, dict)]
            content = "".join(text_parts)
        if not isinstance(content, str) or not content.strip():
            raise LLMClientError("LLM response did not contain text content.")
        return content.strip()
