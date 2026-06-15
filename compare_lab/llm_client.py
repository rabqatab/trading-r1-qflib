"""VLLMClient - OpenAI-compatible chat client with a disk cache.

`transport` is injectable so tests run without a server. The default transport
calls a vLLM OpenAI-compatible endpoint (DGX Spark, single-node,
--enforce-eager BF16; served via sparkq). Cache key = caller-supplied snapshot
hash, so identical inputs always return identical outputs (reproducibility).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

from compare_lab.config import CACHE_DIR


def _default_transport(base_url: str, model: str) -> Callable[[str], str]:
    from openai import OpenAI

    client = OpenAI(base_url=base_url, api_key="EMPTY")

    def _call(prompt: str) -> str:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=2048,
        )
        return resp.choices[0].message.content or ""

    return _call


class VLLMClient:
    def __init__(
        self,
        transport: Callable[[str], str] | None = None,
        base_url: str = "http://localhost:8000/v1",
        model: str = "Qwen/Qwen3-4B",
        cache_dir: Path = CACHE_DIR,
    ):
        self._transport = transport or _default_transport(base_url, model)
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        return self._cache_dir / f"{key}.json"

    def complete(self, prompt: str, key: str) -> str:
        p = self._path(key)
        if p.exists():
            return json.loads(p.read_text())["response"]
        response = self._transport(prompt)
        p.write_text(json.dumps({"prompt": prompt, "response": response}))
        return response
