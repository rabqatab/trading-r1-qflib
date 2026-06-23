"""VLLMClient - OpenAI-compatible chat client with a disk cache.

`transport` is injectable so tests run without a server. The default transport
calls a vLLM OpenAI-compatible endpoint (DGX Spark, single-node,
--enforce-eager BF16; served via sparkq). Cache key = caller-supplied snapshot
hash, so identical inputs always return identical outputs (reproducibility).
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Callable

from compare_lab.config import CACHE_DIR

# Endpoint + served model id, overridable via env so the same code targets
# whatever vLLM server is up (e.g. a sparkq job on a non-default port).
DEFAULT_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
DEFAULT_MODEL = os.environ.get("VLLM_MODEL", "Qwen/Qwen3-4B")
# Cache is keyed by snapshot hash only (model-agnostic), so a different model
# (e.g. an SFT LoRA) MUST use a different cache dir or it reuses the base
# replies. Override per run with VLLM_CACHE_DIR.
DEFAULT_CACHE_DIR = Path(os.environ.get("VLLM_CACHE_DIR", str(CACHE_DIR)))
# Distilled SFT-v2 writes long §8 theses; 2048 truncates ~16% before the final
# [[[CLASS]]] line. Raise per run (e.g. 4096) so the decision tag survives.
DEFAULT_MAX_TOKENS = int(os.environ.get("VLLM_MAX_TOKENS", "2048"))


def _default_transport(base_url: str, model: str) -> Callable[[str], str]:
    from openai import OpenAI

    client = OpenAI(base_url=base_url, api_key="EMPTY")

    def _call(prompt: str) -> str:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=DEFAULT_MAX_TOKENS,
        )
        return resp.choices[0].message.content or ""

    return _call


class VLLMClient:
    def __init__(
        self,
        transport: Callable[[str], str] | None = None,
        base_url: str = DEFAULT_BASE_URL,
        model: str = DEFAULT_MODEL,
        cache_dir: Path = DEFAULT_CACHE_DIR,
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
