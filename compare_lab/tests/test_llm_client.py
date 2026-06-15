import compare_lab  # noqa: F401
from compare_lab.llm_client import VLLMClient


def test_client_caches_by_key(tmp_path):
    calls = {"n": 0}

    def fake_transport(prompt: str) -> str:
        calls["n"] += 1
        return f"reply::{prompt}"

    c = VLLMClient(transport=fake_transport, cache_dir=tmp_path)
    r1 = c.complete("hello", key="k1")
    r2 = c.complete("hello", key="k1")   # served from cache
    assert r1 == r2 == "reply::hello"
    assert calls["n"] == 1               # transport called once only


def test_client_distinct_keys(tmp_path):
    def fake_transport(prompt: str) -> str:
        return prompt.upper()

    c = VLLMClient(transport=fake_transport, cache_dir=tmp_path)
    assert c.complete("a", key="ka") == "A"
    assert c.complete("b", key="kb") == "B"
