"""The SFT builder's --multimodal flag injects the news/fundamentals/sentiment/
macro sections into the prompt (the templated thesis stays price-grounded)."""
import json
import subprocess
import sys


def test_multimodal_flag_injects_sections(tmp_path):
    out = tmp_path / "data"
    subprocess.run(
        [sys.executable, "-m", "compare_lab.sft.build_dataset",
         "--out", str(out), "--multimodal", "--every", "60", "--limit", "5"],
        check=True,
    )
    lines = (out / "train.jsonl").read_text().splitlines()
    lines += (out / "val.jsonl").read_text().splitlines()
    rows = [json.loads(l) for l in lines]
    assert rows, "no records produced"
    assert any("=== NEWS" in r["messages"][0]["content"] for r in rows)
