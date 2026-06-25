"""The GRPO builder's --multimodal flag injects the modality sections into the
prompt and carries a valid 5-class label per example."""
import json
import subprocess
import sys


def test_grpo_multimodal_sections(tmp_path):
    out = tmp_path / "data"
    subprocess.run(
        [sys.executable, "-m", "compare_lab.grpo.build_dataset",
         "--out", str(out), "--multimodal", "--n", "10"],
        check=True,
    )
    lines = (out / "train.jsonl").read_text().splitlines()
    lines += (out / "val.jsonl").read_text().splitlines()
    rows = [json.loads(l) for l in lines]
    assert rows, "no records produced"
    assert any("=== NEWS" in r["prompt"][0]["content"] for r in rows)
    assert all(r["label"] in {"STRONG_SELL", "SELL", "HOLD", "BUY", "STRONG_BUY"}
               for r in rows)
