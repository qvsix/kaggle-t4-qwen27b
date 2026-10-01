#!/usr/bin/env python3
"""Regenerate notebook/swift-gpu-serve.ipynb from kernel/serve_qwen38_boost.py.

    python3 make_notebook.py [--ctx-size N] [--mtp-n-max N]

The notebook embeds the kernel script byte-identical: edit the script, then
run this generator. Never hand-edit the generated notebook's script cell.
"""

import argparse
import json
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
SCRIPT = BASE_DIR / "kernel" / "serve_qwen38_boost.py"
NOTEBOOK = BASE_DIR / "notebook" / "swift-gpu-serve.ipynb"

INTRO_MD = """# Qwen3.8-27B on Kaggle GPU T4 x2 (Q5 KV + MTP + YaRN)

Serves stock **Qwen3.8-27B** (`IQ4_XS` GGUF, ~15.4 GiB) on Kaggle's free
**GPU T4 x2** (2x16 GB) with `llama-server` (pinned `b11095`), exposing an **OpenAI-compatible**
endpoint through a Cloudflare tunnel. Point Codex CLI, opencode, or plain `curl` at it.

| Mode | KV cache | Context | MTP | Binary dataset |
|---|---|---|---|---|
| Throughput (default) | `q5_0` + rotation | 350k via YaRN | 3 draft tokens | `leejiwon00/qwen-3-8-q5-binary` (FA_ALL_QUANTS) |
| Max precision | `q8_0` + rotation | 262144 native | 0 | `leejiwon00/swift-llama-bin` (stock) |

Q5 keeps dot-product fidelity via Hadamard rotation while cutting KV from ~34 to ~22.5 KiB/tok;
MTP uses the model's embedded Q8_0 draft head for speculative decode speedup.
If a 350k load OOMs, the script steps ctx down 20% per retry (floor 131072, then MTP
stepdown) until one fits, so a single session finds the max servable context.

## Before you run — Session options + Add Input (right sidebar)

1. **Accelerator → GPU T4 x2**
2. **Internet → ON** (binary fallback + tunnel need it)
3. **Add Input** → attach all three (all public, repo vendors no binary):
   - `dieniw/qwen38-iq4xs` (weights, `Qwen3.8-27B-IQ4_XS.gguf`)
   - `leejiwon00/qwen-3-8-q5-binary` (FA_ALL_QUANTS server for Q5)
   - `leejiwon00/swift-llama-bin` (stock server, Q8 fallback)

Then run the cells top to bottom. The last cell keeps running on purpose: that is your
server. The `READY` banner prints `ENDPOINT` + `API KEY` (~10 min with the dataset binary).
"""

CONFIG_MD = """### Configuration
Edit `serve_config.json` before launching. Q8 max-precision mode: set
`cache_type_k/v` to `q8_0`, `mtp_n_max` to `0`, `ctx_size` to `262144`.
- `ctx_size`: `350000` default (YaRN scale derived automatically); `262144` for safe native.
- `cache_type_k/v`: `q5_0` (or `q5_1`) needs the FA_ALL_QUANTS binary; `q8_0` works on stock.
- `mtp_n_max`: `3` draft tokens; `0` disables speculative decoding.
- `parallel`: keep `1` — slots multiply KV residency.
- `n_gpu_layers`: `99` = all on GPU; `~60` spills to CPU RAM (slower, frees VRAM for KV).
- `keepalive_min`: auto-shutdown guard (Kaggle caps sessions at ~9 h).
"""

SCRIPT_MD = """### The serving script
The next cell writes the serving script (source: `kernel/serve_qwen38_boost.py`
in the repo) — nothing to edit here; scroll past it.
"""

LAUNCH_MD = """### Launch (leave this cell running: it is the server)
What you will see:
1. **Binary** — FA_ALL_QUANTS `llama-server` from the attached dataset (stock fallback, then prebuilt download, then source build)
2. **GPU** — `nvidia-smi` check (2x T4 expected) + VRAM report
3. **Weights** — the ~15.4 GiB GGUF (+ arch detect for the YaRN override)
4. **Server** — Q5 + MTP load, then the `READY` banner with `ENDPOINT` + `API KEY`, self-test + tok/s benchmark

Use it:
```
export OPENAI_BASE_URL="<ENDPOINT>"   # ends with /v1
export OPENAI_API_KEY="<API KEY>"
# model: qwen3.8-27b
```
"""


def cell(kind, text):
    lines = text.splitlines(keepends=True)
    c = {"cell_type": kind, "metadata": {}, "source": lines, "id": None}
    if kind == "code":
        c["execution_count"] = None
        c["outputs"] = []
    return c


def main():
    parser = argparse.ArgumentParser(description="Regenerate the release notebook")
    parser.add_argument("--ctx-size", type=int, default=350000)
    parser.add_argument("--mtp-n-max", type=int, default=3)
    args = parser.parse_args()

    src = SCRIPT.read_text()
    assert "ANTIGRAVITY BOOST" in src, f"{SCRIPT} missing boost header - refusing"

    config = {
        "ctx_size": args.ctx_size,
        "cache_type_k": "q5_0",
        "cache_type_v": "q5_0",
        "mtp_n_max": args.mtp_n_max,
        "flash_attn": "on",
        "parallel": 1,
        "bin_dataset": "leejiwon00/qwen-3-8-q5-binary",
        "bin_filename": "llama-server-b11095-sm75-faall.tar.gz",
        "keepalive_min": 480,
    }

    cells = [
        cell("markdown", INTRO_MD),
        cell("code", "%%writefile serve_config.json\n" + json.dumps(config, indent=2) + "\n"),
        cell("markdown", CONFIG_MD),
        cell("markdown", SCRIPT_MD),
        cell("code", f"%%writefile {SCRIPT.name}\n" + src),
        cell("markdown", LAUNCH_MD),
        cell("code", f"!python3 {SCRIPT.name}\n"),
    ]
    for i, c in enumerate(cells):
        c["id"] = f"serve-{i}"

    NOTEBOOK.parent.mkdir(parents=True, exist_ok=True)
    NOTEBOOK.write_text(json.dumps({
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }, indent=1, ensure_ascii=False) + "\n")

    embedded = "".join(json.loads(NOTEBOOK.read_text())["cells"][4]["source"])
    assert embedded == f"%%writefile {SCRIPT.name}\n" + src, "embedded script drifted!"
    print(f"wrote {NOTEBOOK.relative_to(BASE_DIR)} ({NOTEBOOK.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    sys.exit(main())
