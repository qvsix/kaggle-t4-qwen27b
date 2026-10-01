# Qwen3.8-27B IQ4_XS on Kaggle GPU T4 x2 (Q5 KV + MTP + YaRN)

Serve stock `Qwen3.8-27B-IQ4_XS` at up to **350k-token context** on
Kaggle's free `GPU T4 x2` (2x16 GB) with `llama-server` (pinned `b11095`), exposing an
**OpenAI-compatible endpoint** through a Cloudflare tunnel for Codex CLI / opencode / `curl`.

| Mode | KV cache | Context | MTP | Binary |
|---|---|---|---|---|
| Throughput (default) | `q5_0` + rotation | 350k via YaRN | 3 draft tokens | `leejiwon00/qwen-3-8-q5-binary` (FA_ALL_QUANTS) |
| Max precision | `q8_0` + rotation | 262144 native | 0 | `leejiwon00/swift-llama-bin` (stock) |

Why this stack:

* Weight format decides the engine: GGUF -> `llama.cpp`. Only `llama.cpp` reaches
  262k+ on 2x16 GB here: `IQ4_XS ~15.4 GiB` + `q5_0` KV `~5.9 GB @262k` split over both GPUs.
* Only 16/64 layers carry KV (hybrid GDN); the 48 recurrent layers keep F32 state
  untouched by `--cache-type-k/v`.
* Rotated KV (Hadamard on Q/K/V, on by default in `b11095`) keeps quantized cache
  usable at 100k+ context. `q5_0` cuts KV ~34% vs `q8_0` (22.5 vs 34 KiB/tok);
  `q4_0` KV is a known quality cliff; `f16` KV would need ~16 GB at 262k and not fit.
* Stock builds only ship FlashAttention for `f16`/`bf16`/`q8_0`/`q4_0`, so Q5 needs the
  `FA_ALL_QUANTS` binary; MTP uses the model's embedded Q8_0 draft head
  (`--spec-type draft-mtp --spec-draft-n-max 3`); YaRN scale derives from ctx automatically.
* If a 350k load OOMs, the script steps ctx down 20% per retry (floor 131072, then MTP
  stepdown) until one fits.

## Quick start

1. Session options (right sidebar): **Accelerator = GPU T4 x2**, **Internet = ON**.
2. **Add Input**: attach all three public datasets (repo vendors no binary):
   - [`dieniw/qwen38-iq4xs`](https://www.kaggle.com/datasets/dieniw/qwen38-iq4xs) (weights, `Qwen3.8-27B-IQ4_XS.gguf`)
   - [`leejiwon00/qwen-3-8-q5-binary`](https://www.kaggle.com/datasets/leejiwon00/qwen-3-8-q5-binary) (FA_ALL_QUANTS server)
   - [`leejiwon00/swift-llama-bin`](https://www.kaggle.com/datasets/leejiwon00/swift-llama-bin) (stock server, Q8 fallback)
3. Upload `notebook/swift-gpu-serve.ipynb` to Kaggle (or copy its cells) and run top to
   bottom. The last cell is the server; the `READY` banner prints `ENDPOINT` + `API KEY`
   (~10 min with the dataset binary).

Binary fallback order (automatic): FA_ALL_QUANTS dataset -> stock dataset -> prebuilt
download -> source build (CUDA sm_75, server target, `FA_ALL_QUANTS` on).

```bash
export OPENAI_BASE_URL="https://<your-tunnel>.trycloudflare.com/v1"
export OPENAI_API_KEY="sk-<your-key>"
# model: qwen3.8-27b

curl "$OPENAI_BASE_URL/chat/completions" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model": "qwen3.8-27b", "messages": [{"role": "user", "content": "Hello!"}]}'
```

Q8 max-precision mode: set `cache_type_k/v` to `q8_0`, `mtp_n_max` to `0`, `ctx_size` to
`262144` in the first (`serve_config.json`) cell before launching.

## YaRN 350k on Q8 (retired experiment, no separate code)

350k on stock Q8 was tried (`ctx 350000`, `--rope-scaling yarn --rope-scale 1.3351
--yarn-orig-ctx 262144`, `q8_0`, MTP off). Retired — the main script already covers it.

* Result: weights+KV fit but compute buffers OOM (~2.9 GB short on GPU0, ~1.8 GB on
  GPU1); the loader steps ctx down 20% per retry until one fits.
* `q6_0` is not a valid `--cache-type` in `b11095`.
* To retry: `ctx_size 350000`, `cache_type_k/v q8_0`, `mtp_n_max 0` — `rope_scale`
  derives from ctx automatically.

## If it fails

* `no GPU` — Accelerator is not T4 x2. Re-create the session with GPU.
* `GGUF missing` — weights dataset not attached (see step 2).
* Load OOM — watch for `ctx-stepdown`/`mtp-stepdown` phases; the session finds a fittable
  ctx itself. Or pre-set `ctx_size 262144`, `mtp_n_max 0`. Keep `parallel: 1`; slots
  multiply KV residency.
* First 100k+ prompt is slow on T4 (minutes of prefill) — expected, not a bug.

## Files

* `kernel/serve_qwen38_boost.py` — the serving script (source of truth).
* `notebook/swift-gpu-serve.ipynb` — run-all notebook, embeds the script byte-identical.
* `make_notebook.py` — regenerates the notebook from the script; never hand-edit the
  script cell (`python3 make_notebook.py`).

## License

The code here is MIT. Model weights keep their own licenses.

## Acknowledgments

Weights: stock `Qwen3.8-27B-IQ4_XS` via `dieniw/qwen38-iq4xs` (upstream `bartowski/Qwen3.8-27B-GGUF`).
`llama.cpp` `b11095` for the server; Cloudflare for the tunnel.
