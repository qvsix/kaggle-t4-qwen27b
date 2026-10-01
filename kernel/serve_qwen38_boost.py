"""Serve Qwen3.8-27B IQ4_XS GGUF on Kaggle GPU T4 x2 with llama-server.

ANTIGRAVITY BOOST RECIPE:
  Priority 0: Q5 KV cache (q5_0 / q5_1) with rotation (Hadamard on Q/K/V)
              via FA_ALL_QUANTS prebuilt binary (leejiwon00/qwen-3-8-q5-binary).
  Priority 1: MTP enabled (--spec-type draft-mtp --spec-draft-n-max 3) using
              embedded Q8_0 MTP draft head for 1.5x-2.5x speculative decode speed.
  Priority 2: Native 262,144 context leaving ~5.8 GiB VRAM headroom per GPU for
              MTP draft graph and compute buffers without danger-zone OOM.
              YaRN dynamic scaling automatically engages if ctx > 262144.

Target:
  Weights: IQ4_XS ~15.4 GiB (~7.7 GiB / GPU)
  KV cache: q5_0 ~5.6 GiB @ 262k (~2.81 GiB / GPU)
  Total base residency: ~10.5 GiB / GPU out of 16 GB, leaving ~5.4 GiB headroom for MTP.

Steps:
  1/5 binary   - FA_ALL_QUANTS llama-server b11095 (dataset, else prebuilt, else source build)
  2/5 gpu      - nvidia-smi check (2x T4 sm_75 expected), VRAM report
  3/5 weights  - locate mounted GGUF dataset (+ arch detect for YaRN override-kv)
  4/5 server   - start llama-server with Q5 + MTP + safe VRAM context
  5/5 tunnel   - cloudflared URL, READY banner, self-test & benchmark, keepalive
"""

import collections
import glob
import json
import os
import re
import secrets
import shutil
import subprocess
import sys
import tarfile
import threading
import time
import urllib.request
from pathlib import Path

CFG = None  # __LAUNCHER_CONFIG__

DEFAULTS = {
    # Pinned llama.cpp build: post-2026-04-01 KV rotation, GDN rsqrt fix,
    # Qwen3.5 recurrent_layers metadata, MTP KV-alloc fix.
    "llama_build": "b11095",
    "llama_tag": "b11095",
    "cuda_arch": "75",           # T4 = sm_75; single-arch build keeps nvcc fast
    "llama_asset_cuda": "llama-b11095-bin-ubuntu-cuda-12.8-x64.tar.gz",
    "llama_asset_cudart": "cudart-llama-b11095-bin-ubuntu-cuda-12.8-x64.tar.gz",
    "dataset": "dieniw/qwen38-iq4xs",
    "gguf_filename": "Qwen3.8-27B-IQ4_XS.gguf",
    # FA_ALL_QUANTS binary dataset: pre-compiled with -DGGML_CUDA_FA_ALL_QUANTS=ON
    # so FlashAttention supports q5_0 and q5_1.
    "bin_dataset": "leejiwon00/qwen-3-8-q5-binary",
    "bin_filename": "llama-server-b11095-sm75-faall.tar.gz",
    "stock_bin_dataset": "leejiwon00/swift-llama-bin",
    "stock_bin_filename": "llama-server-b11095-sm75.tar.gz",
    # PRIORITY 0: Q5 KV cache (22.5 KiB/tok vs 34 KiB/tok for q8_0).
    # FlashAttention required (-fa on); rotation stays ON.
    "cache_type_k": "q5_0",
    "cache_type_v": "q5_0",
    "flash_attn": "on",
    "fa_all_quants": True,
    # PRIORITY 1: MTP speculative decoding enabled.
    # Embedded Q8_0 MTP draft head inside the same GGUF; 3 draft tokens is the sweet spot.
    "mtp_n_max": 3,
    # PRIORITY 2: Safe native context leaving headroom for MTP without OOM danger.
    "ctx_size": 350000,
    "yarn_orig_ctx": 262144,
    "rope_scaling": "yarn",
    "rope_scale": None,          # derived dynamically if ctx > yarn_orig_ctx
    "yarn_ext_factor": None,
    "yarn_attn_factor": None,
    "yarn_context_length": 1000000,
    "n_gpu_layers": 99,          # offload all layers to GPU
    "split_mode": "layer",       # layer-split across both T4s
    "tensor_split": "",          # empty = even split
    "parallel": 1,               # server slots; keep 1 for max context / agent headroom
    "threads": -1,               # -1 = auto (all CPUs)
    "jinja": True,               # embedded chat template
    # Sampling defaults from model card
    "temp": 1.0,
    "top_p": 0.95,
    "top_k": 20,
    "min_p": 0.0,
    "repeat_penalty": 1.0,
    "presence_penalty": 0.0,
    "reasoning_effort": "default",
    "keepalive_min": 480,
    "api_key": "",
    "ntfy_topic": "",
    "served_model_name": "qwen3.8-27b",
    "verbose": False,
}

CFG = {**DEFAULTS, **(CFG or {})}
_cfg_file = Path("serve_config.json")
if _cfg_file.exists():
    try:
        CFG.update(json.loads(_cfg_file.read_text()))
    except Exception as e:
        print(f"Warning: could not parse serve_config.json: {e}", flush=True)

if not CFG["api_key"]:
    CFG["api_key"] = "sk-" + secrets.token_hex(16)

PORT = 8000
LLAMA_DIR = Path("/tmp/llama")
LLAMA_BIN = LLAMA_DIR / "llama-server"
LLAMA_EXE = None
BIN_EXTRACT = Path("/tmp/llamabin")
WORK = Path("/kaggle/working") if Path("/kaggle/working").is_dir() else Path("/tmp")
RAW_LOG = WORK / "llama.log"
CLOUDFLARED = Path("/tmp/cloudflared")
T0 = time.time()

# KV Rotation: Hadamard on Q/K/V must stay ON for Q5/Q8 cache quality
os.environ.pop("LLAMA_ATTN_ROT_DISABLE", None)

_raw = open(RAW_LOG, "a", buffering=1)


def log(*parts):
    line = time.strftime("[%H:%M:%S] ") + " ".join(str(p) for p in parts)
    print(line, flush=True)
    _raw.write(line + "\n")


def elapsed():
    s = int(time.time() - T0)
    return f"{s // 60} min {s % 60:02d} s"


def banner(step, title, note=""):
    log("")
    log("=" * 70)
    log(f" STEP {step}/5  {title}" + (f"   ({note})" if note else "") + f"   [{elapsed()} so far]")
    log("=" * 70)


def publish(phase, **extra):
    log(f"PHASE {phase}", json.dumps(extra) if extra else "")
    if not CFG["ntfy_topic"]:
        return
    try:
        body = {"topic": CFG["ntfy_topic"], "title": f"kaggle-t4-qwen27b {phase}",
                "message": json.dumps({"phase": phase, **extra})}
        req = urllib.request.Request("https://ntfy.sh", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=10)
    except Exception as e:
        log(f"(ntfy publish failed: {e})")


def sh(cmd, tag, show=None):
    show = CFG["verbose"] if show is None else show
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in p.stdout:
        line = line.rstrip()
        if not line:
            continue
        _raw.write(f"[{tag}] {line}\n")
        if show:
            print(f"[{tag}] {line[:400]}", flush=True)
    return p.wait()


def llama_url(asset):
    return (f"https://github.com/ggml-org/llama.cpp/releases/download/"
            f"{CFG['llama_build']}/{asset}")


def fetch_cloudflared():
    if CLOUDFLARED.exists():
        return True
    try:
        urllib.request.urlretrieve(
            "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64",
            CLOUDFLARED)
        CLOUDFLARED.chmod(0o755)
        return True
    except Exception as e:
        log(f"(cloudflared download failed: {e})")
        return False


def _use_lib_dir(d):
    parts = [p for p in os.environ.get("LD_LIBRARY_PATH", "").split(":")
             if p and "llama" not in p.lower()]
    os.environ["LD_LIBRARY_PATH"] = str(d) + (":" + ":".join(parts) if parts else "")
    log(f"   LD_LIBRARY_PATH -> {os.environ['LD_LIBRARY_PATH'][:220]}")


def _try_bin_candidate(p):
    try:
        if (p.parent / "libllama-server-impl.so").exists() or list(p.parent.glob("libllama*.so*")):
            _use_lib_dir(p.parent)
        p.chmod(0o755)
        out = subprocess.run([str(p), "--version"], capture_output=True,
                             text=True, timeout=120, env=os.environ.copy())
        ver = ((out.stdout or "") + (out.stderr or "")).strip()[:200]
        if out.returncode == 0 and "llama" in ver.lower():
            log(f"   binary ready: {ver} ({p})")
            return str(p)
        log(f"   (binary {p} unusable: rc={out.returncode} {ver[:120]})")
    except Exception as e:
        log(f"   (binary {p} failed: {str(e)[:120]})")
    return None


def _materialize_bin_dir(src):
    dest = BIN_EXTRACT / "lib"
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    fams = {}
    for f in sorted(Path(src).iterdir()):
        if f.is_symlink():
            continue
        if not (f.is_file() and f.stat().st_size > 0):
            continue
        if f.name != "llama-server" and ".so" not in f.name:
            continue
        shutil.copy2(f, dest / f.name)
        m = re.match(r"(lib[\w-]+\.so)(?:\.|$)", f.name)
        if m:
            fams.setdefault(m.group(1), []).append(f.name)
    for fam, names in fams.items():
        real = sorted(names, key=len)[-1]
        parts = real.split(".")
        prefix = parts[0] + "." + parts[1]
        if prefix != real:
            p = dest / prefix
            if p.is_symlink() or p.exists():
                p.unlink()
            p.symlink_to(real)
        curr = prefix
        for part in parts[2:]:
            curr = f"{curr}.{part}"
            if curr != real:
                p = dest / curr
                if p.is_symlink() or p.exists():
                    p.unlink()
                p.symlink_to(real)
    exe = dest / "llama-server"
    exe.chmod(0o755)
    return str(exe), str(dest)


def _safe_extract_tar(tarball, dest):
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tarball) as tf:
        try:
            tf.extractall(dest, filter="fully_trusted")
        except TypeError:
            tf.extractall(dest)


def find_bin():
    """Locate prebuilt llama-server binary.
    Prioritizes FA_ALL_QUANTS archives for Q5 support, then any valid binary."""
    # 1. Search for FA_ALL_QUANTS tarballs first
    cands_tar = sorted(glob.glob("/kaggle/input/**/*.tar.gz", recursive=True))
    faall_tars = [h for h in cands_tar if "faall" in Path(h).name.lower()]
    other_tars = [h for h in cands_tar if "llama-server" in Path(h).name and "faall" not in Path(h).name.lower()]

    for hit in faall_tars + other_tars:
        log(f"   found tarball candidate: {hit}")
        try:
            extract_target = BIN_EXTRACT / Path(hit).stem
            _safe_extract_tar(hit, extract_target)
            for cand in sorted(extract_target.rglob("llama-server")):
                if cand.is_file():
                    got = _try_bin_candidate(cand)
                    if got:
                        return got
        except Exception as e:
            log(f"   (tarball {hit} extract failed: {e})")

    # 2. Search uncompressed dataset mounts
    pats = [f"/kaggle/input/{CFG['bin_filename']}",
            f"/kaggle/input/*/{CFG['bin_filename']}",
            f"/kaggle/input/{CFG['stock_bin_filename']}",
            f"/kaggle/input/*/{CFG['stock_bin_filename']}",
            "/kaggle/input/**/bin/llama-server*",
            "/kaggle/input/**/llama-server*"]
    for pat in pats:
        for hit in sorted(glob.glob(pat, recursive=True)):
            p = Path(hit)
            if not p.is_file() or p.suffix in (".gz", ".tar"):
                continue
            if list(p.parent.glob("*.so*")):
                exe, _ = _materialize_bin_dir(p.parent)
                got = _try_bin_candidate(Path(exe))
            else:
                got = _try_bin_candidate(p)
            if got:
                return got
    return None


def fetch_llama():
    for h in sorted(LLAMA_DIR.rglob("llama-server")):
        if h.is_file() and (h.parent / "libllama-server-impl.so").exists():
            got = _try_bin_candidate(h)
            if got:
                return got
    LLAMA_DIR.mkdir(parents=True, exist_ok=True)
    for asset in (CFG["llama_asset_cuda"], CFG["llama_asset_cudart"]):
        tgz = LLAMA_DIR / asset
        try:
            if not tgz.exists():
                log(f"   downloading {asset}...")
                urllib.request.urlretrieve(llama_url(asset), tgz)
            _safe_extract_tar(tgz, LLAMA_DIR)
            for cand in sorted(LLAMA_DIR.rglob("llama-server")):
                if cand.is_file():
                    got = _try_bin_candidate(cand)
                    if got:
                        return got
        except Exception as e:
            log(f"   ({asset} failed: {e})")
    log("   prebuilts unusable on this environment; compiling from source...")
    return build_llama()


SRC_DIR = Path("/tmp/llama-src")
BUILD_DIR = SRC_DIR / "build"


def ensure_cmake():
    try:
        r = subprocess.run(["cmake", "--version"], capture_output=True, text=True, timeout=30)
        m = re.search(r"cmake version (\d+)\.(\d+)", r.stdout or "")
        if r.returncode == 0 and m and (int(m.group(1)), int(m.group(2))) >= (3, 24):
            log(f"   cmake present: {m.group(0)}")
            return True
    except FileNotFoundError:
        pass
    log("   system cmake missing or old; installing cmake>=3.28 via pip...")
    if sh([sys.executable, "-m", "pip", "install", "-q", "cmake>=3.28"], "pip") != 0:
        return False
    r = subprocess.run([sys.executable, "-c", "import sysconfig; print(sysconfig.get_path('scripts'))"],
                       capture_output=True, text=True)
    bindir = r.stdout.strip()
    if bindir and Path(bindir, "cmake").exists():
        os.environ["PATH"] = bindir + ":" + os.environ.get("PATH", "")
    return shutil.which("cmake") is not None


def _cuda_driver_stub(env):
    nvcc = shutil.which("nvcc", path=env.get("PATH", "")) or shutil.which("nvcc")
    cuda_root = str(Path(nvcc).resolve().parent.parent) if nvcc else "/usr/local/cuda"
    cands = []
    try:
        lp = subprocess.run(["ldconfig", "-p"], capture_output=True, text=True, timeout=30)
        for line in (lp.stdout or "").splitlines():
            if "libcuda.so" in line and "=>" in line:
                cands.append(line.split("=>", 1)[1].strip())
    except Exception:
        pass
    for d in ("/usr/local/nvidia/lib64", "/usr/lib/x86_64-linux-gnu",
              f"{cuda_root}/lib64", f"{cuda_root}/lib64/stubs",
              f"{cuda_root}/targets/x86_64-linux/lib",
              f"{cuda_root}/targets/x86_64-linux/lib/stubs"):
        cands += sorted(glob.glob(f"{d}/libcuda.so*"))
    seen = list(dict.fromkeys(c for c in cands if os.path.exists(c)))
    if any(c.endswith("libcuda.so") for c in seen):
        linkable = next(c for c in seen if c.endswith("libcuda.so"))
        return cuda_root, str(Path(linkable).parent)
    versioned = next((c for c in seen if re.search(r"libcuda\.so\.\d", c)), None)
    if versioned:
        stub = Path("/tmp/cuda-stubs")
        stub.mkdir(parents=True, exist_ok=True)
        link = stub / "libcuda.so"
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(versioned)
        return cuda_root, str(stub)
    return cuda_root, None


def build_llama():
    if not ensure_cmake():
        return None
    env = os.environ.copy()
    if not (SRC_DIR / ".git").exists():
        log(f"   cloning llama.cpp @ {CFG['llama_tag']}...")
        publish("cloning", tag=CFG["llama_tag"])
        if sh(["git", "clone", "--depth", "1", "--branch", CFG["llama_tag"],
               "https://github.com/ggml-org/llama.cpp.git", str(SRC_DIR)], "git") != 0:
            return None
    jobs = os.cpu_count() or 4
    cuda_root, extra_lib = _cuda_driver_stub(env)
    cfg_cmd = ["cmake", "-S", str(SRC_DIR), "-B", str(BUILD_DIR),
                "-DGGML_CUDA=ON", "-DCMAKE_BUILD_TYPE=Release",
                f"-DCMAKE_CUDA_ARCHITECTURES={CFG['cuda_arch']}",
                "-DLLAMA_BUILD_SERVER=ON",
                "-DLLAMA_BUILD_TESTS=OFF", "-DLLAMA_BUILD_EXAMPLES=OFF",
                "-DGGML_BUILD_TESTS=OFF", "-DGGML_BUILD_EXAMPLES=OFF"]
    if CFG.get("fa_all_quants"):
        cfg_cmd.append("-DGGML_CUDA_FA_ALL_QUANTS=ON")
    if os.path.isdir(cuda_root):
        cfg_cmd.append(f"-DCUDAToolkit_ROOT={cuda_root}")
    if extra_lib:
        cfg_cmd.append(f"-DCMAKE_LIBRARY_PATH={extra_lib}")

    log(f"   cmake configure (CUDA sm_{CFG['cuda_arch']}, FA_ALL_QUANTS={CFG.get('fa_all_quants')})...")
    p = subprocess.Popen(cfg_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
    for line in p.stdout:
        _raw.write(f"[cmake] {line}")
    if p.wait() != 0:
        log("   cmake configure failed")
        return None

    log(f"   compiling llama-server with -j{jobs}...")
    publish("building", jobs=jobs)
    p = subprocess.Popen(["cmake", "--build", str(BUILD_DIR), "--target", "llama-server", "-j", str(jobs)],
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
    last = time.time()
    for line in p.stdout:
        _raw.write(f"[build] {line}")
        if time.time() - last > 300:
            last = time.time()
            log("   ... still compiling")
    if p.wait() != 0:
        log("   build failed")
        return None

    exe = BUILD_DIR / "bin" / "llama-server"
    if not exe.exists():
        cands = [c for c in BUILD_DIR.rglob("llama-server") if c.is_file() and not c.is_symlink()]
        if not cands:
            return None
        exe = cands[0]
    exe.chmod(0o755)

    lib_hits = sorted(BUILD_DIR.rglob("libllama-server-impl.so*"))
    if lib_hits:
        _use_lib_dir(lib_hits[0].parent)

    dest = WORK / CFG["bin_filename"]
    with tarfile.open(dest, "w:gz") as tf:
        tf.add(exe, arcname="bin/llama-server")
        seen = {"bin/llama-server"}
        for so in sorted(BUILD_DIR.rglob("*.so*")):
            if so.is_file() and so.resolve() != exe.resolve():
                arc = f"bin/{so.name}"
                if arc not in seen:
                    seen.add(arc)
                    tf.add(so, arcname=arc)
    log(f"   harvested binary archive to {dest} ({dest.stat().st_size / 1e6:.1f} MB)")
    return str(exe)


def gpu_check():
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.total",
                            "--format=csv,noheader"], capture_output=True, text=True, timeout=60)
    except FileNotFoundError:
        return 0, "nvidia-smi not found: not a GPU session?"
    if r.returncode != 0:
        return 0, f"nvidia-smi failed: {(r.stderr or r.stdout)[:200]}"
    gpus = [l.strip() for l in (r.stdout or "").splitlines() if l.strip()]
    return len(gpus), "; ".join(gpus)


def gpu_vram_snapshot():
    """Query current VRAM usage per GPU and compute remaining headroom in MiB and GiB."""
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used,memory.total",
                            "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=30)
        if r.returncode != 0:
            return []
        res = []
        for line in (r.stdout or "").splitlines():
            parts = [p.strip() for p in line.split(",") if p.strip()]
            if len(parts) >= 3:
                idx, used, total = int(parts[0]), float(parts[1]), float(parts[2])
                free = total - used
                res.append({
                    "gpu": idx,
                    "used_mib": round(used, 1),
                    "total_mib": round(total, 1),
                    "free_mib": round(free, 1),
                    "free_gib": round(free / 1024.0, 2)
                })
        return res
    except Exception:
        return []


def find_gguf():
    exact = [f"/kaggle/input/{CFG['gguf_filename']}",
             f"/kaggle/input/*/{CFG['gguf_filename']}",
             f"/kaggle/input/datasets/*/*/{CFG['gguf_filename']}"]
    for pat in exact:
        hits = sorted(glob.glob(pat))
        if hits:
            return hits[0]
    cands = sorted(glob.glob("/kaggle/input/**/*.gguf", recursive=True))
    return cands[-1] if cands else None


def detect_gguf_arch(path):
    import struct
    try:
        with open(path, "rb") as f:
            if f.read(4) != b"GGUF":
                return None
            ver, n_tensors, n_kv = struct.unpack("<III", f.read(12))
            if ver not in (2, 3):
                return None
            def read_str():
                l = struct.unpack("<Q", f.read(8))[0]
                return f.read(l).decode("utf-8", "replace")
            def skip_val(t):
                if t in (0, 7): f.seek(1, 1)
                elif t in (1, 8): f.seek(2, 1)
                elif t in (2, 3, 9): f.seek(4, 1)
                elif t in (4, 5, 10, 11): f.seek(8, 1)
                elif t == 6: f.seek(1, 1)
                elif t == 8: read_str()
                elif t == 9:
                    sub_t, cnt = struct.unpack("<IQ", f.read(12))
                    for _ in range(cnt): skip_val(sub_t)
            for _ in range(n_kv):
                k = read_str()
                t = struct.unpack("<I", f.read(4))[0]
                if k == "general.architecture":
                    if t == 8:
                        return read_str()
                skip_val(t)
    except Exception as e:
        log(f"   (GGUF arch detect failed: {str(e)[:120]})")
    return None


def server_args(cfg, model_path):
    args = [str(LLAMA_EXE or LLAMA_BIN),
            "-m", model_path,
            "-a", cfg["served_model_name"],
            "--host", "127.0.0.1", "--port", str(PORT),
            "-c", str(cfg["ctx_size"]),
            "-ctk", cfg["cache_type_k"], "-ctv", cfg["cache_type_v"],
            "-fa", cfg["flash_attn"],
            "-ngl", str(cfg["n_gpu_layers"]),
            "-sm", cfg["split_mode"],
            "-np", str(cfg["parallel"]),
            "-t", str(cfg["threads"] if cfg["threads"] > 0 else (os.cpu_count() or 8)),
            "--api-key", cfg["api_key"],
            "--temp", str(cfg["temp"]), "--top-p", str(cfg["top_p"]),
            "--top-k", str(cfg["top_k"]), "--min-p", str(cfg["min_p"]),
            "--repeat-penalty", str(cfg["repeat_penalty"]),
            "--presence-penalty", str(cfg["presence_penalty"]),
            "--reasoning-effort", cfg["reasoning_effort"],
            "-to", "3600"]
    if cfg["jinja"]:
        args.append("--jinja")
    else:
        args.append("--no-jinja")
    if cfg["tensor_split"]:
        args += ["-ts", cfg["tensor_split"]]

    # PRIORITY 1: MTP Multi-Token Prediction draft head
    if cfg.get("mtp_n_max") and int(cfg["mtp_n_max"]) > 0:
        args += ["--spec-type", "draft-mtp",
                 "--spec-draft-n-max", str(int(cfg["mtp_n_max"]))]

    # PRIORITY 2: YaRN rope scaling (only when ctx > yarn_orig_ctx)
    if cfg.get("rope_scaling") and cfg.get("rope_scale"):
        args += ["--rope-scaling", cfg["rope_scaling"],
                 "--rope-scale", str(cfg["rope_scale"])]
        if cfg.get("yarn_orig_ctx"):
            args += ["--yarn-orig-ctx", str(cfg["yarn_orig_ctx"])]
        if cfg.get("yarn_ext_factor") is not None:
            args += ["--yarn-ext-factor", str(cfg["yarn_ext_factor"])]
        if cfg.get("yarn_attn_factor") is not None:
            args += ["--yarn-attn-factor", str(cfg["yarn_attn_factor"])]

    for kv in cfg.get("override_kv") or []:
        args += ["--override-kv", kv]

    return args


def healthy():
    try:
        req = urllib.request.Request(f"http://127.0.0.1:{PORT}/health")
        with urllib.request.urlopen(req, timeout=5) as r:
            return r.status == 200
    except Exception:
        return False


def wait_healthy(server, expect_min, die=True):
    t = time.time()
    while time.time() - t < 5400:
        if server.poll() is not None:
            tail = "\n".join(list(server.tail)[-40:]) if getattr(server, "tail", None) else ""
            if die:
                server_died(server, "failed", step="server")
            return "died", tail
        if healthy():
            return "ready", int(time.time() - t)
        el = int(time.time() - t)
        if el and el % 120 < 6:
            publish("loading", elapsed_s=el)
            log(f"   ... {el // 60} min into load (~{expect_min} min expected for weights + KV alloc)")
        time.sleep(5)
    publish("failed", step="health-timeout")
    sys.exit(1)


def server_died(server, phase, **extra):
    tail = "\n".join(list(server.tail)[-40:]) if getattr(server, "tail", None) else ""
    log(f"server exited rc={server.returncode}")
    if tail:
        log("--- last output ---\n" + tail[-3000:])
    publish(phase, rc=server.returncode, tail=tail[-2200:], **extra)
    sys.exit(1)


def chat(messages, max_tokens=128, timeout=600):
    body = {"model": CFG["served_model_name"], "messages": messages,
            "max_tokens": max_tokens, "temperature": 0.0}
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/v1/chat/completions", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {CFG['api_key']}"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)["choices"][0]["message"]["content"]


def self_test():
    """Verify endpoint functionality + benchmark streaming tok/s to prove MTP speedup."""
    tps = None
    try:
        ans = chat([{"role": "user", "content": "Reply with exactly: OK"}], max_tokens=128)
        log(f"   self-test chat response -> {ans.strip()[:60]!r}")

        # Streaming benchmark
        body = {"model": CFG["served_model_name"],
                "messages": [{"role": "user", "content": "Write a concise python function for binary search."}],
                "max_tokens": 128, "temperature": 0.0, "stream": True,
                "stream_options": {"include_usage": True}}
        req = urllib.request.Request(
            f"http://127.0.0.1:{PORT}/v1/chat/completions", data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {CFG['api_key']}"})
        t0 = time.time()
        ttft = None
        gen = 0
        with urllib.request.urlopen(req, timeout=900) as r:
            for raw in r:
                line = raw.decode("utf-8", "ignore").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    obj = json.loads(data)
                except Exception:
                    continue
                delta = ((obj.get("choices") or [{}])[0].get("delta") or {}).get("content")
                if delta:
                    if ttft is None:
                        ttft = time.time() - t0
                    gen += 1
                if obj.get("usage"):
                    gen = obj["usage"].get("completion_tokens", gen)
        total = time.time() - t0
        if gen and ttft is not None and total > ttft:
            tps = (gen - 1) / (total - ttft)
        publish("benchmark", decode_tok_s=round(tps, 1) if tps else None,
                mtp=CFG.get("mtp_n_max"), kv=CFG["cache_type_k"])
        log(f"   benchmark: {tps:.1f} tok/s decode speed (MTP={CFG.get('mtp_n_max')}, KV={CFG['cache_type_k']})"
            if tps else "   benchmark complete")
    except Exception as e:
        publish("benchmark-error", err=str(e)[:200])
        log(f"   SELF-TEST ERROR: {str(e)[:300]}")
    return tps


KV_KIB_PER_TOK = {
    "f32": 128.0, "f16": 64.0, "bf16": 64.0, "f8_e5m2": 32.0,
    "q8_0": 34.0, "q6_0": 26.0, "q5_1": 24.0, "q5_0": 22.5,
    "q4_1": 20.0, "q4_0": 18.0
}


def kv_estimate_gb(cfg):
    per_tok = max(KV_KIB_PER_TOK.get(cfg["cache_type_k"], 22.5),
                  KV_KIB_PER_TOK.get(cfg["cache_type_v"], 22.5))
    return per_tok, per_tok * cfg["ctx_size"] / 1e6


def start_tunnel():
    banner(5, "Public Tunnel Endpoint")
    url = None
    for _ in range(60):
        if CLOUDFLARED.exists():
            break
        time.sleep(2)
    if CLOUDFLARED.exists():
        tunnel = subprocess.Popen([str(CLOUDFLARED), "tunnel", "--url",
                                   f"http://127.0.0.1:{PORT}",
                                   "--no-autoupdate", "--protocol", "quic"],
                                  stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        pat = re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com")
        lines = []

        def pump_cf():
            for line in tunnel.stdout:
                lines.append(line.rstrip())
                _raw.write(f"[cloudflared] {line}\n")
        threading.Thread(target=pump_cf, daemon=True).start()
        deadline = time.time() + 180
        while time.time() < deadline and url is None:
            for ln in lines:
                m = pat.search(ln)
                if m:
                    url = m.group(0).rstrip("/")
                    break
            time.sleep(1)
    if url:
        log(f"   Public Endpoint URL : {url}/v1")
        publish("tunnel-url", endpoint=f"{url}/v1")
    else:
        publish("tunnel-failed", note="server accessible on localhost:8000")
    return url


def classify_death(tail):
    low = (tail or "").lower()
    if any(k in low for k in ("out of memory", "cudamalloc failed", "failed to allocate",
                              "bad_alloc", "cudamalloc failed")):
        return "oom"
    if any(k in low for k in ("cache-type", "cache type", "quant", "rope", "yarn", "rope-scale")):
        return "flags"
    return "other"


def normalize_yarn():
    """Derive YaRN rope_scale from ctx_size; below/at native ctx, YaRN is inactive."""
    orig = CFG.get("yarn_orig_ctx", 262144)
    if CFG["ctx_size"] <= orig:
        CFG["rope_scale"] = None
        CFG["override_kv"] = []
    else:
        CFG["rope_scale"] = round(CFG["ctx_size"] / orig, 4)


def stepdown_on_oom():
    """Multi-stage graceful stepdown for load OOM:
    1. If ctx > 131072, shrink ctx by 20% down to floor 131072.
    2. If ctx is already at floor, drop MTP draft tokens (3 -> 2 -> 1 -> 0).
    """
    orig_ctx = CFG.get("yarn_orig_ctx", 262144)
    if CFG["ctx_size"] > 131072:
        new_ctx = max(131072, int(CFG["ctx_size"] * 0.8))
        if new_ctx < CFG["ctx_size"]:
            CFG["ctx_size"] = new_ctx
            normalize_yarn()
            return f"ctx-stepdown to {new_ctx}"
    # Try stepping down MTP if ctx can't be reduced further
    if CFG.get("mtp_n_max") and int(CFG["mtp_n_max"]) > 0:
        old_mtp = int(CFG["mtp_n_max"])
        CFG["mtp_n_max"] = old_mtp - 1
        return f"mtp-stepdown from {old_mtp} to {CFG['mtp_n_max']}"
    return None


def launch_server(cfg, model_path):
    publish("server-launch", ctx=cfg["ctx_size"], kv=f"{cfg['cache_type_k']}/{cfg['cache_type_v']}",
            mtp=cfg.get("mtp_n_max"), rope=f"yarn:{cfg.get('rope_scale')}" if cfg.get("rope_scale") else "native")
    tail = collections.deque(maxlen=200)
    env = os.environ.copy()
    env.pop("LLAMA_ATTN_ROT_DISABLE", None)
    args = server_args(cfg, model_path)
    p = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
    NOISE = ("load_stat", "llama_model_load", "print_info")

    def pump():
        for line in p.stdout:
            line = line.rstrip()
            if not line:
                continue
            tail.append(line)
            _raw.write(f"[llama] {line}\n")
            if cfg["verbose"] or not any(k in line for k in NOISE):
                if len(line) < 500 or "error" in line.lower() or cfg["verbose"]:
                    print(f"[llama] {line[:400]}", flush=True)
    threading.Thread(target=pump, daemon=True).start()
    p.tail = tail
    return p


def main():
    if "--print-args" in sys.argv:
        normalize_yarn()
        print(json.dumps(server_args(CFG, "<MODEL>"), indent=2))
        return 0

    log("=" * 70)
    log("ANTIGRAVITY BOOST SERVER - Qwen3.8-27B IQ4_XS (Q5 KV + MTP + 262k Safe Context)")
    log("=" * 70)

    # 1. Binary resolution
    banner(1, "llama-server Binary", f"build {CFG['llama_build']}")
    threading.Thread(target=fetch_cloudflared, daemon=True).start()
    publish("install", build=CFG["llama_build"])
    t = time.time()
    exe = find_bin()
    if not exe:
        exe = fetch_llama()
    if not exe:
        publish("failed", step="install", tail="llama-server binary could not be found or built")
        sys.exit(1)
    global LLAMA_EXE
    LLAMA_EXE = exe
    publish("installed", secs=int(time.time() - t))
    log(f"   llama-server binary verified at: {LLAMA_EXE}")
    log("   rotation ON (Hadamard Q/K/V active for Q5 KV fidelity)")

    # 2. GPU verification
    banner(2, "GPU Check", "T4 x2 target (sm_75)")
    n_gpus, desc = gpu_check()
    log(f"   GPUs available: {n_gpus} ({desc[:300]})")
    if n_gpus == 0:
        msg = f"no GPU in this session ({desc}). Set Accelerator to GPU T4 x2."
        publish("failed", step="no-gpu", tail=msg)
        sys.exit(1)
    publish("gpu-ok", count=n_gpus, desc=desc[:300])

    # 3. Model weights
    banner(3, "Model Weights", CFG["gguf_filename"])
    model_path = find_gguf()
    if model_path and os.path.getsize(model_path) > 10_000_000_000:
        gb_size = os.path.getsize(model_path) / 1e9
        publish("weights-mounted", path=model_path, gb=round(gb_size, 1))
        log(f"   found {model_path} ({gb_size:.1f} GB)")
        arch = detect_gguf_arch(model_path)
        if arch and CFG.get("rope_scaling"):
            CFG["override_kv"] = [f"{arch}.context_length=int:{CFG['yarn_context_length']}"]
            log(f"   GGUF arch: {arch} -> override_kv {CFG['override_kv'][0]}")
    else:
        msg = f"dataset {CFG['dataset']} missing or GGUF not found."
        log("   " + msg)
        publish("failed", step="no-weights", tail=msg)
        sys.exit(1)

    # 4. Launch server with OOM resilience
    normalize_yarn()
    url = None
    retries = 4
    startup = None

    while True:
        kib, gb = kv_estimate_gb(CFG)
        banner(4, "Starting llama-server",
               f"ctx={CFG['ctx_size']}, KV={CFG['cache_type_k']}/{CFG['cache_type_v']}, "
               f"MTP={CFG.get('mtp_n_max')}, slots={CFG['parallel']}")
        log(f"   VRAM residency: ~{gb:.1f} GB KV cache (~{kib:.1f} KiB/tok) + ~15.4 GB weights split over {n_gpus} GPUs")
        server = launch_server(CFG, model_path)
        if url is None and not getattr(start_tunnel, "done", False):
            url = start_tunnel()
            start_tunnel.done = True

        outcome, info = wait_healthy(server, 10, die=False)
        if outcome == "ready":
            startup = info
            break
        kind = classify_death(info) if outcome == "died" else "timeout"
        if kind == "oom" and retries > 0:
            step_action = stepdown_on_oom()
            if step_action:
                retries -= 1
                log(f"   Load OOM encountered; applied {step_action} ({retries} retries left)")
                publish("recovery-stepdown", action=step_action, retries_left=retries)
                continue
        if outcome == "died":
            server_died(server, "failed", step="server")
        publish("failed", step="health-timeout")
        sys.exit(1)

    # 5. Announce, self-test & serve
    vram_stats = gpu_vram_snapshot()
    vram_desc = ", ".join(f"GPU{g['gpu']}: {g['free_gib']} GiB free ({g['used_mib']}/{g['total_mib']} MiB)"
                           for g in vram_stats) if vram_stats else "N/A"
    if vram_stats:
        log(f"   VRAM Headroom Snapshot: {vram_desc}")
    publish("serving", startup_secs=startup)
    base = f"{url}/v1" if url else "http://127.0.0.1:8000/v1"
    log("")
    log("#" * 70)
    log(f"#  ANTIGRAVITY BOOST SERVER READY ({elapsed()} from start)")
    log(f"#  ENDPOINT : {base}")
    log(f"#  API KEY  : {CFG['api_key']}")
    log(f"#  MODEL    : {CFG['served_model_name']}")
    log(f"#  CONTEXT  : {CFG['ctx_size']} (YaRN rope_scale={CFG.get('rope_scale') or 'native'})")
    log(f"#  KV CACHE : {CFG['cache_type_k']}/{CFG['cache_type_v']} (FlashAttention on, rotation on)")
    log(f"#  MTP DRAFT: {CFG.get('mtp_n_max')} tokens speculative decoding")
    log(f"#  HEADROOM : {vram_desc}")
    log("#" * 70)
    publish("ready", endpoint=(f"{url}/v1" if url else None), api_key=CFG["api_key"],
            model=CFG["served_model_name"], ctx_size=CFG["ctx_size"],
            kv=CFG["cache_type_k"], mtp=CFG.get("mtp_n_max"),
            rope_scale=CFG.get("rope_scale"), vram_headroom=vram_stats)

    banner("6", "Self-Test & Benchmark", "Verifying MTP speculative decoding speedup")
    self_test()

    t_serve = time.time()
    while time.time() - t_serve < CFG["keepalive_min"] * 60:
        time.sleep(120)
        if server.poll() is not None:
            server_died(server, "stopped", reason="server-exit")
        up = int((time.time() - t_serve) / 60)
        if up % 10 < 2:
            publish("heartbeat", up_min=up, endpoint=(f"{url}/v1" if url else None))
            log(f"   serving live ({up} min) - {url + '/v1' if url else ''}")
    publish("auto-shutdown", served_min=CFG["keepalive_min"])
    server.terminate()
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
