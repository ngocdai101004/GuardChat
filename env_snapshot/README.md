# Server environment snapshot

Captured **2026-08-02** from the vast.ai GPU instance used to run the
GuardChat benchmarks (`/workspace/darren_ws/guard/GuardChat/.venv`).

Enough to rebuild the environment from scratch. Read §3 before running
`pip install -r requirements-lock.txt` — a plain install of that file
**will fail**.

---

## 1. What is in this folder

| File | Contents |
|------|----------|
| `requirements-lock.txt` | `pip freeze --all` — 93 packages, exact versions |
| `pip-list-verbose.txt` | same, plus install location and installer per package |
| `python-env.txt` | Python / pip versions, `pyvenv.cfg`, pip config |
| `torch-env.txt` | torch + CUDA + cuDNN + GPU, and the versions of the libraries this repo actually imports |
| `system.txt` | `nvidia-smi`, OS release, `nvcc`, disk |

---

## 2. The environment

| | |
|---|---|
| OS | Ubuntu 24.04.4 LTS, kernel 5.15.0-181 |
| GPU | NVIDIA GeForce RTX 5090, 31.4 GiB, **sm_120** |
| Driver | 580.159.03 (CUDA 13.0) |
| `nvcc` | 13.0.88 |
| Python | 3.12.13 (conda-forge build, from the vast.ai base image) |
| pip | 26.2 |
| torch | **2.12.0+cu130**, `torch.version.cuda` 13.0, cuDNN 9.20 |
| transformers | 4.57.6 |
| accelerate | 1.14.0 |
| huggingface_hub | 0.36.2 (+ `hf-xet` 1.5.0, `hf_transfer` 0.1.9) |
| tokenizers | 0.22.2 |
| safetensors | 0.8.0 |
| numpy | 2.4.6 |

The RTX 5090 is **sm_120** (Blackwell). Older torch builds have no
kernels for it and fall back to a "no kernel image is available" error,
so the CUDA-13 wheels above are not optional — they are the floor.

---

## 3. Rebuilding — four things that will bite

### 3.1 `torch` is not on PyPI at this version

`torch==2.12.0+cu130`, `torchvision==0.27.0+cu130` and
`torchcodec==0.12.0+cu130` are local-version wheels published only on
PyTorch's own index. `pip install -r requirements-lock.txt` alone
resolves them against PyPI, finds nothing, and fails. Install them
first, from the cu130 index:

```bash
pip install torch==2.12.0 torchvision==0.27.0 torchcodec==0.12.0 \
    --index-url https://download.pytorch.org/whl/cu130
```

The `nvidia-*` packages in the lock file (cublas, cudnn, nccl, …) are
torch's own dependencies and come along with it — do not pin them by
hand.

### 3.2 `packaging` was built by conda, not pip

```
packaging @ file:///home/conda/feedstock_root/build_artifacts/...
```

That path exists only inside the image it was built in. Replace the line
with a plain `packaging` (any recent version works) before installing,
or the install dies on a missing local file.

### 3.3 The venv inherits the base image

```
command = /venv/main/bin/python -m venv --system-site-packages \
          /workspace/darren_ws/guard/GuardChat/.venv
include-system-site-packages = true
```

`.venv` was created **with** `--system-site-packages` on top of the
vast.ai image's `/venv/main`. So `pip freeze --all` lists everything
visible — including packages that live in the base image, not in
`.venv`. On a fresh machine you install all 93 yourself; nothing is
inherited. `pip-list-verbose.txt` shows which came from where.

That is also why `torch` shows up at all despite never being installed
into `.venv` directly.

### 3.4 Three things this repo can want are **not** installed

| Package | Needed for | Consequence |
|---------|-----------|-------------|
| `bitsandbytes` | `--dtype int8` / `--dtype nf4` | quantised loading raises at import. Relevant if the Llama Task-2 run OOMs — the fallback is unavailable until it is installed |
| `datasets` | `load_guardchat()` with an HF repo id | fine as long as `--test` points at a local JSON, which every script here does by default |
| `google-genai` | the Gemini Task-2 rewriter | Gemini was run from the laptop, not the server; install it if you want to run that baseline here |

Install what you need:

```bash
pip install bitsandbytes>=0.43     # CUDA only
pip install datasets>=2.18
pip install "google-genai>=1.0"
```

---

## 4. Full recreate

```bash
python3.12 -m venv .venv
.venv/bin/pip install -U pip

# 1. torch from the CUDA-13 index (see 3.1)
.venv/bin/pip install torch==2.12.0 torchvision==0.27.0 torchcodec==0.12.0 \
    --index-url https://download.pytorch.org/whl/cu130

# 2. fix the conda-built line (see 3.2), then the rest
sed 's|^packaging @ .*|packaging|' env_snapshot/requirements-lock.txt \
    > /tmp/req.txt
.venv/bin/pip install -r /tmp/req.txt

# 3. verify
.venv/bin/python -c "
import torch, transformers
print(torch.__version__, torch.version.cuda, torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-')
print('transformers', transformers.__version__)"
```

Expected: `2.12.0+cu130 13.0 True` / `NVIDIA GeForce RTX 5090` /
`transformers 4.57.6`.

---

## 5. Regenerating this snapshot

The commands that produced these files are in the session that created
this folder; the short version:

```bash
cd /workspace/darren_ws/guard/GuardChat
mkdir -p env_snapshot
.venv/bin/pip freeze --all > env_snapshot/requirements-lock.txt
.venv/bin/pip list -v      > env_snapshot/pip-list-verbose.txt
```

plus `torch-env.txt` / `system.txt` from `nvidia-smi`, `nvcc --version`
and a short torch probe.
