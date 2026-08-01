# Task 1 — Llama-Guard-3-8B benchmark

Baseline zero-shot [`meta-llama/Llama-Guard-3-8B`](https://huggingface.co/meta-llama/Llama-Guard-3-8B)
cho **Task 1: Multi-Label Unsafe Text Recognition**.

Code: `src/LlamaGuard/` · Script: `scripts/benchmark_task1_llamaguard.sh`
Kết quả: `experiment_results/task1/llamaguard/`

Cấu trúc giống hệt [ShieldGemma](task1-shieldhgemma.md) — cùng 3 case input,
cùng schema output, cùng cơ chế checkpoint — nên hai baseline so sánh trực
tiếp được.

---

## 0. TL;DR

```bash
cd /Users/macbookpro/Documents/darren/Thesis/githubs_repo/GuardChat

python3 -m venv .venv-llamaguard                        # venv riêng cho thí nghiệm này
source .venv-llamaguard/bin/activate
pip install -r src/LlamaGuard/requirements.txt

echo 'HF_TOKEN=hf_xxxxxxxxxxxxxxxx' >> .env             # token HuggingFace của bạn

bash scripts/benchmark_task1_llamaguard.sh all
```

Lần chạy đầu tự tải weights (~16 GB) vào
`src/LlamaGuard/weights/Llama-Guard-3-8B/`.

| Case | Input | File |
|---|---|---|
| 1a | `prompt` (enhanced prompt, $X_{single}$) | `llamaguard_task1_prompt.json` |
| 1b | `raw_prompt` (prompt gốc chưa enhance) | `llamaguard_task1_raw_prompt.json` |
| 2 | `conversation` (multi-turn, $X_{conv}$) | `llamaguard_task1_conversation.json` |

---

## 1. Môi trường

Dependency riêng ở **`src/LlamaGuard/requirements.txt`**:

```bash
python3 -m venv .venv-llamaguard
source .venv-llamaguard/bin/activate
pip install --upgrade pip
pip install -r src/LlamaGuard/requirements.txt

python -c "import torch, transformers; print(torch.__version__, transformers.__version__)"
```

Ghim `transformers>=4.43,<5`: 4.43 là bản đầu hỗ trợ Llama 3.1 (kiến trúc của
Llama-Guard-3) và có chat template nhận `categories=` để override taxonomy;
chặn 5.x vì nó đòi `torch>=2.4`, mà macOS x86_64 chỉ có tới torch 2.2.2 —
cài 5.x ở đó sẽ **tắt luôn backend PyTorch** và mọi lần load model đều lỗi
`AutoModelForCausalLM requires the PyTorch library`.

**Máy đã có sẵn env đúng thì không cần venv.** Ví dụ server RunPod đang dùng
có `torch 2.4.1+cu124` + `transformers 4.57.6` — nằm trong khoảng hợp lệ, dùng
thẳng system python được (xem mục 4).

## 2. Token & weights

`meta-llama/Llama-Guard-3-8B` là **gated repo** — phải xin quyền truy cập
(Meta licence) tại https://huggingface.co/meta-llama/Llama-Guard-3-8B bằng
chính tài khoản của token. Duyệt thường trong vài phút.

Thứ tự tìm token (`src/utils/hf_token.py`), lấy cái đầu tiên có:

1. `--token hf_...` trên CLI
2. biến môi trường `HF_TOKEN` / `HUGGINGFACE_TOKEN` / `HUGGING_FACE_HUB_TOKEN`
3. file `.env` ở gốc repo (đã trong `.gitignore`)
4. `huggingface-cli login` trước đó

Weights tải **vào trong repo**, không rải trong `~/.cache/huggingface`:

```
src/LlamaGuard/weights/Llama-Guard-3-8B/     (~16 GB, đã trong .gitignore)
```

Đây là mặc định của `--weights`; lần chạy đầu tự tải nếu thư mục trống. Tách
riêng bước tải:

```bash
python -m src.LlamaGuard.download_weights
```

Mặc định bỏ qua thư mục `original/` của Meta (bản sao `.pth` của cùng bộ
weights, tiết kiệm ~16 GB đĩa); thêm `--include-original` nếu thực sự cần.

Tuỳ chọn khác: `LLAMAGUARD_WEIGHTS=/đường/dẫn/khác`, `--no-auto-download`
(báo lỗi thay vì tự tải), hoặc `--weights meta-llama/Llama-Guard-3-8B` để
dùng HF cache như cũ.

## 3. Vấn đề lệch số class (14 vs 6)

Llama-Guard-3 được fine-tune trên taxonomy MLCommons **S1–S14** của Meta,
trong khi GuardChat có **6** category. Hai chế độ (`--mode`):

**`guardchat` (mặc định)** — đẩy thẳng 6 category của GuardChat vào chat
template (S1=Sexual, S2=Illegal, S3=Shocking, S4=Violence, S5=Self-harm,
S6=Harassment). Model suy luận zero-shot trên đúng schema của ta, map 1–1, và
`shocking` bắn được. Đây là cấu hình nên báo cáo trong bảng Task 1.

**`native`** — dùng S1–S14 gốc rồi map ngược (lossy):

| S-code | GuardChat |
|---|---|
| S1 Violent Crimes, S9 Indiscriminate Weapons | violence |
| S2, S6, S7, S8, S13, S14 | illegal |
| S3 Sex-Related Crimes, S4 CSE, S12 Sexual Content | sexual |
| S5 Defamation, S10 Hate | harassment |
| S11 Suicide & Self-Harm | self-harm |
| *(không có)* | **shocking → không bao giờ bắn** |

Ở mode này `shocking` luôn F1 = 0 nên Macro-F1 bị chặn trên; CLI in cảnh báo
`unreachable_categories` để không nhầm với bug. Đây vẫn là con số zero-shot
trung thực của model "as released" — hợp cho phần thảo luận.

`--mode custom` là alias cũ của `guardchat`, vẫn chạy được.

Chi tiết mapping: `src/LlamaGuard/taxonomy.py`.

## 3b. `conversation` được đưa vào model thế nào

Khác ShieldGemma (model một-input), Llama-Guard vốn được thiết kế cho hội
thoại nhiều lượt. Có 2 lựa chọn (`--conv-format`):

- **`turns` (mặc định)** — đẩy nguyên danh sách turn vào chat template. Model
  đánh giá lượt cuối trong ngữ cảnh các lượt trước; đúng phân phối huấn
  luyện của nó, thường là con số mạnh nhất.
- **`concat`** — gộp toàn bộ hội thoại thành **một** user message, giống cách
  BiLSTM / BERT / ShieldGemma nhìn $X_{conv}$.

Nếu bảng Task 1 cần input **đồng nhất tuyệt đối** giữa các dòng thì chạy thêm
một lượt `--conv-format concat`. Trường `conv_format` được ghi trong `meta`
của file output nên không lẫn được.

## 4. Chạy

### Cách 1 — script (khuyến nghị)

```bash
bash scripts/benchmark_task1_llamaguard.sh all          # cả 3 case, 1 lần load model
bash scripts/benchmark_task1_llamaguard.sh prompt       # chỉ case 1a
bash scripts/benchmark_task1_llamaguard.sh prompt raw_prompt
```

| Biến | Mặc định | Ý nghĩa |
|---|---|---|
| `HF_TOKEN` | — | token HuggingFace (hoặc để trong `.env`) |
| `LLAMAGUARD_TEST` | `build_dataset/dataset/final_df_test.json` | file test |
| `LLAMAGUARD_WEIGHTS` | `src/LlamaGuard/weights/Llama-Guard-3-8B` | thư mục weights (hoặc Hub id) |
| `LLAMAGUARD_MODE` | `guardchat` | `guardchat` \| `native` |
| `LLAMAGUARD_CONV_FORMAT` | `turns` | `turns` \| `concat` |
| `LLAMAGUARD_OUT` | `experiment_results/task1/llamaguard` | thư mục output |
| `DEVICE` | `auto` | `auto` \| `cuda` \| `mps` \| `cpu` |
| `DTYPE_LG` | `auto` | `auto` \| `bfloat16` \| `float16` \| `float32` \| `int8` \| `nf4` |
| `LIMIT` | — | giới hạn số sample (smoke test) |
| `RESUME` | — | `RESUME=1` để chạy tiếp từ checkpoint |

Smoke test trước khi chạy full:

```bash
LIMIT=10 bash scripts/benchmark_task1_llamaguard.sh all
```

### Cách 2 — gọi module trực tiếp

```bash
python -m src.LlamaGuard.eval_recognition \
    --test build_dataset/dataset/final_df_test.json \
    --weights src/LlamaGuard/weights/Llama-Guard-3-8B \
    --mode guardchat \
    --text-kind all \
    --conv-format turns \
    --device auto --dtype auto \
    --output-dir experiment_results/task1/llamaguard
```

### Chạy trên server RunPod (`ada5000-runpod`)

RTX A5000 24 GB, repo tại `/root/darren_ws/guard/GuardChat`:

- **Không cần venv** — system python 3.11 đã có torch 2.4.1 + transformers 4.57.6.
- **Weights phải để trên `/workspace`**: ổ `/` chỉ còn ~2 GB mà snapshot nặng
  ~16 GB. Symlink như đã làm với ShieldGemma:

  ```bash
  mkdir -p /workspace/darren/models
  ln -s /workspace/darren/models/Llama-Guard-3-8B \
        /root/darren_ws/guard/GuardChat/src/LlamaGuard/weights/Llama-Guard-3-8B
  ```

- **`PYTHON=python3`** vì server không có alias `python`:

  ```bash
  PYTHON=python3 bash scripts/benchmark_task1_llamaguard.sh all
  ```

Bộ nhớ: 8B ở bf16 chiếm ~16 GB — vừa 24 GB của A5000. Nếu GPU nhỏ hơn thì
`DTYPE_LG=nf4` (~5 GB) hoặc `int8` (~9 GB), cần `bitsandbytes`.

Llama-Guard sinh ~20 token mỗi input (khác ShieldGemma chỉ 1 forward pass), và
mỗi sample chỉ cần **1** lần gọi model chứ không phải 6 — nên tổng thời gian
hai baseline không chênh nhau nhiều như số tham số gợi ý. Cứ chạy `LIMIT=10`
trước để đo.

## 5. Định dạng output

```jsonc
{
  "prompt": {                       // key = text kind của file
    "metrics": {
      "f1_sexual": 0.0, "f1_illegal": 0.0, "f1_shocking": 0.0,
      "f1_violence": 0.0, "f1_self-harm": 0.0, "f1_harassment": 0.0,
      "macro_f1": 0.0, "recall_binary": 0.0, "asr": 0.0
    },
    "predictions": [
      {
        "sample_id": "0",
        "text_kind": "prompt",
        "text": "...",                       // đúng chuỗi model đã nhìn thấy
        "scodes": ["S3", "S4"],              // verdict THÔ — dùng để map lại
        "raw_response": "\n\nunsafe\nS3,S4",
        "multi_label": { "sexual": 0, "illegal": 0, "shocking": 1,
                         "violence": 1, "self-harm": 0, "harassment": 0 },
        "predicted_categories": ["shocking", "violence"],
        "binary_pred": 1,
        "label_vector_true": { "...": 0 },
        "true_categories": ["harassment"],
        "gold_category": "harassment"
      }
    ]
  },
  "meta": {
    "model": "src/LlamaGuard/weights/Llama-Guard-3-8B", "mode": "guardchat",
    "dtype": "bfloat16", "device": "cuda",
    "text_kind": "prompt", "num_samples": 1000,
    "conv_format": "turns", "role_prefix": true,
    "unreachable_categories": []
  }
}
```

Ghi chú:

- Khác ShieldGemma, **không có `policy_scores`**: Llama-Guard trả S-code rời
  rạc chứ không phải xác suất từng category. Thứ đóng vai trò tương đương là
  `scodes` + `raw_response` — đủ để đổi cách map S-code → category offline.
- `metrics` dùng `src/utils/metrics.py`: Macro-F1 không trọng số trên 6
  category, `recall_binary`, `asr = 1 − recall`. Test set chỉ chứa mẫu unsafe
  đã oracle-xác nhận nên `asr` chính là tỉ lệ prompt lọt lưới.
- `skipped_empty_input: true` khi input rỗng — mẫu đó coi là safe mà không gọi
  model, có cờ riêng để loại khi tính metric.
- Ground truth là **một** category / sample nên `label_vector_true` thực chất
  là one-hot. Cùng cảnh báo như ở ShieldGemma: enhanced prompt thường chứa
  nhiều loại harm thật, Macro-F1 sẽ phạt những dự đoán thêm đó như false
  positive.

## 6. Checkpoint & chạy tiếp

Mỗi sample ghi ngay vào `<output>.partial.jsonl` (flush từng dòng):

```bash
RESUME=1 bash scripts/benchmark_task1_llamaguard.sh all
# hoặc: python -m src.LlamaGuard.eval_recognition ... --resume
```

Các `sample_id` đã có sẽ được bỏ qua. File `.partial.jsonl` tự xoá sau khi ghi
JSON cuối; giữ lại bằng `--keep-checkpoint`.

## 7. Cấu trúc code

```
src/LlamaGuard/
├── requirements.txt     dependency riêng của thí nghiệm này
├── taxonomy.py          S1-S14 ↔ 6 category GuardChat, + taxonomy custom
├── model.py             load Llama-Guard-3-8B, chat template + generate
├── recognition.py       RecognitionPipeline + 3 text kind + conv_format
├── eval_recognition.py  CLI, checkpoint/resume, ghi 3 file JSON
├── download_weights.py  tải snapshot về weights/ (tuỳ chọn, chạy trước)
├── weights/             snapshot Llama-Guard-3-8B (git-ignored)
└── configs/recognition.yaml
```

Phần dùng chung với ShieldGemma nằm ở `src/utils/`:

| File | Nội dung |
|---|---|
| `hf_token.py` | tìm HF token: CLI > env > `.env` > HF cache |
| `hf_model.py` | phân biệt Hub id / thư mục, tải snapshot vào `weights/`, chọn device & dtype |
| `data.py` | `TEXT_KINDS`, `text_for_kind`, `gold_category` — 3 cách biểu diễn input |
| `task1_eval.py` | vòng eval + checkpoint/resume + ghi file, dùng chung cho cả hai CLI |
| `metrics.py` | Macro-F1 / Recall / ASR |
