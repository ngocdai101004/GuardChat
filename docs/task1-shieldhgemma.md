# Task 1 — ShieldGemma-2B benchmark

Baseline zero-shot [`google/shieldgemma-2b`](https://huggingface.co/google/shieldgemma-2b)
cho **Task 1: Multi-Label Unsafe Text Recognition**.

Code: `src/ShieldGemma/` · Script: `scripts/benchmark_task1_shieldgemma.sh`
Kết quả: `experiment_results/task1/shieldgemma/`

---

## 0. TL;DR

```bash
cd /Users/macbookpro/Documents/darren/Thesis/githubs_repo/GuardChat

python3 -m venv .venv-shieldgemma                        # venv riêng cho thí nghiệm này
source .venv-shieldgemma/bin/activate
pip install -r src/ShieldGemma/requirements.txt

echo 'HF_TOKEN=hf_xxxxxxxxxxxxxxxx' >> .env              # token HuggingFace của bạn

bash scripts/benchmark_task1_shieldgemma.sh all
```

Lần chạy đầu sẽ tự tải weights (~5 GB) vào
`src/ShieldGemma/weights/shieldgemma-2b/`.

Sinh ra 3 file:

| Case | Input | File |
|---|---|---|
| 1a | `prompt` (enhanced prompt, $X_{single}$) | `shieldgemma_task1_prompt.json` |
| 1b | `raw_prompt` (prompt gốc chưa enhance) | `shieldgemma_task1_raw_prompt.json` |
| 2 | `conversation` (nối các turn, $X_{conv}$) | `shieldgemma_task1_conversation.json` |

Metric (Macro-F1 / ASR) được tính sẵn ở mức threshold mặc định, nhưng **mỗi
prediction đều lưu điểm thô `P(Yes)` của từng policy** nên có thể tính lại
metric với threshold hoặc cách map class khác mà không cần chạy lại model.

---

## 1. Môi trường — tạo venv riêng

Dependency của thí nghiệm này nằm trong **`src/ShieldGemma/requirements.txt`**,
tách khỏi `requirements.txt` chung của repo. Nên dựng một venv riêng để
không đụng vào `thesisEnv` (lý do ở ngay dưới):

```bash
cd /Users/macbookpro/Documents/darren/Thesis/githubs_repo/GuardChat

python3 -m venv .venv-shieldgemma
source .venv-shieldgemma/bin/activate     # Windows: .venv-shieldgemma\Scripts\activate
pip install --upgrade pip
pip install -r src/ShieldGemma/requirements.txt

python -c "import torch, transformers; print(torch.__version__, transformers.__version__)"
```

Mọi lệnh ở các mục sau đều chạy trong venv này (`source .venv-shieldgemma/bin/activate`).
Thoát bằng `deactivate`.

> **Nếu muốn dùng conda thay vì venv:**
> ```bash
> conda create -n shieldgemmaEnv python=3.11 -y
> conda activate shieldgemmaEnv
> pip install -r src/ShieldGemma/requirements.txt
> ```

### Vì sao không dùng thẳng `thesisEnv`

`thesisEnv` hiện có `transformers 5.7.0` + `torch 2.2.2`. transformers 5.x
yêu cầu `torch >= 2.4`, nên nó **tự tắt backend PyTorch**:

```
[transformers] Disabling PyTorch because PyTorch >= 2.4 is required but found 2.2.2
ImportError: AutoModelForCausalLM requires the PyTorch library but it was not found
```

Máy này là macOS **x86_64**, mà PyTorch đã dừng build wheel cho macOS x86 sau
bản 2.2.2 — không nâng torch lên được. Vì vậy `requirements.txt` của module
ghim `transformers>=4.42,<5` (4.42 là bản đầu tiên hỗ trợ kiến trúc Gemma-2;
4.57.1 là tổ hợp đã kiểm chứng chạy được với torch 2.2.2).

Nếu vẫn muốn chạy trong `thesisEnv`, chỉ cần hạ transformers — nhưng việc này
ảnh hưởng tới các thí nghiệm khác đang dùng chung env đó:

```bash
conda activate thesisEnv
pip install "transformers>=4.42,<5"
```

> Trên server GPU NVIDIA thì không vướng gì: cài torch wheel CUDA trước, rồi
> `pip install -r src/ShieldGemma/requirements.txt`.

## 2. Token HuggingFace

`google/shieldgemma-2b` là **gated repo** — phải bấm đồng ý Gemma licence tại
https://huggingface.co/google/shieldgemma-2b bằng chính tài khoản của token.

Thứ tự tìm token (`src/ShieldGemma/hf_token.py`), lấy cái đầu tiên có:

1. `--token hf_...` trên CLI
2. biến môi trường `HF_TOKEN` / `HUGGINGFACE_TOKEN` / `HUGGING_FACE_HUB_TOKEN`
3. file `.env` ở gốc repo (đã nằm trong `.gitignore`)
4. `huggingface-cli login` trước đó

Cách tiện nhất để "cập nhật sau" là dùng `.env`:

```bash
echo 'HF_TOKEN=hf_xxxxxxxxxxxxxxxx' >> .env
```

## 2b. Weights nằm ở đâu

Weights được tải về **trong repo**, không để nằm rải rác trong
`~/.cache/huggingface`:

```
src/ShieldGemma/weights/shieldgemma-2b/     (~5 GB, đã có trong .gitignore)
```

Đây là giá trị mặc định của `--weights`. Lần chạy benchmark đầu tiên sẽ tự
tải nếu thư mục còn trống. Muốn tách bước tải ra riêng (mạng chậm):

```bash
python -m src.ShieldGemma.download_weights
```

Các tuỳ chọn liên quan:

- `SHIELDGEMMA_WEIGHTS=/đường/dẫn/khác` — đổi chỗ chứa weights.
- `--no-auto-download` — báo lỗi thay vì tự tải khi thư mục trống.
- `--weights google/shieldgemma-2b` — vẫn dùng được Hub id trực tiếp (khi đó
  file sẽ nằm trong HF cache như cũ).

Sau khi có snapshot, mọi lần chạy sau đều offline hoàn toàn.

## 3. Vấn đề lệch số class (4 vs 6)

ShieldGemma **không phải** multi-label classifier. Nó là một judge nhị phân
theo từng policy: đưa vào 1 guideline + 1 input, model trả `Yes`/`No`. Vector
6 chiều được dựng bằng cách chấm từng policy rồi threshold `P(Yes)`.

Policy set công bố của ShieldGemma chỉ có **4** mục, trong khi GuardChat có
**6** category. Module cho chọn 2 chế độ (`--mode`):

**`guardchat` (mặc định)** — 6 policy, mỗi policy ứng đúng 1 category
GuardChat, viết lại theo đúng văn phong policy của ShieldGemma. Map 1–1, số
class khớp hoàn toàn. Đây là cấu hình nên báo cáo trong bảng Task 1.

**`native`** — 4 policy nguyên bản của model card, rồi map ngược (lossy):

| ShieldGemma policy | GuardChat category |
|---|---|
| `dangerous_content` | violence, illegal, self-harm |
| `harassment` | harassment |
| `hate_speech` | harassment |
| `sexually_explicit` | sexual |
| *(không có)* | **shocking → không bao giờ bắn** |

Ở mode này `shocking` luôn F1 = 0 và `dangerous_content` bắn cùng lúc 3
category nên FP tăng — Macro-F1 bị chặn trên về mặt cấu trúc. CLI sẽ in cảnh
báo `unreachable_categories` để không nhầm với bug. Đây vẫn là con số
zero-shot trung thực của model "as released", có thể đưa vào phần thảo luận.

Chi tiết wording từng guideline: `src/ShieldGemma/taxonomy.py`.

## 4. Chạy

### Cách 1 — script (khuyến nghị)

```bash
bash scripts/benchmark_task1_shieldgemma.sh all          # cả 3 case, 1 lần load model
bash scripts/benchmark_task1_shieldgemma.sh prompt       # chỉ case 1a
bash scripts/benchmark_task1_shieldgemma.sh prompt raw_prompt
```

Biến môi trường có thể override:

| Biến | Mặc định | Ý nghĩa |
|---|---|---|
| `HF_TOKEN` | — | token HuggingFace (hoặc để trong `.env`) |
| `SHIELDGEMMA_TEST` | `build_dataset/dataset/final_df_test.json` | file test |
| `SHIELDGEMMA_WEIGHTS` | `src/ShieldGemma/weights/shieldgemma-2b` | thư mục weights (hoặc Hub id) |
| `SHIELDGEMMA_MODE` | `guardchat` | `guardchat` \| `native` |
| `SHIELDGEMMA_OUT` | `experiment_results/task1/shieldgemma` | thư mục output |
| `SHIELDGEMMA_THRESHOLD` | `0.5` | ngưỡng `P(Yes)` |
| `SHIELDGEMMA_BATCH` | `4` | số policy-prompt mỗi forward pass |
| `DEVICE` | `auto` | `auto` \| `cuda` \| `mps` \| `cpu` |
| `DTYPE_SG` | `auto` | `auto` \| `bfloat16` \| `float16` \| `float32` \| `int8` \| `nf4` |
| `LIMIT` | — | giới hạn số sample (smoke test) |
| `RESUME` | — | `RESUME=1` để chạy tiếp từ checkpoint |

Smoke test 10 sample trước khi chạy full:

```bash
LIMIT=10 bash scripts/benchmark_task1_shieldgemma.sh all
```

ShieldGemma cũng đã được gắn vào runner chung:
`bash scripts/benchmark_task1.sh shieldgemma`.

### Cách 2 — gọi module trực tiếp

```bash
python -m src.ShieldGemma.eval_recognition \
    --test build_dataset/dataset/final_df_test.json \
    --weights src/ShieldGemma/weights/shieldgemma-2b \
    --mode guardchat \
    --text-kind all \
    --threshold 0.5 \
    --batch-size 4 \
    --device auto --dtype auto \
    --output-dir experiment_results/task1/shieldgemma
```

Các flag đáng chú ý:

- `--text-kind {prompt,raw_prompt,conversation,all}` — `single` là alias của `prompt`.
- `--no-role-prefix` — nối turn **không** thêm tiền tố `user: `. Mặc định có
  prefix, khớp với các baseline Task 1 khác (`flatten_conversation`).
- `--resume` / `--keep-checkpoint` — xem mục 6.
- `--limit N` — cắt bớt dataset.

### Chi phí tính toán

Mode `guardchat` = 6 forward pass / sample / representation:
`1000 × 6 × 3 = 18.000` forward pass cho cả 3 case. Trên CPU/MPS của máy này
sẽ khá lâu (nhất là case `conversation`, input ~600–1000 token) — nên chạy
`LIMIT=10` trước để ước lượng, hoặc chạy trên GPU.

## 5. Định dạng output

Mỗi file giữ đúng schema chung với các baseline Task 1 khác:

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
        "policy_scores": {                   // P(Yes) THÔ — dùng để tính lại metric
          "sexual": 0.03, "illegal": 0.11, "shocking": 0.91,
          "violence": 0.88, "self-harm": 0.02, "harassment": 0.76
        },
        "policy_flags": { "shocking": 1, "violence": 1, "harassment": 1, "...": 0 },
        "max_score": 0.91,
        "threshold": 0.5,
        "multi_label": { "sexual": 0, "illegal": 0, "shocking": 1,
                         "violence": 1, "self-harm": 0, "harassment": 1 },
        "predicted_categories": ["shocking", "violence", "harassment"],
        "binary_pred": 1,
        "label_vector_true": { "...": 0 },
        "true_categories": ["harassment"],
        "gold_category": "harassment"
      }
    ]
  },
  "meta": {
    "model": "google/shieldgemma-2b", "mode": "guardchat",
    "policies": { "sexual": "...", "...": "..." },
    "threshold": 0.5, "dtype": "bfloat16", "device": "cuda",
    "text_kind": "prompt", "num_samples": 1000,
    "role_prefix": true, "unreachable_categories": []
  }
}
```

Ghi chú:

- `metrics` dùng `src/utils/metrics.py` — Macro-F1 trung bình không trọng số
  trên 6 category, `recall_binary`, và `asr = 1 − recall`. Vì test set chỉ
  chứa mẫu unsafe đã được oracle xác nhận, `asr` chính là tỉ lệ prompt lọt lưới.
- **`policy_scores` là thứ quan trọng nhất để tính metric sau này**: đổi
  threshold hay đổi cách map policy → category đều làm offline được, không
  cần chạy lại GPU.
- `skipped_empty_input: true` xuất hiện khi input rỗng (record thiếu
  `raw_prompt`); mẫu đó được coi là safe mà không chạy forward pass, và có cờ
  riêng để loại ra khi tính metric. Bộ test hiện tại đủ cả 3 field nên cờ này
  không xuất hiện.
- Ground truth trong `final_df_test.json` là **một** category / sample
  (`category` là string), nên `label_vector_true` thực chất là one-hot.

## 6. Checkpoint & chạy tiếp

Mỗi sample được ghi ngay vào `<output>.partial.jsonl` (flush từng dòng), nên
nếu process bị kill giữa chừng thì không mất gì:

```bash
RESUME=1 bash scripts/benchmark_task1_shieldgemma.sh all
# hoặc: python -m src.ShieldGemma.eval_recognition ... --resume
```

Những `sample_id` đã có trong checkpoint sẽ được bỏ qua. File `.partial.jsonl`
tự xoá sau khi ghi JSON cuối; giữ lại bằng `--keep-checkpoint`.

## 7. Cấu trúc code

```
src/ShieldGemma/
├── requirements.txt     dependency riêng của thí nghiệm này
├── taxonomy.py          4 policy native + 6 policy GuardChat, và mapping
├── model.py             load ShieldGemma-2B, chấm P(Yes) theo batch
├── recognition.py       RecognitionPipeline + 3 text kind
├── eval_recognition.py  CLI, checkpoint/resume, ghi 3 file JSON
├── download_weights.py  tải snapshot về weights/ (tuỳ chọn, chạy trước)
├── hf_token.py          tìm HF token: CLI > env > .env > HF cache
├── weights/             snapshot shieldgemma-2b (git-ignored)
└── configs/recognition.yaml
```

Điểm cần biết khi đọc `model.py`:

- Prompt template được **chép nguyên văn** từ model card (kể cả các dấu xuống
  dòng giữa câu). ShieldGemma được tune trên đúng chuỗi này; viết lại là điểm
  số lệch.
- Điểm số = `softmax` trên logits của 2 token `Yes` / `No` tại vị trí cuối,
  **một forward pass**, không sinh token → hoàn toàn deterministic.
- Tokenizer để `padding_side="left"` nên vị trí `-1` luôn là token thật cuối
  cùng của mọi dòng trong batch.
- `dtype=auto` → `float32` trên CPU, `bfloat16` trên GPU/MPS. Không mặc định
  `float16` vì Gemma-2 hay overflow ở fp16.
