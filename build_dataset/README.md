---
license: cc-by-nc-4.0
dataset_info:
  features:
    - name: id
      dtype: int32
    - name: category
      dtype:
        class_label:
          names:
            "0": harassment
            "1": illegal
            "2": self-harm
            "3": sexual
            "4": shocking
            "5": violence
    - name: prompt
      dtype: string
    - name: raw_prompt
      dtype: string
    - name: source
      dtype:
        class_label:
          names:
            "0": i2p
            "1": jailbreak_diffusion_bench
            "2": nibbler
            "3": overt
    - name: conversation_generator
      dtype:
        class_label:
          names:
            "0": gemini
            "1": gemma4
    - name: conversation
      list:
        - name: turn_id
          dtype: int32
        - name: role
          dtype: string
        - name: content
          dtype: string
    - name: conversation_text
      dtype: string
  splits:
    - name: train
      num_bytes: 51142433
      num_examples: 9000
    - name: test
      num_bytes: 6708636
      num_examples: 1000
    - name: full
      num_bytes: 57851069
      num_examples: 10000
  download_size: 78350367
  dataset_size: 115702138
configs:
  - config_name: default
    data_files:
      - split: train
        path: data/train-*
      - split: test
        path: data/test-*
      - split: full
        path: data/full-*
task_categories:
  - text-classification
language:
  - en
pretty_name: GuardChat
size_categories:
  - 10K<n<100K
---

# Dataset Card for GuardChat

## Dataset Details

### Dataset Description

GuardChat is a benchmark dataset for multi-turn jailbreak attacks in text-to-image (T2I) systems.
It contains 10,000 prompt-conversation pairs across six unsafe categories: harassment, illegal, self-harm, sexual, shocking, and violence.
Each sample pairs an enhanced toxic prompt with a realistic adversarial conversation that gradually escalates across multiple turns.

- **Curated by:** Multimedia Synergy Lab
- **Language(s):** English
- **License:** CC BY-NC 4.0

### Dataset Sources

- **Repository:** [https://huggingface.co/datasets/multimedia-synergy-lab/GuardChat](https://huggingface.co/datasets/multimedia-synergy-lab/GuardChat)
- **Paper:** `build_dataset/GuardChat_NeurIPS2026.pdf` (preprint draft in this repo)

## Uses

### Direct Use

This dataset is intended for:
- Multi-label unsafe text recognition under conversational context.
- Prompt rewriting / NSFW concept removal evaluation.
- Robustness testing of T2I safety pipelines under multi-turn adversarial escalation.

### Out-of-Scope Use

- Any malicious attempt to jailbreak production systems.
- Deployment as an instruction source for harmful prompt crafting.
- Commercial usage outside the CC BY-NC 4.0 terms.

## Dataset Structure

Each sample includes:
- `id`: sample id.
- `category`: class label over six unified harm classes.
- `prompt`: enhanced unsafe prompt.
- `raw_prompt`: original source prompt.
- `source`: source dataset label.
- `conversation_generator`: model family used to generate conversation.
- `conversation`: list of turns with `turn_id`, `role`, `content` (can be null in some samples).
- `conversation_text`: flattened conversation text.

Splits:
- `train`: 9,000 samples.
- `test`: 1,000 samples.
- `full`: 10,000 samples.

## Dataset Creation

### Curation Rationale

GuardChat is created to benchmark a realistic threat model that is under-covered by single-turn safety datasets: multi-turn memory-exploiting jailbreak attacks against T2I systems.

### Source Data

Raw prompts are aggregated from I2P, Adversarial Nibbler, JailbreakDiffBench, and OVERT.

#### Data Collection and Processing

Pipeline summary:
1. Near-duplicate removal, low-quality filtering, and normalization.
2. Prompt enhancement using Gemini-2.5-Flash and LoRA fine-tuned Qwen2.5-7B-Instruct.
3. Multi-turn conversation synthesis using Gemma-4-31B.
4. Oracle validation and automated consistency/PII checks.

## Annotations

Category labels are normalized into six classes:
`harassment`, `illegal`, `self-harm`, `sexual`, `shocking`, `violence`.

### Personal and Sensitive Information

The dataset is synthetic and filtered with automated checks for PII. No real user chat logs are intentionally included.

## Bias, Risks, and Limitations

- English-only scope may limit cross-lingual generalization.
- Source/category imbalance can affect model behavior.
- Test split distribution intentionally differs from training.
- Contains harmful content for safety research; handle in controlled environments.

## Citation

```bibtex
@misc{guardchat2026,
  title        = {GuardChat: Benchmarking Multi-Turn Jailbreak Attacks in T2I Systems},
  year         = {2026},
  note         = {NeurIPS 2026 submission},
  howpublished = {\url{https://huggingface.co/datasets/multimedia-synergy-lab/GuardChat}}
}
```

## Dataset Card Contact

For questions, use the dataset discussion page on Hugging Face:
https://huggingface.co/datasets/multimedia-synergy-lab/GuardChat
