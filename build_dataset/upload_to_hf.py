from huggingface_hub import login
import json
import os
import logging
import time
from datasets import Dataset, Features, Value, ClassLabel
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


HF_TOKEN = os.getenv("HF_TOKEN")
logger.info("Logging in to Hugging Face Hub...")
login(token=HF_TOKEN)
hf_path = "multimedia-synergy-lab/GuardChat"
full_path = "dataset/final_df.json"
train_path = "dataset/final_df_train.json"
test_path = "dataset/final_df_test.json"
logger.info("Target dataset repo: %s", hf_path)

# ====== FILES TO UPLOAD ======
split_to_path = {
    "train": train_path,
    "test": test_path,
    "full": full_path,
}


def flatten_conv(conv):
    if conv is None:
        return None
    return "\n".join([f"{t['role']}: {t['content']}" for t in conv])


# ====== LOAD ALL DATA FIRST (to infer labels globally) ======
all_records = []
raw_by_split = {}

for split, path in split_to_path.items():
    logger.info("Loading split '%s' from %s", split, path)
    with open(path, "r", encoding="utf-8") as f:
        split_data = json.load(f)
    raw_by_split[split] = split_data
    all_records.extend(split_data)
    logger.info("Loaded split '%s': %d samples", split, len(split_data))


# ====== INFER LABELS FROM ALL FILES ======
category_names = sorted({d["category"] for d in all_records})
source_names = sorted({d["source"] for d in all_records})
conversation_generator_names = sorted(
    {d["conversation_generator"] for d in all_records})

logger.info("Detected %d categories: %s", len(category_names), category_names)
logger.info("Detected %d sources: %s", len(source_names), source_names)
logger.info(
    "Detected %d conversation_generators: %s",
    len(conversation_generator_names),
    conversation_generator_names,
)


# ====== DEFINE FEATURES ======
features = Features({
    "id": Value("int32"),
    "category": ClassLabel(names=category_names),
    "prompt": Value("string"),
    "raw_prompt": Value("string"),
    "source": ClassLabel(names=source_names),
    "conversation_generator": ClassLabel(names=conversation_generator_names),
    # Use list-of-struct to match JSON shape: [{"turn_id": ..., "role": ..., "content": ...}, ...]
    # Keep role as string to avoid nested ClassLabel cast issues.
    "conversation": [{
        "turn_id": Value("int32"),
        "role": Value("string"),
        "content": Value("string"),
    }],
    "conversation_text": Value("string"),
})


# ====== BUILD + PUSH EACH SPLIT ======
total_start = time.perf_counter()
for split, split_data in raw_by_split.items():
    split_start = time.perf_counter()
    logger.info("Preparing split '%s' (%d samples)...", split, len(split_data))
    null_conversation_count = 0
    for d in split_data:
        if d.get("conversation") is None:
            null_conversation_count += 1
        d["conversation_text"] = flatten_conv(d["conversation"])
    if null_conversation_count:
        logger.warning(
            "Split '%s' has %d samples with null conversation (kept as null).",
            split,
            null_conversation_count,
        )

    logger.info("Casting features for split '%s'...", split)
    dataset_split = Dataset.from_list(split_data).cast(features)
    logger.info("Uploading split '%s' to Hub...", split)
    dataset_split.push_to_hub(
        hf_path,
        split=split,
        private=False,
    )
    split_elapsed = time.perf_counter() - split_start
    logger.info(
        "Uploaded split '%s' -> %s (%.2f seconds)",
        split,
        hf_path,
        split_elapsed,
    )

total_elapsed = time.perf_counter() - total_start
logger.info("All uploads completed in %.2f seconds.", total_elapsed)
