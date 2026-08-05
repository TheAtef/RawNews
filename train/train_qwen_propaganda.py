"""
train_qwen_propaganda.py
=========================
Fine-tunes Qwen/Qwen3.5-0.8B (causal LM) as a *generative classifier* for
Arabic propaganda / bias detection, using the QCRI/ArmPro dataset.

Task formulation
-----------------
Qwen3.5-0.8B is a causal LM, not an encoder classifier, so we cast the task
as instruction tuning:

    <|im_start|>system ... <|im_end|>
    <|im_start|>user
    <Arabic paragraph + instruction to analyze it>
    <|im_end|>
    <|im_start|>assistant
    Label: <Technique Name or "Neutral">
    <|im_end|>

We train with TRL's SFTTrainer + PEFT LoRA, masking the prompt so loss is
only computed on the assistant's completion (label span).

Usage
-----
    python train_qwen_propaganda.py

Output
------
    ./train/models/fine_tuned_qwen_propaganda/
        - LoRA adapter weights
        - tokenizer files
"""

from __future__ import annotations

import collections
import collections.abc


collections.Mapping = collections.abc.Mapping
collections.MutableMapping = collections.abc.MutableMapping
collections.Sequence = collections.abc.Sequence
collections.MutableSequence = collections.abc.MutableSequence
collections.MutableSet = collections.abc.MutableSet

import gc
import os
import random
from pathlib import Path
from typing import Optional


os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import structlog
import torch
from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset
from peft import LoraConfig, TaskType, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    set_seed,
)
from trl import SFTConfig, SFTTrainer
import re
import time

structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.add_log_level,
        structlog.dev.ConsoleRenderer(),
    ]
)
log = structlog.get_logger("train_qwen_propaganda")

MODEL_ID = "Qwen/Qwen3.5-0.8B"
DATASET_ID = "QCRI/ArmPro"
DATASET_SUBSET = "multilabel" 


MERGE_BINARY_SUBSET = False
BINARY_SUBSET_NAME = "binary"


OUTPUT_DIR = "./train/models/fine_tuned_qwen_propaganda"
SEED = 42


LOW_VRAM_MODE = True

if LOW_VRAM_MODE:
    MAX_SEQ_LEN = 512

    TRAIN_BATCH_SIZE = 2
    EVAL_BATCH_SIZE = 2
    GRAD_ACCUM_STEPS = 8  
else:
    MAX_SEQ_LEN = 1024
    TRAIN_BATCH_SIZE = 4
    EVAL_BATCH_SIZE = 4
    GRAD_ACCUM_STEPS = 4

TEXT_COLUMN_CANDIDATES = ["paragraph", "text", "content"]
LABEL_COLUMN_CANDIDATES = ["labels", "technique", "techniques", "fine_labels", "label"]

NO_TECHNIQUE_TOKEN = "no technique"
NEUTRAL_OUTPUT = "Neutral"


REBALANCE_TRAIN_SET = True

TARGET_PROPAGANDA_RATIO = 0.40


MAX_TRAIN_HOURS = 7.5


ENABLE_ACCURACY_PROBE_DURING_TRAINING = False
ACCURACY_CHECK_EVERY_N_STEPS = 300
ACCURACY_PROBE_SIZE = 60  
BEST_CHECKPOINT_DIR = OUTPUT_DIR + "_best_binary_acc"

SYSTEM_PROMPT = (
    "أنت محلل إعلامي متخصص في كشف الدعاية والتحيز في النصوص الإخبارية العربية. "
    "مهمتك هي تحليل الفقرة المعطاة وتحديد ما إذا كانت تحتوي على أسلوب دعائي أو متحيز، "
    "وإذا كانت كذلك، تحديد نوع الأسلوب الدعائي المستخدم بدقة.\n\n"
    "يجب أن تكون إجابتك دائما بهذا الشكل بالضبط، بالإنجليزية حرفيا، بدون أي نص إضافي:\n\n"
    "مثال 1:\n"
    "Verdict: Neutral\n\n"
    "مثال 2:\n"
    "Verdict: Propaganda\n"
    "Technique: Loaded_Language"
)

USER_INSTRUCTION_TEMPLATE = (
    "حلل الفقرة الإخبارية التالية وحدد ما إذا كانت تحتوي على أساليب دعائية "
    "(مثل: التحميل اللغوي، التسمية/التشهير، المبالغة أو التهوين، الاستقطاب، "
    "مناشدة السلطة، التشكيك، التبسيط السببي، وغيرها).\n\n"
    "أعد الإجابة بالضبط بهذا التنسيق:\n"
    "Verdict: Propaganda أو Verdict: Neutral\n"
    "إذا كان الحكم Propaganda، أضف سطرا ثانيا: Technique: <نوع الأسلوب الدعائي>\n\n"
    "الفقرة:\n{paragraph}"
)



def _detect_column(dataset: Dataset, candidates: list[str], role: str) -> str:
    columns = set(dataset.column_names)
    for candidate in candidates:
        if candidate in columns:
            log.info("column_detected", role=role, column=candidate)
            return candidate
    raise ValueError(
        f"Could not detect a {role} column in dataset. "
        f"Available columns: {sorted(columns)}. "
        f"Expected one of: {candidates}"
    )


def _normalize_label(raw_label) -> str:

    if raw_label is None:
        return NEUTRAL_OUTPUT

    if isinstance(raw_label, (list, tuple)):
        techniques = []
        for item in raw_label:
            if isinstance(item, dict):
                name = item.get("technique") or item.get("label") or item.get("name")
            else:
                name = item
            if name and str(name).strip().lower() != NO_TECHNIQUE_TOKEN:
                techniques.append(str(name).strip())
        if not techniques:
            return NEUTRAL_OUTPUT
        # seen = []
        # for t in techniques:
        #     if t not in seen:
        #         seen.append(t)
        # return " | ".join(seen)
        return techniques[0]  # just take the first technique for simplicity

    label_str = str(raw_label).strip()
    if label_str.lower() in {NO_TECHNIQUE_TOKEN, "false", "none", ""}:
        return NEUTRAL_OUTPUT
    if label_str.lower() == "true":
        return "Propaganda"
    return label_str


def _load_binary_subset_as_verdict_only(tokenizer: AutoTokenizer) -> Optional[Dataset]:
    try:
        binary_raw = load_dataset(DATASET_ID, BINARY_SUBSET_NAME)
    except Exception as exc:
        log.warning("binary_subset_load_failed_skipping_merge", error=str(exc))
        return None

    if "train" not in binary_raw:
        log.warning("binary_subset_has_no_train_split_skipping_merge")
        return None

    ds = binary_raw["train"]
    try:
        bin_text_col = _detect_column(ds, TEXT_COLUMN_CANDIDATES, role="binary-subset paragraph text")
        bin_label_col = _detect_column(ds, LABEL_COLUMN_CANDIDATES, role="binary-subset label")
    except ValueError as exc:
        log.warning("binary_subset_schema_mismatch_skipping_merge", error=str(exc))
        return None

    ds = ds.filter(
        lambda ex: bool(ex.get(bin_text_col)) and len(str(ex[bin_text_col]).strip()) >= 10
    )

    def to_conversation_verdict_only(example: dict) -> dict:
        paragraph = str(example.get(bin_text_col) or "").strip()
        verdict = _normalize_label(example.get(bin_label_col))
        is_propaganda = verdict.strip().lower() != NEUTRAL_OUTPUT.lower()
        assistant_msg = "Verdict: Propaganda" if is_propaganda else "Verdict: Neutral"
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_INSTRUCTION_TEMPLATE.format(paragraph=paragraph)},
            {"role": "assistant", "content": assistant_msg},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        return {"text": text, "_paragraph_len": len(paragraph), "_is_propaganda": is_propaganda}

    ds = ds.map(
        to_conversation_verdict_only,
        remove_columns=[c for c in ds.column_names if c not in ("text", "_paragraph_len", "_is_propaganda")],
        desc="Formatting ArmPro[binary] as verdict-only rows",
    )
    log.info("binary_subset_merged", rows=len(ds))
    return ds


def load_and_prepare_armpro(tokenizer: AutoTokenizer) -> tuple[DatasetDict, list[dict]]:

    log.info("dataset_loading_start", dataset=DATASET_ID, subset=DATASET_SUBSET)
    try:
        raw = load_dataset(DATASET_ID, DATASET_SUBSET)
    except Exception as exc: 
        log.warning(
            "subset_load_failed_falling_back_to_default",
            subset=DATASET_SUBSET,
            error=str(exc),
        )
        raw = load_dataset(DATASET_ID)

    if "train" not in raw:
        raise ValueError(f"Expected a 'train' split, got splits: {list(raw.keys())}")

    schema_probe = raw["train"]
    text_col = _detect_column(schema_probe, TEXT_COLUMN_CANDIDATES, role="paragraph text")
    label_col = _detect_column(schema_probe, LABEL_COLUMN_CANDIDATES, role="technique label")

    probe_split_name = next(
        (s for s in ("dev", "validation", "test") if s in raw), "train"
    )

    def to_conversation(example: dict) -> dict:
        paragraph = (example.get(text_col) or "").strip()
        technique = _normalize_label(example.get(label_col))
        is_propaganda = technique.strip().lower() != NEUTRAL_OUTPUT.lower()

        user_msg = USER_INSTRUCTION_TEMPLATE.format(paragraph=paragraph)

        if is_propaganda:
            assistant_msg = f"Verdict: Propaganda\nTechnique: {technique}"
        else:
            assistant_msg = "Verdict: Neutral"

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": assistant_msg},
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        return {"text": text, "_paragraph_len": len(paragraph), "_is_propaganda": is_propaganda}

    def is_valid(example: dict) -> bool:
        paragraph = example.get(text_col)
        if not paragraph or not str(paragraph).strip():
            return False
        if len(str(paragraph).strip()) < 10:
            return False
        return True

    prepared = DatasetDict()
    accuracy_probe_examples: list[dict] = []

    for split_name, split_ds in raw.items():
        before = len(split_ds)
        split_ds = split_ds.filter(is_valid)
        after_filter = len(split_ds)

        if split_name == probe_split_name:
            accuracy_probe_examples = _build_stratified_probe_set(
                split_ds, text_col, label_col, ACCURACY_PROBE_SIZE
            )
            log.info(
                "accuracy_probe_set_built",
                source_split=probe_split_name,
                probe_size=len(accuracy_probe_examples),
            )

        split_ds = split_ds.map(
            to_conversation,
            remove_columns=[c for c in split_ds.column_names if c != "text"],
            desc=f"Formatting ArmPro[{split_name}] into ChatML",
        )

        if split_name == "train" and MERGE_BINARY_SUBSET:
            binary_extra = _load_binary_subset_as_verdict_only(tokenizer)
            if binary_extra is not None:
                before_merge = len(split_ds)
                split_ds = concatenate_datasets([split_ds, binary_extra])
                log.info(
                    "train_split_merged_with_binary_subset",
                    multilabel_rows=before_merge,
                    binary_rows=len(binary_extra),
                    combined_rows=len(split_ds),
                )

        if split_name == "train" and REBALANCE_TRAIN_SET:
            split_ds = _oversample_propaganda_rows(split_ds, target_ratio=TARGET_PROPAGANDA_RATIO)

        split_ds = split_ds.remove_columns(
            [c for c in split_ds.column_names if c not in ("text",)]
        )
        prepared[split_name] = split_ds
        log.info(
            "split_prepared",
            split=split_name,
            rows_before=before,
            rows_after_filter=after_filter,
            rows_final=len(split_ds),
        )

    if not accuracy_probe_examples:
        log.warning("accuracy_probe_set_empty_falling_back_to_no_accuracy_checkpointing")

    return prepared, accuracy_probe_examples


def _build_stratified_probe_set(
    dataset: Dataset, text_col: str, label_col: str, target_size: int
) -> list[dict]:
    rng = random.Random(SEED)
    prop_rows, neutral_rows = [], []
    for row in dataset:
        paragraph = str(row.get(text_col) or "").strip()
        if not paragraph:
            continue
        technique = _normalize_label(row.get(label_col))
        is_prop = technique.strip().lower() != NEUTRAL_OUTPUT.lower()
        (prop_rows if is_prop else neutral_rows).append(paragraph)

    rng.shuffle(prop_rows)
    rng.shuffle(neutral_rows)
    half = target_size // 2
    probe = [{"paragraph": p, "is_propaganda": True} for p in prop_rows[:half]]
    probe += [{"paragraph": p, "is_propaganda": False} for p in neutral_rows[:half]]
    rng.shuffle(probe)
    return probe


def _oversample_propaganda_rows(dataset: Dataset, target_ratio: float) -> Dataset:

    is_prop_flags = dataset["_is_propaganda"]
    n_total = len(dataset)
    n_prop = sum(is_prop_flags)
    n_neutral = n_total - n_prop

    if n_prop == 0 or n_neutral == 0:
        log.warning("oversampling_skipped_degenerate_split", n_prop=n_prop, n_neutral=n_neutral)
        return dataset

    # n_prop_new / (n_prop_new + n_neutral) == target_ratio
    n_prop_target = int((target_ratio * n_neutral) / (1 - target_ratio))
    n_extra_copies_needed = max(0, n_prop_target - n_prop)

    if n_extra_copies_needed == 0:
        log.info("oversampling_not_needed", current_ratio=round(n_prop / n_total, 3))
        return dataset

    prop_indices = [i for i, flag in enumerate(is_prop_flags) if flag]
    prop_subset = dataset.select(prop_indices)

    full_repeats = n_extra_copies_needed // len(prop_subset)
    remainder = n_extra_copies_needed % len(prop_subset)

    pieces = [dataset] + [prop_subset] * full_repeats
    if remainder > 0:
        pieces.append(prop_subset.select(range(remainder)))

    balanced = concatenate_datasets(pieces).shuffle(seed=SEED)
    new_ratio = (n_prop + n_extra_copies_needed) / len(balanced)
    log.info(
        "oversampling_applied",
        original_ratio=round(n_prop / n_total, 3),
        new_ratio=round(new_ratio, 3),
        original_size=n_total,
        new_size=len(balanced),
    )
    return balanced

def load_model_and_tokenizer():
    log.info("model_loading_start", model=MODEL_ID, low_vram_mode=LOW_VRAM_MODE)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token


    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    quantization_config = None
    # if LOW_VRAM_MODE and torch.cuda.is_available():
    #     quantization_config = BitsAndBytesConfig(
    #         load_in_4bit=True,
    #         bnb_4bit_quant_type="nf4",
    #         bnb_4bit_compute_dtype=torch.bfloat16,
    #         bnb_4bit_use_double_quant=True,
    #     )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        quantization_config=quantization_config,
        device_map={"": device}, 
        trust_remote_code=True,
    )
    model.config.use_cache = False

    if quantization_config is not None:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=True
        )

    log.info(
        "model_loading_complete",
        device=device,
        quantized_4bit=quantization_config is not None,
    )
    return model, tokenizer


def build_lora_config() -> LoraConfig:
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=24,
        lora_alpha=48,
        lora_dropout=0.05,
        bias="none",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )


class TimeBudgetCallback(TrainerCallback):

    def __init__(self, max_hours: float):
        self.max_seconds = max_hours * 3600
        self.start_time: Optional[float] = None

    def on_train_begin(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        self.start_time = time.time()
        log.info("time_budget_active", max_hours=self.max_seconds / 3600)

    def on_step_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        elapsed = time.time() - self.start_time
        if elapsed >= self.max_seconds:
            log.warning(
                "time_budget_exceeded_stopping_training",
                elapsed_hours=round(elapsed / 3600, 2),
                step=state.global_step,
            )
            control.should_training_stop = True
        return control


_VERDICT_PATTERN = re.compile(r"verdict\s*:\s*(\w+)", re.IGNORECASE)


def _quick_predict_is_propaganda(generated_text: str) -> bool:
    match = _VERDICT_PATTERN.search(generated_text)
    raw = (match.group(1) if match else generated_text).strip().split("\n")[0].strip()
    return raw.lower() not in {NEUTRAL_OUTPUT.lower(), "", "none"}


class BinaryAccuracyCheckpointCallback(TrainerCallback):

    def __init__(self, tokenizer, probe_examples: list[dict], every_n_steps: int, best_dir: str):
        self.tokenizer = tokenizer
        self.probe_examples = probe_examples
        self.every_n_steps = every_n_steps
        self.best_dir = best_dir
        self.best_accuracy = -1.0

    def on_step_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        if not self.probe_examples:
            return control
        if state.global_step == 0 or state.global_step % self.every_n_steps != 0:
            return control

        model = kwargs["model"]
        accuracy = self._evaluate_probe_set(model)
        log.info(
            "binary_accuracy_probe_check",
            step=state.global_step,
            accuracy=round(accuracy, 4),
            best_so_far=round(self.best_accuracy, 4),
        )

        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
            log.info("new_best_checkpoint_saving", accuracy=round(accuracy, 4), path=self.best_dir)
            model.save_pretrained(self.best_dir)
            self.tokenizer.save_pretrained(self.best_dir)

        return control

    @torch.no_grad()
    def _evaluate_probe_set(self, model) -> float:
        was_training = model.training
        model.eval()
        prev_use_cache = getattr(model.config, "use_cache", None)
        model.config.use_cache = True 
        prev_padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left" 

        correct = 0
        batch_size = 8
        try:
            for i in range(0, len(self.probe_examples), batch_size):
                batch = self.probe_examples[i : i + batch_size]
                prompts = []
                for ex in batch:
                    messages = [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {
                            "role": "user",
                            "content": USER_INSTRUCTION_TEMPLATE.format(paragraph=ex["paragraph"]),
                        },
                    ]
                    prompts.append(
                        self.tokenizer.apply_chat_template(
                            messages, tokenize=False, add_generation_prompt=True
                        )
                    )
                inputs = self.tokenizer(
                    prompts, return_tensors="pt", padding=True, truncation=True, max_length=MAX_SEQ_LEN
                ).to(model.device)

                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=20,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
                for j, ex in enumerate(batch):
                    input_len = inputs["input_ids"][j].shape[0]
                    decoded = self.tokenizer.decode(
                        output_ids[j][input_len:], skip_special_tokens=True
                    )
                    predicted_is_prop = _quick_predict_is_propaganda(decoded)
                    if predicted_is_prop == ex["is_propaganda"]:
                        correct += 1
        finally:
            model.config.use_cache = prev_use_cache
            self.tokenizer.padding_side = prev_padding_side
            if was_training:
                model.train()

        return correct / len(self.probe_examples)


def train() -> None:
    set_seed(SEED)
    random.seed(SEED)

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    model, tokenizer = load_model_and_tokenizer()
    dataset, accuracy_probe_examples = load_and_prepare_armpro(tokenizer)

    train_dataset = dataset.get("train")
    eval_dataset = dataset.get("dev") or dataset.get("validation") or dataset.get("test")

    if train_dataset is None or len(train_dataset) == 0:
        raise RuntimeError("Training split is empty after filtering — aborting.")

    log.info(
        "dataset_ready",
        train_size=len(train_dataset),
        eval_size=len(eval_dataset) if eval_dataset else 0,
        accuracy_probe_size=len(accuracy_probe_examples),
    )
    log.info("sample_formatted_example", example=train_dataset[0]["text"][:600])

    lora_config = build_lora_config()


    if eval_dataset is not None and LOW_VRAM_MODE and len(eval_dataset) > 200:
        eval_dataset = eval_dataset.select(range(200))
        log.info("eval_dataset_capped_for_low_vram", size=len(eval_dataset))

    sft_config = SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        weight_decay=0.01,
        max_grad_norm=0.3,
        logging_steps=10,
   
        eval_strategy="epoch" if eval_dataset is not None else "no",
        save_strategy="epoch",
        load_best_model_at_end=eval_dataset is not None,
        metric_for_best_model="eval_loss" if eval_dataset is not None else None,
        greater_is_better=False,
        save_total_limit=1,
        bf16=True,

        optim="adamw_bnb_8bit",
        dataloader_num_workers=0,  
        report_to="none",
        seed=SEED,
        max_length=MAX_SEQ_LEN,
        packing=False,
        dataset_text_field="text",
        completion_only_loss=True,  
    )

    callbacks = [TimeBudgetCallback(max_hours=MAX_TRAIN_HOURS)]
    accuracy_callback = None
    if ENABLE_ACCURACY_PROBE_DURING_TRAINING and accuracy_probe_examples:
        accuracy_callback = BinaryAccuracyCheckpointCallback(
            tokenizer=tokenizer,
            probe_examples=accuracy_probe_examples,
            every_n_steps=ACCURACY_CHECK_EVERY_N_STEPS,
            best_dir=BEST_CHECKPOINT_DIR,
        )
        callbacks.append(accuracy_callback)
    elif not ENABLE_ACCURACY_PROBE_DURING_TRAINING:
        log.info(
            "accuracy_probe_disabled",
            note="Using eval_loss + load_best_model_at_end instead. "
            "Run test_qwen_propaganda.py after training for the real accuracy number.",
        )
    else:
        log.warning("no_accuracy_probe_set_best_checkpoint_selection_disabled")

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=lora_config,
        processing_class=tokenizer,
        callbacks=callbacks,
    )

    log.info("training_start", max_hours_budget=MAX_TRAIN_HOURS)
    trainer.train()
    log.info("training_complete")

    log.info("saving_model", output_dir=OUTPUT_DIR)
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    log.info("save_complete", output_dir=OUTPUT_DIR)

    if accuracy_callback is not None and accuracy_callback.best_accuracy >= 0:
        log.info(
            "recommendation",
            message=(
                "Two checkpoints were saved. Evaluate BOTH with "
                "test_qwen_propaganda.py and use whichever scores higher — "
                "the best-by-accuracy checkpoint is often better than the "
                "final step, since loss and binary accuracy don't always "
                "move together."
            ),
            final_checkpoint=OUTPUT_DIR,
            best_by_accuracy_checkpoint=BEST_CHECKPOINT_DIR,
            best_probe_accuracy_seen_during_training=round(accuracy_callback.best_accuracy, 4),
        )

    del trainer, model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    log.info("memory_cleanup_complete")


if __name__ == "__main__":
    try:
        train()
    except Exception:
        log.exception("training_failed")
        raise
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()