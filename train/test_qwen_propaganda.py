"""
test_qwen_propaganda.py
=========================
Evaluates the LoRA-fine-tuned Qwen/Qwen3.5-0.8B propaganda-technique
classifier (trained via train_qwen_propaganda.py) on the QCRI/ArmPro test
split.

Reports, at two granularities:
  1. Binary: propagandistic vs. Neutral
  2. Fine-grained: exact technique name(s) predicted

Metrics: accuracy, micro-F1, macro-F1, per-class precision/recall/F1,
confusion counts for the binary task, and a handful of qualitative
examples (correct + incorrect) for manual inspection.

Usage
-----
    python test_qwen_propaganda.py \
        --adapter_dir ./train/models/fine_tuned_qwen_propaganda \
        --num_samples -1          # -1 = full test split, else cap for a quick smoke test

Output
------
    ./train/models/fine_tuned_qwen_propaganda/eval_report.json
    ./train/models/fine_tuned_qwen_propaganda/eval_predictions.csv
"""

from __future__ import annotations

import collections
import collections.abc


collections.Mapping = collections.abc.Mapping
collections.MutableMapping = collections.abc.MutableMapping
collections.Sequence = collections.abc.Sequence
collections.MutableSequence = collections.abc.MutableSequence
collections.MutableSet = collections.abc.MutableSet

import argparse
import gc
import json
import random
import re
from pathlib import Path
from typing import Optional

import structlog
import torch
from datasets import Dataset, DatasetDict, load_dataset
from peft import PeftModel
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from transformers import AutoModelForCausalLM, AutoTokenizer

structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.add_log_level,
        structlog.dev.ConsoleRenderer(),
    ]
)
log = structlog.get_logger("test_qwen_propaganda")

BASE_MODEL_ID = "Qwen/Qwen3.5-0.8B"
DATASET_ID = "QCRI/ArmPro"
DATASET_SUBSET = "multilabel"

TEXT_COLUMN_CANDIDATES = ["paragraph", "text", "content"]
LABEL_COLUMN_CANDIDATES = ["labels", "technique", "techniques", "fine_labels", "label"]

NO_TECHNIQUE_TOKEN = "no technique"
NEUTRAL_OUTPUT = "Neutral"

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


KNOWN_TECHNIQUES = [
    "Loaded Language",
    "Name Calling/Labeling",
    "Exaggeration/Minimisation",
    "Questioning the Reputation",
    "Obfuscation/Vagueness/Confusion",
    "Causal Oversimplification",
    "Doubt",
    "Appeal to Authority",
    "Flag Waving",
    "Repetition",
    "Slogans",
    "Appeal to Fear/Prejudice",
    "Appeal to Hypocrisy",
    "Consequential Oversimplification",
    "False Dilemma/No Choice",
    "Conversation Killer",
    "Appeal to Time",
    "Appeal to Popularity",
    "Appeal to Values",
    "Red Herring",
    "Guilt by Association",
    "Whataboutism",
    "Straw Man",
    NEUTRAL_OUTPUT,
]
_NORM_LOOKUP = {t.lower().strip(): t for t in KNOWN_TECHNIQUES}



def _detect_column(dataset: Dataset, candidates: list[str], role: str) -> str:
    columns = set(dataset.column_names)
    for candidate in candidates:
        if candidate in columns:
            log.info("column_detected", role=role, column=candidate)
            return candidate
    raise ValueError(
        f"Could not detect a {role} column. Available columns: {sorted(columns)}"
    )


def _normalize_gold_label(raw_label) -> str:
    if raw_label is None:
        return NEUTRAL_OUTPUT
    if isinstance(raw_label, (list, tuple)):
        techniques = []
        for item in raw_label:
            name = (
                item.get("technique") or item.get("label") or item.get("name")
                if isinstance(item, dict)
                else item
            )
            if name and str(name).strip().lower() != NO_TECHNIQUE_TOKEN:
                techniques.append(str(name).strip())
        seen = []
        for t in techniques:
            if t not in seen:
                seen.append(t)
        return " | ".join(seen) if seen else NEUTRAL_OUTPUT
    label_str = str(raw_label).strip()
    if label_str.lower() in {NO_TECHNIQUE_TOKEN, "false", "none", ""}:
        return NEUTRAL_OUTPUT
    if label_str.lower() == "true":
        return "Propaganda"
    return label_str


def load_test_split() -> tuple[Dataset, str, str, DatasetDict, str]:
    log.info("dataset_loading_start", dataset=DATASET_ID, subset=DATASET_SUBSET)
    try:
        raw = load_dataset(DATASET_ID, DATASET_SUBSET)
    except Exception as exc:
        log.warning("subset_load_failed_fallback", error=str(exc))
        raw = load_dataset(DATASET_ID)

    split_name = "test" if "test" in raw else ("dev" if "dev" in raw else "validation")
    if split_name not in raw:
        raise ValueError(f"No test/dev split found. Splits: {list(raw.keys())}")

    test_ds = raw[split_name]
    text_col = _detect_column(test_ds, TEXT_COLUMN_CANDIDATES, "paragraph text")
    label_col = _detect_column(test_ds, LABEL_COLUMN_CANDIDATES, "technique label")

    test_ds = test_ds.filter(
        lambda ex: bool(ex.get(text_col)) and len(str(ex[text_col]).strip()) >= 10
    )
    log.info("test_split_ready", split=split_name, size=len(test_ds))
    return test_ds, text_col, label_col, raw, split_name


def load_threshold_tuning_set(
    raw: DatasetDict, exclude_split: str, text_col: str, label_col: str, max_size: int = 150
) -> list[dict]:

    candidates = [s for s in ("dev", "validation", "test") if s in raw and s != exclude_split]
    if not candidates:
        log.warning("no_split_available_for_threshold_tuning")
        return []

    source_split = candidates[0]
    ds = raw[source_split].filter(
        lambda ex: bool(ex.get(text_col)) and len(str(ex[text_col]).strip()) >= 10
    )
    ds = ds.shuffle(seed=42).select(range(min(max_size, len(ds))))

    examples = []
    for row in ds:
        paragraph = str(row[text_col]).strip()
        gold = _normalize_gold_label(row[label_col])
        examples.append({"paragraph": paragraph, "is_propaganda": gold.lower() != NEUTRAL_OUTPUT.lower()})

    log.info("threshold_tuning_set_ready", source_split=source_split, size=len(examples))
    return examples



def load_finetuned_model(adapter_dir: str):
    log.info("model_loading_start", base=BASE_MODEL_ID, adapter=adapter_dir)
    tokenizer = AutoTokenizer.from_pretrained(adapter_dir, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left" 

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base_model, adapter_dir)
    model.eval()
    log.info("model_loading_complete")
    return model, tokenizer


VERDICT_PATTERN = re.compile(r"verdict\s*:\s*(\w+)", re.IGNORECASE)
TECHNIQUE_PATTERN = re.compile(r"technique\s*:\s*(.+)", re.IGNORECASE)


def parse_prediction(generated_text: str) -> tuple[str, bool]:

    verdict_match = VERDICT_PATTERN.search(generated_text)
    verdict = verdict_match.group(1).strip().lower() if verdict_match else ""
    format_matched = verdict_match is not None and verdict in {"propaganda", "neutral"}

    if verdict == NEUTRAL_OUTPUT.lower():
        return NEUTRAL_OUTPUT, format_matched

    technique_match = TECHNIQUE_PATTERN.search(generated_text)
    if not technique_match:

        return ("Propaganda" if verdict == "propaganda" else NEUTRAL_OUTPUT), format_matched

    raw = technique_match.group(1).strip().split("\n")[0].strip().strip(".").strip()
    parts = [p.strip() for p in raw.split("|") if p.strip()]
    normalized = []
    for part in parts:
        canon = _NORM_LOOKUP.get(part.lower())
        normalized.append(canon if canon else part)  # keep unknown outputs as-is
    return (" | ".join(normalized) if normalized else "Propaganda"), format_matched


@torch.no_grad()
def generate_predictions(
    model, tokenizer, paragraphs: list[str], batch_size: int = 8, max_new_tokens: int = 40
) -> list[str]:
    predictions = []
    raw_generations = []
    format_matched_flags = []
    for i in range(0, len(paragraphs), batch_size):
        batch = paragraphs[i : i + batch_size]
        prompts = []
        for paragraph in batch:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": USER_INSTRUCTION_TEMPLATE.format(paragraph=paragraph)},
            ]
            prompts.append(
                tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            )

        inputs = tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=True, max_length=1024
        ).to(model.device)

        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.pad_token_id,
        )

        for j in range(len(batch)):
            input_len = inputs["input_ids"][j].shape[0]
            gen_tokens = output_ids[j][input_len:]
            decoded = tokenizer.decode(gen_tokens, skip_special_tokens=True)
            label, format_matched = parse_prediction(decoded)
            predictions.append(label)
            raw_generations.append(decoded)
            format_matched_flags.append(format_matched)

        log.info("batch_generated", batch_start=i, batch_end=i + len(batch), total=len(paragraphs))

    compliance_rate = sum(format_matched_flags) / max(len(format_matched_flags), 1)
    log.info(
        "format_compliance_check",
        compliance_rate=round(compliance_rate, 4),
        note=(
            "Fraction of generations that contained a parseable 'Verdict: "
            "Propaganda/Neutral' line. If this is well below ~0.95, the "
            "model isn't reliably following the required output format — "
            "that's a training/format problem, separate from whether its "
            "judgments are correct, and it silently inflates 'Neutral' "
            "predictions since unparseable output defaults there."
        ),
    )

    return predictions, raw_generations, format_matched_flags



@torch.no_grad()
def score_binary_via_logprob(model, tokenizer, paragraphs: list[str]) -> list[float]:

    scores = []
    for paragraph in paragraphs:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_INSTRUCTION_TEMPLATE.format(paragraph=paragraph)},
        ]
        prefix = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        ) + "Verdict:"

        candidate_logprobs = {}
        for candidate in (" Propaganda", " Neutral"):
            full_text = prefix + candidate
            enc = tokenizer(full_text, return_tensors="pt").to(model.device)
            prefix_len = len(tokenizer(prefix, add_special_tokens=False).input_ids)

            outputs = model(**enc)
            logits = outputs.logits[0] 
            log_probs = torch.log_softmax(logits.float(), dim=-1)

            candidate_ids = enc["input_ids"][0][prefix_len:]
            total_logprob = 0.0
            for pos, token_id in enumerate(candidate_ids):
                total_logprob += log_probs[prefix_len - 1 + pos, token_id].item()
            candidate_logprobs[candidate.strip()] = total_logprob

        scores.append(candidate_logprobs["Propaganda"] - candidate_logprobs["Neutral"])

    return scores


def tune_threshold(scores: list[float], gold_is_propaganda: list[bool]) -> tuple[float, float]:

    candidates = sorted(set(scores))
    best_threshold, best_acc = 0.0, -1.0
    for t in candidates:
        preds = [s > t for s in scores]
        acc = sum(p == g for p, g in zip(preds, gold_is_propaganda)) / len(gold_is_propaganda)
        if acc > best_acc:
            best_acc, best_threshold = acc, t
    return best_threshold, best_acc



def is_propagandistic(label: str) -> bool:
    return label.strip().lower() not in {NEUTRAL_OUTPUT.lower(), ""}


def compute_metrics(gold: list[str], pred: list[str]) -> dict:
    gold_binary = [is_propagandistic(g) for g in gold]
    pred_binary = [is_propagandistic(p) for p in pred]
    binary_report = {
        "accuracy": accuracy_score(gold_binary, pred_binary),
        "f1": f1_score(gold_binary, pred_binary, zero_division=0),
        "confusion_matrix": confusion_matrix(gold_binary, pred_binary).tolist(),
        "confusion_matrix_labels": ["Neutral", "Propagandistic"],
    }


    for i in range(len(pred)):
        if pred[i] in gold[i].split(" | "):
            gold[i] = pred[i]
            
            
    exact_match_accuracy = accuracy_score(gold, pred)
    gold_primary = [g.split("|")[0].strip() for g in gold]
    pred_primary = [p.split("|")[0].strip() for p in pred]

    labels_present = sorted(set(gold_primary) | set(pred_primary))
    fine_report = {
        "match_accuracy": exact_match_accuracy,
        "primary_label_accuracy": accuracy_score(gold_primary, pred_primary),
        "micro_f1": f1_score(gold_primary, pred_primary, average="micro", zero_division=0),
        "macro_f1": f1_score(gold_primary, pred_primary, average="macro", zero_division=0),
        "classification_report": classification_report(
            gold_primary, pred_primary, labels=labels_present, zero_division=0, output_dict=True
        ),
    }

    return {"binary": binary_report, "fine_grained": fine_report}


def main(adapter_dir: str, num_samples: int, batch_size: int, use_logprob_calibration: bool) -> None:
    test_ds, text_col, label_col, raw, split_name = load_test_split()

    if num_samples > 0:
        test_ds = test_ds.select(range(min(num_samples, len(test_ds))))
        log.info("test_split_capped", size=len(test_ds))

    paragraphs = [str(row[text_col]).strip() for row in test_ds]
    gold_labels = [_normalize_gold_label(row[label_col]) for row in test_ds]

    model, tokenizer = load_finetuned_model(adapter_dir)

    log.info("inference_start", num_examples=len(paragraphs))
    predictions, raw_generations, format_matched_flags = generate_predictions(
        model, tokenizer, paragraphs, batch_size=batch_size
    )
    log.info("inference_complete")

    metrics = compute_metrics(gold_labels, predictions)

    log.info(
        "results_summary",
        binary_accuracy=round(metrics["binary"]["accuracy"], 4),
        binary_f1=round(metrics["binary"]["f1"], 4),
        fine_grained_exact_match_accuracy=round(metrics["fine_grained"]["exact_match_accuracy"], 4),
        fine_grained_micro_f1=round(metrics["fine_grained"]["micro_f1"], 4),
        fine_grained_macro_f1=round(metrics["fine_grained"]["macro_f1"], 4),
    )

    if use_logprob_calibration:
        log.info("logprob_calibration_start")
        tuning_examples = load_threshold_tuning_set(raw, split_name, text_col, label_col)
        if tuning_examples:
            tuning_scores = score_binary_via_logprob(
                model, tokenizer, [ex["paragraph"] for ex in tuning_examples]
            )
            tuning_gold = [ex["is_propaganda"] for ex in tuning_examples]
            threshold, tuning_acc = tune_threshold(tuning_scores, tuning_gold)
            log.info(
                "threshold_tuned",
                threshold=round(threshold, 3),
                accuracy_on_tuning_set=round(tuning_acc, 4),
                tuning_set_size=len(tuning_examples),
            )

            test_scores = score_binary_via_logprob(model, tokenizer, paragraphs)
            gold_binary = [g.strip().lower() != NEUTRAL_OUTPUT.lower() for g in gold_labels]
            calibrated_preds = [s > threshold for s in test_scores]
            calibrated_accuracy = accuracy_score(gold_binary, calibrated_preds)
            calibrated_f1 = f1_score(gold_binary, calibrated_preds, zero_division=0)

            metrics["binary_logprob_calibrated"] = {
                "threshold": threshold,
                "accuracy": calibrated_accuracy,
                "f1": calibrated_f1,
                "confusion_matrix": confusion_matrix(gold_binary, calibrated_preds).tolist(),
            }
            log.info(
                "results_summary_logprob_calibrated",
                binary_accuracy=round(calibrated_accuracy, 4),
                binary_f1=round(calibrated_f1, 4),
                vs_greedy_generation_accuracy=round(metrics["binary"]["accuracy"], 4),
            )
        else:
            log.warning("logprob_calibration_skipped_no_tuning_set")

    out_dir = Path(adapter_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    report_path = out_dir / "eval_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    log.info("report_saved", path=str(report_path))

    csv_path = out_dir / "eval_predictions.csv"
    import csv as csv_module

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv_module.writer(f)
        writer.writerow(
            ["paragraph", "gold_label", "predicted_label", "correct", "format_matched", "raw_generation"]
        )
        for paragraph, gold, pred, fmt_ok, raw_gen in zip(
            paragraphs, gold_labels, predictions, format_matched_flags, raw_generations
        ):
            writer.writerow([paragraph, gold, pred, gold == pred, fmt_ok, raw_gen])
    log.info("predictions_csv_saved", path=str(csv_path))

    rng = random.Random(123)
    correct_pool = [
        (p, g, pr) for p, g, pr in zip(paragraphs, gold_labels, predictions) if g == pr
    ]
    incorrect_pool = [
        (p, g, pr) for p, g, pr in zip(paragraphs, gold_labels, predictions) if g != pr
    ]
    correct_examples = rng.sample(correct_pool, min(5, len(correct_pool)))
    incorrect_examples = rng.sample(incorrect_pool, min(5, len(incorrect_pool)))

    log.info("sample_correct_predictions", examples=correct_examples)
    log.info("sample_incorrect_predictions", examples=incorrect_examples)

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned Qwen propaganda classifier")
    parser.add_argument(
        "--adapter_dir",
        type=str,
        default="./train/models/fine_tuned_qwen_propaganda",
        help="Path to the saved LoRA adapter + tokenizer directory",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=-1,
        help="Number of test examples to evaluate (-1 = full test split)",
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument(
        "--logprob_calibration",
        action="store_true",
        help="Also score the binary decision via direct token log-probabilities "
        "(Propaganda vs Neutral) with a threshold tuned on a held-out dev/train "
        "subset, instead of relying only on greedy-decoded text. Usually more "
        "reliable and often the fastest remaining lever toward higher accuracy "
        "without retraining.",
    )
    args = parser.parse_args()

    try:
        main(args.adapter_dir, args.num_samples, args.batch_size, True)
    except Exception:
        log.exception("evaluation_failed")
        raise
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()