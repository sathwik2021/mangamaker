# finetune.py
#
# Fine-tunes Phi-3-mini-4k-instruct on the validated beat JSON pages produced
# by Step 1 (stored in STEP1_OUTPUTS_DIR/pages/).
#
# Training format (instruction-tuning):
#   <|system|>  …schema description…
#   <|user|>    EXCERPT: <chunk source text not stored — we use description>
#   <|assistant|> <beat JSON>
#
# Since Step 1 pages do NOT store the original source text, we reconstruct
# a minimal user prompt from the page metadata (page_id, characters list,
# emotional_flow summary) so the model learns the input → JSON mapping.
#
# Usage (Colab cell):
#   from finetune import FineTuner
#   ft = FineTuner()
#   ft.run()
#
# Or command line:
#   python finetune.py [--skip-dataset-build] [--skip-training]

import json
import logging
import math
import os
import random
import sys
from typing import Any, Dict, List, Optional, Tuple

import torch
from datasets import Dataset, DatasetDict
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

import config

# ── Logging ───────────────────────────────────────────────────────────────────

def _setup_logging() -> None:
    os.makedirs(config.FINETUNE_DIR, exist_ok=True)
    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(config.FINETUNE_LOG_FILE, mode="a", encoding="utf-8"),
    ]
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=handlers,
        force=True,
    )


logger = logging.getLogger(__name__)

# ── Prompt templates ──────────────────────────────────────────────────────────

_SYSTEM_PROMPT = (
    "You are a manga story analyst. Given a description of a scene, extract structured "
    "narrative beats as a JSON object. The JSON must contain: page_id, source_chunk_id, "
    "characters (list of names), beats (list of beat objects with id, order, type, characters, "
    "text, verb, emotion, intensity, causes, description), and emotional_flow (list of strings). "
    "Return ONLY the JSON object with no extra text."
)


def _build_prompt(page: Dict[str, Any]) -> str:
    """
    Build a user prompt from page metadata since source text is not stored.
    We describe the scene via its characters and emotional arc to give the
    model a meaningful input signal.
    """
    characters = ", ".join(page.get("characters", [])) or "unknown"
    emotional_flow = " → ".join(page.get("emotional_flow", [])) or "unspecified"
    beat_count = len(page.get("beats", []))
    return (
        f"Scene involves characters: {characters}. "
        f"Emotional arc: {emotional_flow}. "
        f"Extract {beat_count} narrative beats as structured JSON."
    )


def _format_training_example(page: Dict[str, Any], tokenizer: AutoTokenizer) -> str:
    """
    Format one page as a Phi-3 chat template string for training.
    The full conversation (system + user + assistant) is returned as a
    single string that the tokenizer will encode.
    """
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": _build_prompt(page)},
        {"role": "assistant", "content": json.dumps(page, ensure_ascii=False)},
    ]
    # apply_chat_template with tokenize=False gives us the raw string
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )


# ── Dataset builder ───────────────────────────────────────────────────────────

class DatasetBuilder:
    """
    Scans STEP1_OUTPUTS_DIR/pages/ for validated page JSON files,
    builds train/eval JSONL datasets, and saves them to Drive.
    """

    PAGES_DIR = os.path.join(config.STEP1_OUTPUTS_DIR, "pages")

    def __init__(self, tokenizer: AutoTokenizer):
        self.tokenizer = tokenizer

    def _load_all_pages(self) -> List[Dict[str, Any]]:
        """Load every page_*.json from the pages directory."""
        if not os.path.isdir(self.PAGES_DIR):
            raise FileNotFoundError(
                f"Pages directory not found: {self.PAGES_DIR}. "
                "Run multi_novel_pipeline.py first."
            )

        pages: List[Dict[str, Any]] = []
        filenames = sorted(f for f in os.listdir(self.PAGES_DIR) if f.endswith(".json"))

        if not filenames:
            raise ValueError(
                f"No JSON page files found in {self.PAGES_DIR}. "
                "Run multi_novel_pipeline.py first."
            )

        for fname in filenames:
            fpath = os.path.join(self.PAGES_DIR, fname)
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    page = json.load(f)
                pages.append(page)
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning("Skipping unreadable page file %s: %s", fpath, exc)

        logger.info("Loaded %d valid page files from %s", len(pages), self.PAGES_DIR)
        return pages

    def _tokenize_example(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Tokenize a formatted example.  Returns None if the sequence exceeds
        FINETUNE_MAX_SEQ_LENGTH (those examples are dropped to avoid padding issues).
        """
        encoded = self.tokenizer(
            text,
            truncation=False,
            padding=False,
            return_tensors=None,
        )
        if len(encoded["input_ids"]) > config.FINETUNE_MAX_SEQ_LENGTH:
            return None
        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "labels": list(encoded["input_ids"]),  # causal LM: labels = input_ids
        }

    def build(self) -> Tuple[int, int]:
        """
        Build train and eval JSONL files on Drive.

        Returns (train_count, eval_count).
        """
        pages = self._load_all_pages()

        random.seed(42)
        random.shuffle(pages)

        eval_size = max(1, math.ceil(len(pages) * config.FINETUNE_EVAL_SPLIT))
        eval_pages = pages[:eval_size]
        train_pages = pages[eval_size:]

        logger.info("Dataset split: %d train / %d eval", len(train_pages), len(eval_pages))

        train_count = self._write_jsonl(train_pages, config.FINETUNE_DATASET_TRAIN_FILE)
        eval_count = self._write_jsonl(eval_pages, config.FINETUNE_DATASET_EVAL_FILE)

        # Also write the full combined file for reference
        self._write_jsonl(pages, config.FINETUNE_DATASET_FILE)

        return train_count, eval_count

    def _write_jsonl(self, pages: List[Dict[str, Any]], filepath: str) -> int:
        """Format pages as training examples and write to a JSONL file."""
        written = 0
        skipped = 0
        with open(filepath, "w", encoding="utf-8") as f:
            for page in pages:
                try:
                    text = _format_training_example(page, self.tokenizer)
                    encoded = self._tokenize_example(text)
                    if encoded is None:
                        skipped += 1
                        logger.debug(
                            "Skipped page %s: exceeds max_seq_length %d",
                            page.get("page_id", "?"), config.FINETUNE_MAX_SEQ_LENGTH,
                        )
                        continue
                    # Write the raw text (HuggingFace datasets will tokenize from text)
                    record = {"text": text}
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    written += 1
                except Exception as exc:
                    logger.warning("Failed to format page %s: %s", page.get("page_id", "?"), exc)
                    skipped += 1

        logger.info("Wrote %d examples to %s (%d skipped)", written, filepath, skipped)
        return written

    def load_hf_datasets(self) -> DatasetDict:
        """Load the JSONL files into HuggingFace DatasetDict."""
        train_ds = Dataset.from_json(config.FINETUNE_DATASET_TRAIN_FILE)
        eval_ds = Dataset.from_json(config.FINETUNE_DATASET_EVAL_FILE)
        return DatasetDict({"train": train_ds, "eval": eval_ds})


# ── Drive checkpoint callback ─────────────────────────────────────────────────

class DriveCheckpointCallback(TrainerCallback):
    """Logs checkpoint saves to Drive log file."""

    def on_save(self, args, state, control, **kwargs):
        logger.info("Trainer saved checkpoint at step %d", state.global_step)

    def on_epoch_end(self, args, state, control, **kwargs):
        logger.info(
            "Epoch %d/%d complete — train loss: %s",
            int(state.epoch),
            args.num_train_epochs,
            f"{state.log_history[-1].get('loss', 'N/A')}" if state.log_history else "N/A",
        )


# ── FineTuner ─────────────────────────────────────────────────────────────────

class FineTuner:
    """
    Loads Phi-3-mini with QLoRA, trains on beat JSON pages, and saves
    the final merged model and LoRA adapter to Drive.
    """

    def __init__(self):
        self.tokenizer: Optional[AutoTokenizer] = None
        self.model: Optional[AutoModelForCausalLM] = None

    # ── Model loading ─────────────────────────────────────────────────────────

    def _load_tokenizer(self) -> None:
        logger.info("Loading tokenizer from %s", config.MODEL_NAME)
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.MODEL_NAME,
            trust_remote_code=True,
        )
        # Phi-3 uses eos as pad; set explicitly to avoid warnings
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

    def _load_model_for_training(self) -> None:
        logger.info("Loading base model with 4-bit quantization…")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            config.MODEL_NAME,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        logger.info("Preparing model for k-bit training…")
        self.model = prepare_model_for_kbit_training(
            self.model,
            use_gradient_checkpointing=True,
        )

    def _apply_lora(self) -> None:
        logger.info(
            "Applying LoRA: r=%d alpha=%d dropout=%.2f target_modules=%s",
            config.LORA_R, config.LORA_ALPHA, config.LORA_DROPOUT, config.LORA_TARGET_MODULES,
        )
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=config.LORA_R,
            lora_alpha=config.LORA_ALPHA,
            lora_dropout=config.LORA_DROPOUT,
            target_modules=config.LORA_TARGET_MODULES,
            bias="none",
            inference_mode=False,
        )
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()

    # ── Tokenization ──────────────────────────────────────────────────────────

    def _tokenize_dataset(self, dataset: DatasetDict) -> DatasetDict:
        """Tokenize the 'text' column in place."""

        def _tokenize(batch):
            tokenized = self.tokenizer(
                batch["text"],
                truncation=True,
                max_length=config.FINETUNE_MAX_SEQ_LENGTH,
                padding=False,
            )
            labels = []
            for i, text in enumerate(batch["text"]):
                input_ids = tokenized["input_ids"][i]
                
                # Mask out the system and user prompts
                # Phi-3 places assistant response directly after <|assistant|>\n
                marker = "<|assistant|>\n"
                idx = text.find(marker)
                
                if idx != -1:
                    prompt_text = text[:idx + len(marker)]
                    # Tokenize prompt to get exact token length
                    prompt_ids = self.tokenizer(
                        prompt_text,
                        truncation=True,
                        max_length=config.FINETUNE_MAX_SEQ_LENGTH,
                        padding=False
                    )["input_ids"]
                    prompt_len = min(len(prompt_ids), len(input_ids))
                    label = [-100] * prompt_len + input_ids[prompt_len:]
                else:
                    label = list(input_ids)
                
                labels.append(label)
                
            tokenized["labels"] = labels
            return tokenized

        logger.info("Tokenizing datasets…")
        tokenized = dataset.map(
            _tokenize,
            batched=True,
            remove_columns=["text"],
        )
        return tokenized

    # ── Training ──────────────────────────────────────────────────────────────

    def _build_training_args(self) -> TrainingArguments:
        os.makedirs(config.FINETUNE_OUTPUT_DIR, exist_ok=True)
        return TrainingArguments(
            output_dir=config.FINETUNE_OUTPUT_DIR,
            num_train_epochs=config.FINETUNE_EPOCHS,
            per_device_train_batch_size=config.FINETUNE_BATCH_SIZE,
            per_device_eval_batch_size=config.FINETUNE_BATCH_SIZE,
            gradient_accumulation_steps=config.FINETUNE_GRAD_ACCUMULATION_STEPS,
            learning_rate=config.FINETUNE_LEARNING_RATE,
            warmup_ratio=config.FINETUNE_WARMUP_RATIO,
            weight_decay=config.FINETUNE_WEIGHT_DECAY,
            lr_scheduler_type="cosine",
            fp16=False,
            bf16=True,
            logging_steps=config.FINETUNE_LOGGING_STEPS,
            save_steps=config.FINETUNE_SAVE_STEPS,
            eval_steps=config.FINETUNE_EVAL_STEPS,
            evaluation_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to="none",          # disable wandb/tensorboard unless user opts in
            save_total_limit=2,
            optim="paged_adamw_8bit",
            dataloader_num_workers=0,  # Drive I/O; avoid multiprocessing issues in Colab
            remove_unused_columns=False,
        )

    def _save_final_model(self) -> None:
        """Save the LoRA adapter and merge into the base model on Drive."""
        os.makedirs(config.FINETUNE_FINAL_MODEL_DIR, exist_ok=True)

        # Save the LoRA adapter
        adapter_dir = os.path.join(config.FINETUNE_FINAL_MODEL_DIR, "lora_adapter")
        os.makedirs(adapter_dir, exist_ok=True)
        logger.info("Saving LoRA adapter to %s", adapter_dir)
        self.model.save_pretrained(adapter_dir)
        self.tokenizer.save_pretrained(adapter_dir)

        # Merge LoRA weights into base model and save full model
        logger.info("Merging LoRA weights into base model…")
        try:
            merged = self.model.merge_and_unload()
            merged_dir = os.path.join(config.FINETUNE_FINAL_MODEL_DIR, "merged_model")
            os.makedirs(merged_dir, exist_ok=True)
            merged.save_pretrained(merged_dir, safe_serialization=True)
            self.tokenizer.save_pretrained(merged_dir)
            logger.info("Merged model saved to %s", merged_dir)
        except Exception as exc:
            logger.warning(
                "Could not merge LoRA weights (%s). LoRA adapter is still saved.", exc
            )

    # ── Public run method ─────────────────────────────────────────────────────

    def run(
        self,
        skip_dataset_build: bool = False,
        skip_training: bool = False,
    ) -> None:
        """
        Full fine-tuning run.

        Parameters
        ----------
        skip_dataset_build : bool
            If True, assume JSONL files already exist on Drive and skip building.
        skip_training      : bool
            If True, only build the dataset without training (useful for inspection).
        """
        _setup_logging()

        # ── Ensure output dirs ────────────────────────────────────────────
        for d in [config.FINETUNE_DIR, config.FINETUNE_OUTPUT_DIR, config.FINETUNE_FINAL_MODEL_DIR]:
            os.makedirs(d, exist_ok=True)

        # ── Load tokenizer first (needed for dataset building) ────────────
        self._load_tokenizer()

        # ── Build dataset ─────────────────────────────────────────────────
        if not skip_dataset_build:
            logger.info("Building fine-tuning dataset from Step 1 pages…")
            builder = DatasetBuilder(self.tokenizer)
            train_count, eval_count = builder.build()
            logger.info("Dataset ready: %d train, %d eval examples", train_count, eval_count)
            if train_count == 0:
                logger.error("No training examples generated — aborting fine-tune.")
                return
        else:
            logger.info("Skipping dataset build (skip_dataset_build=True)")
            if not os.path.exists(config.FINETUNE_DATASET_TRAIN_FILE):
                logger.error(
                    "Train file not found at %s. Cannot skip dataset build.",
                    config.FINETUNE_DATASET_TRAIN_FILE,
                )
                return

        if skip_training:
            logger.info("Skipping training (skip_training=True). Dataset files written to Drive.")
            return

        # ── Load dataset into HuggingFace DatasetDict ─────────────────────
        logger.info("Loading HuggingFace datasets from JSONL files…")
        builder = DatasetBuilder(self.tokenizer)
        hf_datasets = builder.load_hf_datasets()

        # ── Tokenize ──────────────────────────────────────────────────────
        tokenized_datasets = self._tokenize_dataset(hf_datasets)

        # ── Load model + LoRA ─────────────────────────────────────────────
        self._load_model_for_training()
        self._apply_lora()

        # ── Build Trainer ─────────────────────────────────────────────────
        training_args = self._build_training_args()

        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            model=self.model,
            padding=True,
            pad_to_multiple_of=8,
            label_pad_token_id=-100,  # ignore padding in loss
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["eval"],
            data_collator=data_collator,
            callbacks=[DriveCheckpointCallback()],
        )

        # ── Train ─────────────────────────────────────────────────────────
        logger.info(
            "Starting fine-tuning: %d epochs, lr=%.2e, batch=%d, grad_accum=%d",
            config.FINETUNE_EPOCHS,
            config.FINETUNE_LEARNING_RATE,
            config.FINETUNE_BATCH_SIZE,
            config.FINETUNE_GRAD_ACCUMULATION_STEPS,
        )
        trainer.train()

        # ── Save final model ──────────────────────────────────────────────
        self._save_final_model()

        logger.info("Fine-tuning complete. Model saved to %s", config.FINETUNE_FINAL_MODEL_DIR)


# ── Entrypoint ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fine-tune Phi-3 on Step 1 beat JSON pages")
    parser.add_argument(
        "--skip-dataset-build",
        action="store_true",
        help="Skip building JSONL dataset (use existing files on Drive)",
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Only build dataset, do not start training",
    )
    args = parser.parse_args()

    ft = FineTuner()
    ft.run(
        skip_dataset_build=args.skip_dataset_build,
        skip_training=args.skip_training,
    )
