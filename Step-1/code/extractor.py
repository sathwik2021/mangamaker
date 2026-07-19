# extractor.py
import json
import re
import logging
from typing import Any, Dict, Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from transformers.cache_utils import Cache

import config

# Compatibility shim for older model code that expects `past_key_values.seen_tokens`
if not hasattr(Cache, "seen_tokens"):
    @property
    def seen_tokens(self) -> int:
        return self.get_seq_length()

    Cache.seen_tokens = seen_tokens

logger = logging.getLogger(__name__)

# ── JSON schema description injected into every prompt ──────────────────────
_SCHEMA_DESCRIPTION = """
Return ONLY a valid JSON object with exactly this structure — no extra fields, no markdown:

{
  "page_id": "<string: unique identifier for this page>",
  "source_chunk_id": "<string: chunk_id passed in>",
  "characters": ["<string: character name>", ...],
  "beats": [
    {
      "id": "<string: beat identifier, e.g. beat_1>",
      "order": <integer: 1-based sequential index>,
      "type": "<string: one of action|dialogue|reaction|description|transition>",
      "characters": ["<string: must exist in top-level characters list>"],
      "text": "<string: for dialogue beats — the spoken words>",
      "verb": "<string: for action beats — the main action verb>",
      "emotion": "<string: for reaction beats — the emotion felt>",
      "intensity": <integer: 1–10, emotion intensity>,
      "causes": ["<string: id of beat that caused this one>"],
      "description": "<string: visual description of the beat>"
    },
    ...
  ],
  "emotional_flow": ["<string: emotion label>", ...]
}

Rules:
- beats must be non-empty
- order values must be sequential integers starting at 1
- id must be "beat_<order>" (e.g. beat_1, beat_2)
- characters at beat level must be a subset of top-level characters
- dialogue beats MUST have a non-empty "text" field
- action beats MUST have a non-empty "verb" field
- reaction beats MUST have a non-empty "emotion" field
- emotional_flow must be a non-empty list of strings
- causes may be an empty list []
- Do NOT add any fields not listed above
"""

_SYSTEM_PROMPT = (
    "You are a manga story analyst. Extract narrative beats from the novel excerpt. "
    + _SCHEMA_DESCRIPTION
)


def _build_user_prompt(chunk_text: str, chunk_id: str, feedback: Optional[str] = None) -> str:
    base = (
        f"Analyze the following novel excerpt and extract narrative beats.\n"
        f"source_chunk_id: {chunk_id}\n"
        f"page_id: page_{chunk_id}\n\n"
        f"EXCERPT:\n{chunk_text}\n\n"
        "Return only the JSON object."
    )
    if feedback:
        base += f"\n\nPREVIOUS VALIDATION ERRORS — fix these:\n{feedback}"
    return base


def _extract_json_from_text(raw: str) -> Optional[Dict[str, Any]]:
    """
    Attempt to parse JSON from model output.
    Handles:
      - Markdown code fences (```json ... ```)
      - Extra text before/after JSON
      - Partial malformed JSON via regex recovery
    """
    # 1. Strip markdown fences
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
    if fenced:
        candidate = fenced.group(1)
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    # 2. Find the outermost { ... } block or everything after { if unclosed
    brace_match = re.search(r"(\{.*)", raw, re.DOTALL)
    if brace_match:
        candidate = brace_match.group(1)
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    # 3. Progressive truncation and stack-based recovery
    if brace_match:
        candidate = brace_match.group(1)
        for end in range(len(candidate), len(candidate) // 2, -1):
            trimmed = candidate[:end]
            
            # Stack parsing to find unclosed structures
            stack = []
            in_string = False
            escape = False
            for char in trimmed:
                if escape:
                    escape = False
                    continue
                if char == '\\':
                    escape = True
                    continue
                if char == '"':
                    in_string = not in_string
                    continue
                if not in_string:
                    if char in '{[':
                        stack.append(char)
                    elif char == '}':
                        if stack and stack[-1] == '{': stack.pop()
                    elif char == ']':
                        if stack and stack[-1] == '[': stack.pop()
            
            closed = trimmed
            if in_string:
                closed += '"'
                
            # Pop remaining from stack and add corresponding closing tags
            for top in reversed(stack):
                if top == '{': closed += '}'
                elif top == '[': closed += ']'
                
            try:
                return json.loads(closed)
            except json.JSONDecodeError:
                continue

    return None


class Extractor:
    """Loads Phi-3 and extracts structured beat JSON from text chunks."""

    def __init__(self):
        self.tokenizer: Optional[AutoTokenizer] = None
        self.model: Optional[AutoModelForCausalLM] = None
        self._loaded = False

    def load_model(self) -> None:
        """Load the Phi-3 model with optional 4-bit quantization."""
        logger.info("Loading tokenizer from %s", config.MODEL_NAME)
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.MODEL_NAME, trust_remote_code=True
        )

        if config.USE_4BIT_QUANTIZATION:
            logger.info("Applying 4-bit quantization (BitsAndBytesConfig)")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                config.MODEL_NAME,
                quantization_config=bnb_config,
                trust_remote_code=True,
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                config.MODEL_NAME,
                device_map="cpu",
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
            )

        self.model.eval()
        self._loaded = True
        logger.info("Model loaded successfully")

    def _run_inference(self, user_prompt: str) -> str:
        """Run a single forward pass and return the raw generated text."""
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        # Phi-3 chat template
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(self.model.device)

        with torch.no_grad():
            attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=self.model.device)
            output_ids = self.model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=config.MODEL_MAX_NEW_TOKENS,
                temperature=config.MODEL_TEMPERATURE,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                use_cache=False,
            )

        # Decode only the newly generated tokens
        generated = output_ids[0][input_ids.shape[-1]:]
        return self.tokenizer.decode(generated, skip_special_tokens=True)

    def extract(
        self,
        chunk_text: str,
        chunk_id: str,
        feedback: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Extract beats from a text chunk using Phi-3.

        Parameters
        ----------
        chunk_text : str  – the novel excerpt
        chunk_id   : str  – identifier to embed as source_chunk_id
        feedback   : str  – validation error feedback for retry prompts

        Returns
        -------
        Parsed JSON dict or None if parsing fails.
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        user_prompt = _build_user_prompt(chunk_text, chunk_id, feedback)
        raw_output = self._run_inference(user_prompt)
        logger.debug("Raw model output:\n%s", raw_output)

        parsed = _extract_json_from_text(raw_output)
        if parsed is None:
            logger.warning("Failed to parse JSON from model output for chunk %s", chunk_id)
        return parsed
