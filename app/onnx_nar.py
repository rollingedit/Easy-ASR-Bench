from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from .frontend import input_features
from .onnx_common import choose_providers, load_tokenizer, make_session, session_input_names


class GraniteOnnxNAR:
    def __init__(
        self,
        model_root: Path,
        precision: str,
        provider: str = "auto",
        cpu_threads: int = 0,
    ) -> None:
        start = time.perf_counter()
        self.model_root = model_root
        self.precision = precision
        self.providers = choose_providers(provider)
        folder = model_root / precision
        self.encoder = make_session(folder / "encoder.onnx", self.providers, cpu_threads)
        self.embed_tokens = make_session(folder / "embed_tokens.onnx", self.providers, cpu_threads)
        self.editor = make_session(folder / "editor.onnx", self.providers, cpu_threads)
        self.tokenizer = load_tokenizer(model_root)
        self.eos_token_id = self.tokenizer.token_to_id("<|end_of_text|>")
        self.blank_token_id = self._blank_token_id()
        self.model_load_seconds = time.perf_counter() - start

    @property
    def actual_provider(self) -> str:
        return ",".join(self.encoder.get_providers())

    def _blank_token_id(self) -> int:
        metadata = self.model_root / "granite_export_metadata.json"
        if metadata.exists():
            data = json.loads(metadata.read_text(encoding="utf-8"))
            for key in ["ctc_blank_token_id", "blank_token_id", "ctc_blank_id"]:
                if key in data:
                    return int(data[key])
        return 0

    def _embed(self, input_ids: np.ndarray) -> np.ndarray:
        name = session_input_names(self.embed_tokens)[0]
        return self.embed_tokens.run(None, {name: input_ids.astype(np.int64)})[0]

    def _ctc_decode(self, logits: np.ndarray, mask: np.ndarray | None) -> list[int]:
        ids = np.argmax(logits, axis=-1)
        if ids.ndim == 2:
            ids = ids[0]
        if mask is not None:
            valid = np.asarray(mask).astype(bool)
            if valid.ndim > 1:
                valid = valid[0]
            ids = ids[: len(valid)][valid[: len(ids)]]
        decoded: list[int] = []
        previous: int | None = None
        for token in ids.tolist():
            token = int(token)
            if token != previous and token != self.blank_token_id:
                decoded.append(token)
            previous = token
        return decoded

    def _slots(self, draft_ids: list[int]) -> np.ndarray:
        ids: list[int] = [self.eos_token_id]
        for token in draft_ids:
            ids.extend([token, self.eos_token_id])
        while len(ids) < 8:
            ids.append(self.eos_token_id)
        return np.array([ids], dtype=np.int64)

    def transcribe_array(self, audio_float32_16k_mono: np.ndarray, prompt: str = "") -> dict:
        del prompt
        started = time.perf_counter()
        features = input_features(audio_float32_16k_mono, self.model_root)
        attention_mask = np.ones(features.shape[:2], dtype=np.int64)
        feed = {}
        for name in session_input_names(self.encoder):
            lower = name.lower()
            if "feature" in lower or "input" in lower:
                feed[name] = features
            elif "mask" in lower:
                feed[name] = attention_mask
            else:
                raise ValueError(f"Unsupported NAR encoder input: {name}")
        outputs = self.encoder.run(None, feed)
        bpe_logits = outputs[0]
        bpe_mask = outputs[1] if len(outputs) > 1 else None
        audio_embeds = outputs[2] if len(outputs) > 2 else None
        audio_lengths = outputs[3] if len(outputs) > 3 else None
        if audio_embeds is None:
            raise ValueError("NAR encoder did not return audio embeddings")

        draft_ids = self._ctc_decode(bpe_logits, bpe_mask)
        slots = self._slots(draft_ids)
        text_embeds = self._embed(slots)
        audio_len = int(np.ravel(audio_lengths)[0]) if audio_lengths is not None else audio_embeds.shape[1]
        inputs_embeds = np.concatenate([audio_embeds[:, :audio_len, :], text_embeds], axis=1).astype(np.float32)
        total = inputs_embeds.shape[1]
        editor_feed = {}
        for name in session_input_names(self.editor):
            lower = name.lower()
            if "embed" in lower:
                editor_feed[name] = inputs_embeds
            elif "position" in lower:
                editor_feed[name] = np.arange(total, dtype=np.int64)[None, :]
            elif "mask" in lower:
                editor_feed[name] = np.zeros((1, 1, total, total), dtype=np.float32)
            else:
                raise ValueError(f"Unsupported NAR editor input: {name}")
        editor_outputs = self.editor.run(None, editor_feed)
        text_logits = editor_outputs[0][:, audio_len:, :]
        final_ids = np.argmax(text_logits, axis=-1)[0].tolist()
        final_ids = [int(token) for token in final_ids if int(token) not in {self.eos_token_id, self.blank_token_id}]
        text = self.tokenizer.decode(final_ids, skip_special_tokens=True).strip()
        return {
            "text": text,
            "tokens_generated": len(final_ids),
            "inference_seconds": time.perf_counter() - started,
            "stop_reason": "complete",
        }
