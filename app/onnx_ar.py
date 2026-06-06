from __future__ import annotations

import time
from pathlib import Path

from jinja2 import Template
import numpy as np

from .frontend import input_features
from .onnx_common import causal_mask, choose_providers, decode_mask, load_tokenizer, make_session, session_input_names


class GraniteOnnxAR:
    def __init__(
        self,
        model_root: Path,
        precision: str,
        provider: str = "auto",
        cpu_threads: int = 0,
        max_new_tokens: int = 1024,
    ) -> None:
        start = time.perf_counter()
        self.model_root = model_root
        self.precision = precision
        self.max_new_tokens = max_new_tokens
        self.providers = choose_providers(provider)
        folder = model_root / precision
        self.encoder = make_session(folder / "encoder.onnx", self.providers, cpu_threads)
        self.embed_tokens = make_session(folder / "embed_tokens.onnx", self.providers, cpu_threads)
        self.prompt_encode = make_session(folder / "prompt_encode.onnx", self.providers, cpu_threads)
        self.decode_step = make_session(folder / "decode_step.onnx", self.providers, cpu_threads)
        self.tokenizer = load_tokenizer(model_root)
        self.audio_token_id = self.tokenizer.token_to_id("<|audio|>")
        self.eos_token_id = self.tokenizer.token_to_id("<|end_of_text|>")
        self.pad_token_id = self.tokenizer.token_to_id("<|pad|>")
        if self.audio_token_id != 100352:
            raise ValueError(f"Unexpected <|audio|> token id {self.audio_token_id}; expected 100352")
        self.model_load_seconds = time.perf_counter() - start

    @property
    def actual_provider(self) -> str:
        return ",".join(self.encoder.get_providers())

    def _chat_prompt(self, prompt: str) -> str:
        content = prompt if "<|audio|>" in prompt else f"<|audio|>{prompt}"
        template_path = self.model_root / "chat_template.jinja"
        if template_path.exists():
            rendered = Template(template_path.read_text(encoding="utf-8")).render(
                messages=[{"role": "user", "content": content}],
                add_generation_prompt=True,
            )
            if rendered.strip():
                return rendered
        return f"USER: {content}\n\nASSISTANT:"

    def _embed(self, input_ids: np.ndarray) -> np.ndarray:
        name = session_input_names(self.embed_tokens)[0]
        return self.embed_tokens.run(None, {name: input_ids.astype(np.int64)})[0]

    def transcribe_array(self, audio_float32_16k_mono: np.ndarray, prompt: str) -> dict:
        started = time.perf_counter()
        features = input_features(audio_float32_16k_mono, self.model_root)
        encoder_names = session_input_names(self.encoder)
        encoder_outputs = self.encoder.run(None, {encoder_names[0]: features})
        audio_embeds = encoder_outputs[0]
        audio_embed_sizes = encoder_outputs[1] if len(encoder_outputs) > 1 else np.array([audio_embeds.shape[1]], dtype=np.int64)

        encoded = self.tokenizer.encode(self._chat_prompt(prompt))
        input_ids = np.array([encoded.ids], dtype=np.int64)
        text_embeds = self._embed(input_ids)
        positions = np.where(input_ids[0] == self.audio_token_id)[0]
        if len(positions) != 1:
            raise ValueError(f"Expected exactly one <|audio|> token in prompt; got {len(positions)}")
        pos = int(positions[0])
        n_audio = int(np.ravel(audio_embed_sizes)[0])
        inputs_embeds = np.concatenate(
            [text_embeds[:, :pos, :], audio_embeds[:, :n_audio, :], text_embeds[:, pos + 1 :, :]],
            axis=1,
        ).astype(np.float32)

        prompt_len = inputs_embeds.shape[1]
        prompt_feed = {}
        for name in session_input_names(self.prompt_encode):
            lower = name.lower()
            if "embed" in lower:
                prompt_feed[name] = inputs_embeds
            elif "position" in lower:
                prompt_feed[name] = np.arange(prompt_len, dtype=np.int64)[None, :]
            elif "mask" in lower:
                prompt_feed[name] = causal_mask(prompt_len)
            else:
                raise ValueError(f"Unsupported prompt_encode input: {name}")
        prompt_outputs = self.prompt_encode.run(None, prompt_feed)
        logits = prompt_outputs[0]
        cache = list(prompt_outputs[1:])

        generated: list[int] = []
        next_token = int(np.argmax(logits[:, -1, :], axis=-1)[0])
        stop_reason = "max_new_tokens"
        current_position = prompt_len
        for _ in range(self.max_new_tokens):
            if next_token == self.eos_token_id:
                stop_reason = "eos"
                break
            generated.append(next_token)
            token_ids = np.array([[next_token]], dtype=np.int64)
            token_embed = self._embed(token_ids)
            feed = {}
            cache_index = 0
            for name in session_input_names(self.decode_step):
                lower = name.lower()
                if "embed" in lower:
                    feed[name] = token_embed
                elif "position" in lower:
                    feed[name] = np.array([[current_position]], dtype=np.int64)
                elif "mask" in lower:
                    feed[name] = decode_mask(current_position + 1)
                else:
                    if cache_index >= len(cache):
                        raise ValueError(f"No KV cache tensor available for decode input {name}")
                    feed[name] = cache[cache_index]
                    cache_index += 1
            decode_outputs = self.decode_step.run(None, feed)
            logits = decode_outputs[0]
            cache = list(decode_outputs[1:])
            next_token = int(np.argmax(logits[:, -1, :], axis=-1)[0])
            current_position += 1

        text = self.tokenizer.decode(generated, skip_special_tokens=True).strip()
        return {
            "text": text,
            "tokens_generated": len(generated),
            "inference_seconds": time.perf_counter() - started,
            "stop_reason": stop_reason,
        }
