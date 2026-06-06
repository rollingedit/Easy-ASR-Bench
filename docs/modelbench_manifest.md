# Generic ONNX Manifest

Generic ONNX ASR models require `modelbench.json`.

Easy ASR Bench does not guess arbitrary ONNX preprocessing and decoding. The manifest tells the app which built-in safe recipe to use.

## CTC Example

```json
{
  "schema": "easy_asr_bench.model_manifest.v1",
  "display_name": "Custom ONNX CTC ASR",
  "task": "automatic-speech-recognition",
  "backend": "onnxruntime",
  "precision": "int8",
  "files": {"model": "model.onnx"},
  "audio": {
    "sample_rate": 16000,
    "channels": 1
  },
  "inputs": {
    "waveform": {"name": "input_values", "dtype": "float32"},
    "attention_mask": {"name": "attention_mask", "dtype": "int64", "optional": true}
  },
  "outputs": {"logits": "logits"},
  "preprocessing": {"type": "raw_waveform", "normalize": true},
  "decoding": {"type": "ctc", "blank_token_id": 0, "vocab_file": "vocab.json"}
}
```

CTC manifests must include either `decoding.vocab` or `decoding.vocab_file`. Without a vocab, the app cannot produce text and the model is shown as incomplete instead of emitting numeric token IDs.

Unsupported manifests are shown in the model scanner with exact missing fields.
