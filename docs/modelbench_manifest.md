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
  "files": {
    "model": "model.onnx"
  },
  "audio": {
    "sample_rate": 16000,
    "channels": 1
  },
  "preprocessing": {
    "type": "granite_log_mel"
  },
  "decoding": {
    "type": "ctc",
    "blank_token_id": 0,
    "vocab": {
      "0": "",
      "1": "a",
      "2": "b"
    }
  }
}
```

Unsupported manifests are shown in the model scanner with exact missing fields.
