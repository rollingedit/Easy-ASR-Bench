from .granite_onnx_ar import GraniteOnnxARAdapter
from .granite_onnx_nar import GraniteOnnxNARAdapter
from .hf_transformers_asr import HFTransformersASRAdapter
from .generic_onnx_manifest import GenericOnnxManifestAdapter
from .gguf_llm_reference import GGUFLLMReferenceAdapter


BUILTIN_ADAPTERS = [
    GraniteOnnxARAdapter(),
    GraniteOnnxNARAdapter(),
    HFTransformersASRAdapter(),
    GenericOnnxManifestAdapter(),
    GGUFLLMReferenceAdapter(),
]
