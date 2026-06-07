from .granite_onnx_ar import GraniteOnnxARAdapter
from .granite_onnx_nar import GraniteOnnxNARAdapter
from .hf_transformers_asr import HFTransformersASRAdapter
from .hf_whisper_asr import HFWhisperASRAdapter
from .faster_whisper_asr import FasterWhisperASRAdapter
from .whisper_cpp_asr import WhisperCppASRAdapter
from .openai_whisper_pt import OpenAIWhisperPTAdapter
from .generic_onnx_manifest import GenericOnnxManifestAdapter
from .gguf_asr_mmproj import GGUFASRMMProjAdapter
from .gguf_llm_reference import GGUFLLMReferenceAdapter


BUILTIN_ADAPTERS = [
    GraniteOnnxARAdapter(),
    GraniteOnnxNARAdapter(),
    HFWhisperASRAdapter(),
    HFTransformersASRAdapter(),
    FasterWhisperASRAdapter(),
    WhisperCppASRAdapter(),
    OpenAIWhisperPTAdapter(),
    GenericOnnxManifestAdapter(),
    GGUFASRMMProjAdapter(),
    GGUFLLMReferenceAdapter(),
]
