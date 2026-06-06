from .granite_onnx_ar import GraniteOnnxARAdapter
from .granite_onnx_nar import GraniteOnnxNARAdapter


BUILTIN_ADAPTERS = [
    GraniteOnnxARAdapter(),
    GraniteOnnxNARAdapter(),
]
