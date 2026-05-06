"""Quantized Vision Transformer model registrations.

Importing this package registers all quantized model variants with timm.

Complex quantization mainline:
    - quant_i: complex-phase weight and activation quantizers
    - quant_rpfq_vit / quant_rpfq_deit / quant_rpfq_swin: RPFQViT-family model adapters
    - quant_rpfq_vit_vision: vision-aware complex quantization variant

Baseline quantization:
    - quant_ops: ordinary quantized layers
    - quant_vit / quant_deit / quant_swin: baseline model adapters
"""

from .quant_vit import *
from .quant_deit import *
from .quant_swin import *
from .quant_rpfq_vit import *
from .quant_rpfq_deit import *
from .quant_rpfq_swin import *
from .quant_rpfq_vit_vision import *
from .quant_ops import QuantLinear, QuantConv2d, QuantAttention

__all__ = [
    "QuantLinear",
    "QuantConv2d",
    "QuantAttention",
]
