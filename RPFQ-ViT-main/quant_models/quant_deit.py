"""Standard quantized DeiT baseline model.

This module builds a quantized version of timm VisionTransformerDistilled.
DeiT is essentially ViT with a distillation token, so it can reuse QuantVisionTransformer logic.

This file is retained as a q8-style standard quantization comparison path.
The main complex-quantized DeiT implementation lives in quant_rpfq_deit.py.
"""

import torch.nn as nn
from timm.models.deit import VisionTransformerDistilled
from timm.models.registry import register_model

from .quant_vit import QuantVisionTransformer


class QuantDeiT(QuantVisionTransformer):
    """Quantized DeiT (Data-efficient Image Transformer).
    
    Inherits from QuantVisionTransformer. DeiT adds a distillation token to ViT,
    while the quantization logic remains the same.
    
    Args:
        *args: positional VisionTransformer arguments.
        quant_bits: quantization bit width, default 8.
        quant_scheme: quantization scheme, either 'symmetric' or 'asymmetric'.
        **kwargs: keyword VisionTransformer arguments.
    """
    
    def __init__(self, *args, quant_bits: int = 8, quant_scheme: str = 'symmetric', **kwargs):
        # DeiT needs special initialization, so the distilled VisionTransformer path needs care.
        
        # First call QuantVisionTransformer initialization, then handle DeiT-specific pieces.
        
        # Current approach: create the standard model and then replace eligible layers.
        super().__init__(*args, quant_bits=quant_bits, quant_scheme=quant_scheme, **kwargs)
        
        # The DeiT-specific distillation head also needs quantization.
        if hasattr(self, 'head_dist') and isinstance(self.head_dist, nn.Linear):
            from .quant_ops import QuantLinear
            quant_head_dist = QuantLinear(
                self.head_dist.in_features,
                self.head_dist.out_features,
                bias=self.head_dist.bias is not None,
                quant_bits=quant_bits,
                quant_scheme=quant_scheme,
            )
            quant_head_dist.weight.data.copy_(self.head_dist.weight.data)
            if self.head_dist.bias is not None:
                quant_head_dist.bias.data.copy_(self.head_dist.bias.data)
            self.head_dist = quant_head_dist


# Alternative implementation: inherit directly from VisionTransformerDistilled and apply quantization.
class QuantDeiTBase(VisionTransformerDistilled):
    """Alternative quantized DeiT implementation."""
    
    def __init__(self, *args, quant_bits: int = 8, quant_scheme: str = 'symmetric', **kwargs):
        # Drop unsupported parameters passed by timm create_model.
        kwargs.pop('pretrained_cfg', None)
        kwargs.pop('pretrained_cfg_overlay', None)
        kwargs.pop('cache_dir', None)
        
        super().__init__(*args, **kwargs)
        self.quant_bits = quant_bits
        self.quant_scheme = quant_scheme
        self._replace_quant_layers()
    
    def _replace_quant_layers(self):
        """Replace all Linear layers with quantized variants."""
        from .quant_ops import QuantLinear
        
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                quant_module = QuantLinear(
                    module.in_features,
                    module.out_features,
                    bias=module.bias is not None,
                    quant_bits=self.quant_bits,
                    quant_scheme=self.quant_scheme,
                )
                quant_module.weight.data.copy_(module.weight.data)
                if module.bias is not None:
                    quant_module.bias.data.copy_(module.bias.data)
                
                # Replace the module.
                parent = self
                *parents, last = name.split('.')
                for p in parents:
                    parent = getattr(parent, p)
                setattr(parent, last, quant_module)


@register_model
def deit_base_patch16_224_q(pretrained: bool = False, **kwargs):
    """Quantized DeiT-Base/16."""
    quant_bits = kwargs.pop('quant_bits', 8)
    quant_scheme = kwargs.pop('quant_scheme', 'symmetric')
    
    model = QuantDeiTBase(
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=nn.LayerNorm,
        quant_bits=quant_bits,
        quant_scheme=quant_scheme,
        **kwargs
    )
    return model


@register_model
def deit_small_patch16_224_q(pretrained: bool = False, **kwargs):
    """Quantized DeiT-Small/16."""
    quant_bits = kwargs.pop('quant_bits', 8)
    quant_scheme = kwargs.pop('quant_scheme', 'symmetric')
    
    model = QuantDeiTBase(
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=nn.LayerNorm,
        quant_bits=quant_bits,
        quant_scheme=quant_scheme,
        **kwargs
    )
    return model


@register_model
def deit_tiny_patch16_224_q(pretrained: bool = False, **kwargs):
    """Quantized DeiT-Tiny/16."""
    quant_bits = kwargs.pop('quant_bits', 8)
    quant_scheme = kwargs.pop('quant_scheme', 'symmetric')
    
    model = QuantDeiTBase(
        patch_size=16,
        embed_dim=192,
        depth=12,
        num_heads=3,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=nn.LayerNorm,
        quant_bits=quant_bits,
        quant_scheme=quant_scheme,
        **kwargs
    )
    return model
