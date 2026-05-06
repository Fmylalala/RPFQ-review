"""Standard quantized ViT baseline model.

This module builds a quantized version of timm VisionTransformer by replacing
Linear and Attention layers with quantized variants.

This file is retained as a q8-style standard quantization comparison path.
The main complex-quantized ViT implementation lives in quant_rpfq_vit.py.
"""

import torch.nn as nn
from timm.models.vision_transformer import VisionTransformer
from timm.models.registry import register_model
from timm.layers import trunc_normal_

from .quant_ops import QuantLinear, QuantAttention


class QuantVisionTransformer(VisionTransformer):
    """Quantized Vision Transformer.
    
    Inherits from timm VisionTransformer and replaces Linear and Attention
    layers with quantized versions.
    
    Args:
        *args: positional VisionTransformer arguments.
        quant_bits: quantization bit width, default 8.
        quant_scheme: quantization scheme, either 'symmetric' or 'asymmetric'.
        **kwargs: keyword VisionTransformer arguments.
    """
    
    def __init__(self, *args, quant_bits: int = 8, quant_scheme: str = 'symmetric', **kwargs):
        # Drop parameters passed by timm create_model but not accepted by VisionTransformer.
        kwargs.pop('pretrained_cfg', None)
        kwargs.pop('pretrained_cfg_overlay', None)
        kwargs.pop('cache_dir', None)
        
        # Initialize the parent class first.
        super().__init__(*args, **kwargs)
        
        # Store quantization configuration.
        self.quant_bits = quant_bits
        self.quant_scheme = quant_scheme
        
        # Replace all Linear and Attention layers with quantized variants.
        self._replace_quant_layers()
    
    def _replace_quant_layers(self):
        """Recursively replace Linear and Attention layers with quantized variants."""
        # Replace projection after patch embedding if needed.
        # Conv2d inside PatchEmbed usually does not need quantization, but can be handled here.
        
        # Replace all Linear layers.
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear) and not isinstance(module, QuantLinear):
                # Skip layers that are handled inside Attention modules.
                parent_name = '.'.join(name.split('.')[:-1])
                if parent_name and hasattr(self, parent_name.split('.')[0]):
                    # Attention modules are handled separately.
                    parent_module = self._get_module_by_name(parent_name)
                    if parent_module is not None and hasattr(parent_module, '__class__'):
                        if 'Attention' in parent_module.__class__.__name__:
                            continue  # Attention modules are handled separately.
                
                # Create a quantized Linear layer.
                quant_module = QuantLinear(
                    module.in_features,
                    module.out_features,
                    bias=module.bias is not None,
                    quant_bits=self.quant_bits,
                    quant_scheme=self.quant_scheme,
                )
                # Copy weights.
                quant_module.weight.data.copy_(module.weight.data)
                if module.bias is not None:
                    quant_module.bias.data.copy_(module.bias.data)
                
                # Replace the module.
                self._set_module_by_name(name, quant_module)
        
        # Replace Attention layers; they need special handling because they contain multiple Linear layers.
        self._replace_attention_layers()
    
    def _get_module_by_name(self, name: str):
        """Get a module by dotted name."""
        names = name.split('.')
        module = self
        for n in names:
            if n:
                module = getattr(module, n, None)
                if module is None:
                    return None
        return module
    
    def _set_module_by_name(self, name: str, module: nn.Module):
        """Set a module by dotted name."""
        names = name.split('.')
        parent = self
        for n in names[:-1]:
            if n:
                parent = getattr(parent, n)
        setattr(parent, names[-1], module)
    
    def _replace_attention_layers(self):
        """Replace Attention layers with quantized variants.
        
        TODO: refine this based on the exact timm Attention implementation if needed.
        """
        # Traverse all modules and find Attention layers.
        for name, module in self.named_modules():
            # Identify Attention modules by class name.
            if hasattr(module, '__class__') and 'Attention' in module.__class__.__name__:
                # Replace Linear layers inside Attention.
                if hasattr(module, 'qkv') and isinstance(module.qkv, nn.Linear):
                    quant_qkv = QuantLinear(
                        module.qkv.in_features,
                        module.qkv.out_features,
                        bias=module.qkv.bias is not None,
                        quant_bits=self.quant_bits,
                        quant_scheme=self.quant_scheme,
                    )
                    quant_qkv.weight.data.copy_(module.qkv.weight.data)
                    if module.qkv.bias is not None:
                        quant_qkv.bias.data.copy_(module.qkv.bias.data)
                    module.qkv = quant_qkv
                
                if hasattr(module, 'proj') and isinstance(module.proj, nn.Linear):
                    quant_proj = QuantLinear(
                        module.proj.in_features,
                        module.proj.out_features,
                        bias=module.proj.bias is not None,
                        quant_bits=self.quant_bits,
                        quant_scheme=self.quant_scheme,
                    )
                    quant_proj.weight.data.copy_(module.proj.weight.data)
                    if module.proj.bias is not None:
                        quant_proj.bias.data.copy_(module.proj.bias.data)
                    module.proj = quant_proj


# Register quantized models with timm-compatible names plus the _q suffix.
@register_model
def vit_base_patch16_224_q(pretrained: bool = False, **kwargs):
    """Quantized ViT-Base/16.
    
    Args:
        pretrained: whether to load pretrained weights; FP32 weights need adaptation.
        **kwargs: additional model parameters, including quant_bits and quant_scheme.
    """
    quant_bits = kwargs.pop('quant_bits', 8)
    quant_scheme = kwargs.pop('quant_scheme', 'symmetric')
    
    model = QuantVisionTransformer(
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
    
    if pretrained:
        # TODO: load and adapt FP32 pretrained weights.
        # Example:
        # from timm.models import load_pretrained
        # checkpoint = load_pretrained('vit_base_patch16_224', pretrained=True)
        # model.load_state_dict(checkpoint, strict=False)
        # Quantization calibration may be needed afterward.
        pass
    
    return model


@register_model
def vit_large_patch16_224_q(pretrained: bool = False, **kwargs):
    """Quantized ViT-Large/16."""
    quant_bits = kwargs.pop('quant_bits', 8)
    quant_scheme = kwargs.pop('quant_scheme', 'symmetric')
    
    model = QuantVisionTransformer(
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=nn.LayerNorm,
        quant_bits=quant_bits,
        quant_scheme=quant_scheme,
        **kwargs
    )
    return model


@register_model
def vit_small_patch16_224_q(pretrained: bool = False, **kwargs):
    """Quantized ViT-Small/16."""
    quant_bits = kwargs.pop('quant_bits', 8)
    quant_scheme = kwargs.pop('quant_scheme', 'symmetric')
    
    model = QuantVisionTransformer(
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
def vit_tiny_patch16_224_q(pretrained: bool = False, **kwargs):
    """Quantized ViT-Tiny/16."""
    quant_bits = kwargs.pop('quant_bits', 8)
    quant_scheme = kwargs.pop('quant_scheme', 'symmetric')
    
    model = QuantVisionTransformer(
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
