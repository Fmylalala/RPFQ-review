"""Standard quantized Swin Transformer baseline model.

This module builds a quantized version of timm SwinTransformer. Swin uses
window attention and convolutional layers, so both Linear and Conv2d layers
need quantized variants.

This file is retained as a q8-style standard quantization comparison path.
The main complex-quantized Swin implementation lives in quant_rpfq_swin.py.
"""

import torch.nn as nn
from timm.models.swin_transformer import SwinTransformer
from timm.models.registry import register_model

from .quant_ops import QuantLinear, QuantConv2d


class QuantSwinTransformer(SwinTransformer):
    """Quantized Swin Transformer.
    
    Inherits from timm SwinTransformer and replaces Linear and Conv2d layers
    with quantized variants.
    
    Args:
        *args: positional SwinTransformer arguments.
        quant_bits: quantization bit width, default 8.
        quant_scheme: quantization scheme, either 'symmetric' or 'asymmetric'.
        **kwargs: keyword SwinTransformer arguments.
    """
    
    def __init__(self, *args, quant_bits: int = 8, quant_scheme: str = 'symmetric', **kwargs):
        # Drop parameters passed by timm create_model but not accepted by SwinTransformer.
        kwargs.pop('pretrained_cfg', None)
        kwargs.pop('pretrained_cfg_overlay', None)
        kwargs.pop('cache_dir', None)
        
        # Initialize the parent class first.
        super().__init__(*args, **kwargs)
        
        # Store quantization configuration.
        self.quant_bits = quant_bits
        self.quant_scheme = quant_scheme
        
        # Replace all Linear and Conv2d layers with quantized variants.
        self._replace_quant_layers()
    
    def _replace_quant_layers(self):
        """Recursively replace Linear and Conv2d layers with quantized variants."""
        # Replace all Linear layers.
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear) and not isinstance(module, QuantLinear):
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
                self._set_module_by_name(name, quant_module)
            
            # Replace Conv2d layers used by Swin components such as patch embedding.
            elif isinstance(module, nn.Conv2d) and not isinstance(module, QuantConv2d):
                # Normalize parameters that may be int or tuple.
                kernel_size = module.kernel_size[0] if isinstance(module.kernel_size, tuple) else module.kernel_size
                stride = module.stride[0] if isinstance(module.stride, tuple) else module.stride
                padding = module.padding[0] if isinstance(module.padding, tuple) else module.padding
                dilation = module.dilation[0] if isinstance(module.dilation, tuple) else module.dilation
                
                quant_module = QuantConv2d(
                    module.in_channels,
                    module.out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    dilation=dilation,
                    groups=module.groups,
                    bias=module.bias is not None,
                    quant_bits=self.quant_bits,
                    quant_scheme=self.quant_scheme,
                )
                quant_module.weight.data.copy_(module.weight.data)
                if module.bias is not None:
                    quant_module.bias.data.copy_(module.bias.data)
                
                # Replace the module.
                self._set_module_by_name(name, quant_module)
        
        # Special handling for QKV projection inside WindowAttention.
        self._replace_window_attention()
    
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
    
    def _replace_window_attention(self):
        """Replace Linear layers inside WindowAttention.
        
        TODO: refine this based on the exact timm WindowAttention implementation if needed.
        """
        for name, module in self.named_modules():
            # Identify WindowAttention modules.
            if hasattr(module, '__class__') and 'WindowAttention' in module.__class__.__name__:
                # Replace QKV projection.
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
                
                # Replace output projection.
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


@register_model
def swin_base_patch4_window7_224_q(pretrained: bool = False, **kwargs):
    """Quantized Swin-Base.
    
    Args:
        pretrained: whether to load pretrained weights.
        **kwargs: additional model parameters, including quant_bits and quant_scheme.
    """
    # Drop parameters passed by timm create_model.
    kwargs.pop('pretrained_cfg', None)
    kwargs.pop('pretrained_cfg_overlay', None)
    kwargs.pop('cache_dir', None)
    
    quant_bits = kwargs.pop('quant_bits', 8)
    quant_scheme = kwargs.pop('quant_scheme', 'symmetric')
    
    model = QuantSwinTransformer(
        patch_size=4,
        window_size=7,
        embed_dim=128,
        depths=(2, 2, 18, 2),
        num_heads=(4, 8, 16, 32),
        quant_bits=quant_bits,
        quant_scheme=quant_scheme,
        **kwargs
    )
    return model


@register_model
def swin_small_patch4_window7_224_q(pretrained: bool = False, **kwargs):
    """Quantized Swin-Small."""
    kwargs.pop('pretrained_cfg', None)
    kwargs.pop('pretrained_cfg_overlay', None)
    kwargs.pop('cache_dir', None)
    
    quant_bits = kwargs.pop('quant_bits', 8)
    quant_scheme = kwargs.pop('quant_scheme', 'symmetric')
    
    model = QuantSwinTransformer(
        patch_size=4,
        window_size=7,
        embed_dim=96,
        depths=(2, 2, 18, 2),
        num_heads=(3, 6, 12, 24),
        quant_bits=quant_bits,
        quant_scheme=quant_scheme,
        **kwargs
    )
    return model


@register_model
def swin_tiny_patch4_window7_224_q(pretrained: bool = False, **kwargs):
    """Quantized Swin-Tiny."""
    kwargs.pop('pretrained_cfg', None)
    kwargs.pop('pretrained_cfg_overlay', None)
    kwargs.pop('cache_dir', None)
    
    quant_bits = kwargs.pop('quant_bits', 8)
    quant_scheme = kwargs.pop('quant_scheme', 'symmetric')
    
    model = QuantSwinTransformer(
        patch_size=4,
        window_size=7,
        embed_dim=96,
        depths=(2, 2, 6, 2),
        num_heads=(3, 6, 12, 24),
        quant_bits=quant_bits,
        quant_scheme=quant_scheme,
        **kwargs
    )
    return model
