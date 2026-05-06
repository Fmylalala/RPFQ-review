from __future__ import annotations

"""
RPFQViT quantization core module.

This file provides three layers of functionality:
1) complex weight quantization with several strategies and STE
2) activation quantization with symmetric int quantization and STE
3) complex-structured linear layers via QATLinearComplexPhase

Implementation approach:
- factor the quantization flow into reusable pure functions for masks, scales, and residual steps
- wrap those functions with autograd.Function STE quantizers
- compose them into trainable, replaceable linear layers through nn.Module
"""

import math
from typing import Callable, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Common types and numerical-stability constants.
TensorPair = Tuple[torch.Tensor, torch.Tensor]
EPS = 1e-6


def _safe_mean_abs(x: torch.Tensor, mask: torch.Tensor, min_val: float = 1e-5) -> torch.Tensor:
    """Compute masked mean absolute value and clamp it to keep scales stable."""
    mask_f = mask.to(dtype=x.dtype)
    mean_abs = (x.abs() * mask_f).sum() / mask_f.sum().clamp_min(1.0)
    return mean_abs.clamp_min(min_val)


def _axis_masks(w_real: torch.Tensor, w_imag: torch.Tensor) -> Tuple[torch.Tensor, ...]:
    """
    Build four masks from the complex direction:
    - real_pos/real_neg: projected to positive or negative real axis
    - imag_pos/imag_neg: projected to positive or negative imaginary axis

    The dominant axis is selected by comparing |real| and |imag|.
    Ties are assigned stably by quadrant.
    """
    abs_real = w_real.abs()
    abs_imag = w_imag.abs()

    real_dominant = abs_real > abs_imag
    imag_dominant = abs_imag > abs_real
    tie = ~(real_dominant | imag_dominant)

    zero_tie = tie & (w_real == 0) & (w_imag == 0)
    tie_q1 = tie & (w_real > 0) & (w_imag > 0)
    tie_q2 = tie & (w_real < 0) & (w_imag > 0)
    tie_q3 = tie & (w_real < 0) & (w_imag < 0)
    tie_q4 = tie & (w_real > 0) & (w_imag < 0)

    real_pos = (real_dominant & (w_real >= 0)) | tie_q4 | zero_tie
    real_neg = (real_dominant & (w_real < 0)) | tie_q2
    imag_pos = (imag_dominant & (w_imag >= 0)) | tie_q1
    imag_neg = (imag_dominant & (w_imag < 0)) | tie_q3
    return real_pos, real_neg, imag_pos, imag_neg


def _quantize_phase_axis(w_real: torch.Tensor, w_imag: torch.Tensor) -> TensorPair:
    """Axial phase quantization to {+/-s_real, 0} or {0, +/-s_imag}."""
    real_pos, real_neg, imag_pos, imag_neg = _axis_masks(w_real, w_imag)

    real_mask = real_pos | real_neg
    imag_mask = imag_pos | imag_neg

    real_mean = _safe_mean_abs(w_real, real_mask)
    imag_mean = _safe_mean_abs(w_imag, imag_mask)

    qw_real = torch.zeros_like(w_real)
    qw_imag = torch.zeros_like(w_imag)
    qw_real = torch.where(real_pos, real_mean, qw_real)
    qw_real = torch.where(real_neg, -real_mean, qw_real)
    qw_imag = torch.where(imag_pos, imag_mean, qw_imag)
    qw_imag = torch.where(imag_neg, -imag_mean, qw_imag)
    return qw_real, qw_imag


def _quantize_sign_phase(w_real: torch.Tensor, w_imag: torch.Tensor) -> TensorPair:
    """Quadrant sign quantization using mean absolute scales for real and imaginary parts."""
    real_scale = w_real.abs().mean().clamp_min(EPS)
    imag_scale = w_imag.abs().mean().clamp_min(EPS)

    qw_real = torch.where(w_real >= 0, real_scale, -real_scale).to(w_real.dtype)
    qw_imag = torch.where(w_imag >= 0, imag_scale, -imag_scale).to(w_imag.dtype)
    return qw_real, qw_imag


def _residual_quantize(
    w_real: torch.Tensor,
    w_imag: torch.Tensor,
    steps: int,
    base_quantizer: Callable[[torch.Tensor, torch.Tensor], TensorPair],
) -> TensorPair:
    """
    Generic residual quantization framework.

    Each step quantizes the current residual and accumulates the result,
    reducing quantization error through a multi-step approximation.
    """
    q_real = torch.zeros_like(w_real)
    q_imag = torch.zeros_like(w_imag)
    residual_real = w_real
    residual_imag = w_imag

    for _ in range(steps):
        step_real, step_imag = base_quantizer(residual_real, residual_imag)
        q_real = q_real + step_real
        q_imag = q_imag + step_imag
        residual_real = residual_real - step_real
        residual_imag = residual_imag - step_imag

    return q_real, q_imag


class PhaseQuantSTE(torch.autograd.Function):
    """iFairy 2-bit axial complex quantization with straight-through gradients."""

    @staticmethod
    def forward(ctx, w_real: torch.Tensor, w_imag: torch.Tensor) -> TensorPair:
        return _quantize_phase_axis(w_real, w_imag)

    @staticmethod
    def backward(ctx, grad_real: torch.Tensor, grad_imag: torch.Tensor) -> TensorPair:
        return grad_real, grad_imag


class PhaseQuantSTE_V2(torch.autograd.Function):
    """Two-step residual axial quantization."""

    @staticmethod
    def forward(ctx, w_real: torch.Tensor, w_imag: torch.Tensor) -> TensorPair:
        return _residual_quantize(w_real, w_imag, steps=2, base_quantizer=_quantize_phase_axis)

    @staticmethod
    def backward(ctx, grad_real: torch.Tensor, grad_imag: torch.Tensor) -> TensorPair:
        return grad_real, grad_imag


class PhaseQuantSTE_V3(torch.autograd.Function):
    """Three-step residual axial quantization."""

    @staticmethod
    def forward(ctx, w_real: torch.Tensor, w_imag: torch.Tensor) -> TensorPair:
        return _residual_quantize(w_real, w_imag, steps=3, base_quantizer=_quantize_phase_axis)

    @staticmethod
    def backward(ctx, grad_real: torch.Tensor, grad_imag: torch.Tensor) -> TensorPair:
        return grad_real, grad_imag


class PhaseQuantSTE_V4(torch.autograd.Function):
    """Four-step residual axial quantization."""

    @staticmethod
    def forward(ctx, w_real: torch.Tensor, w_imag: torch.Tensor) -> TensorPair:
        return _residual_quantize(w_real, w_imag, steps=4, base_quantizer=_quantize_phase_axis)

    @staticmethod
    def backward(ctx, grad_real: torch.Tensor, grad_imag: torch.Tensor) -> TensorPair:
        return grad_real, grad_imag


class NewPhaseQuantSTE(torch.autograd.Function):
    """Quadrant sign quantization with separate global real and imaginary scales."""

    @staticmethod
    def forward(ctx, w_real: torch.Tensor, w_imag: torch.Tensor) -> TensorPair:
        return _quantize_sign_phase(w_real, w_imag)

    @staticmethod
    def backward(ctx, grad_w_real: torch.Tensor, grad_w_imag: torch.Tensor) -> TensorPair:
        return grad_w_real, grad_w_imag


class NewPhaseQuantSTE_V2(torch.autograd.Function):
    """Two-step residual quadrant sign quantization."""

    @staticmethod
    def forward(ctx, w_real: torch.Tensor, w_imag: torch.Tensor) -> TensorPair:
        return _residual_quantize(w_real, w_imag, steps=2, base_quantizer=_quantize_sign_phase)

    @staticmethod
    def backward(ctx, grad_real: torch.Tensor, grad_imag: torch.Tensor) -> TensorPair:
        return grad_real, grad_imag


class LearnableGroupedPhaseQuant(nn.Module):
    """
    Learnable grouped complex phase quantizer.

    Features:
    1) group-wise rotation to align distribution directions
    2) learnable phase codebook for directional discretization
    3) least-squares alpha scaling to fit magnitude
    4) lightweight residual sign quantization for extra error compensation

    Implementation:
    - group rows and rotate each group
    - normalize rotated vectors and match them to the codebook by cosine similarity
    - compute group-level alpha scale with least squares for the selected codeword
    - optionally add one residual sign-quantization step with beta
    - rotate back and use STE to preserve input gradients
    """

    def __init__(
        self,
        n_rows: int,
        group_size: int = 32,
        n_codewords: int = 4,
        enable_rotation: bool = True,
        use_residual: bool = True,
        init_mode: str = "quadrant",
    ):
        super().__init__()
        self.n_rows = n_rows
        self.group_size = group_size
        self.n_codewords = n_codewords
        self.enable_rotation = enable_rotation
        self.use_residual = use_residual

        self.num_groups = math.ceil(n_rows / group_size)

        if init_mode == "quadrant":
            init_phases = torch.tensor(
                [math.pi / 4, 3 * math.pi / 4, -3 * math.pi / 4, -math.pi / 4],
                dtype=torch.float32,
            )
        elif init_mode == "axis":
            init_phases = torch.tensor(
                [0.0, math.pi / 2, math.pi, -math.pi / 2],
                dtype=torch.float32,
            )
        else:
            raise ValueError(f"Unsupported init_mode: {init_mode}")

        if n_codewords != 4:
            init_phases = torch.linspace(-math.pi, math.pi, steps=n_codewords + 1)[:-1]

        self.phase_angles = nn.Parameter(init_phases)
        if self.enable_rotation:
            self.rotation_angles = nn.Parameter(torch.zeros(self.num_groups, 1, 1))
        else:
            self.register_parameter("rotation_angles", None)

    def _group_pair_rows(
        self, w_real: torch.Tensor, w_imag: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """Group rows with padding and return real/imag groups, valid mask, and original rows."""
        if w_real.shape != w_imag.shape:
            raise ValueError("w_real and w_imag must have the same shape.")

        rows, cols = w_real.shape
        pad_rows = (self.group_size - rows % self.group_size) % self.group_size
        mask = w_real.new_ones(rows, 1)

        if pad_rows:
            w_real = F.pad(w_real, (0, 0, 0, pad_rows))
            w_imag = F.pad(w_imag, (0, 0, 0, pad_rows))
            mask = F.pad(mask, (0, 0, 0, pad_rows))

        groups = w_real.shape[0] // self.group_size
        return (
            w_real.reshape(groups, self.group_size, cols),
            w_imag.reshape(groups, self.group_size, cols),
            mask.reshape(groups, self.group_size, 1),
            rows,
        )

    @staticmethod
    def _ungroup_rows(x_grouped: torch.Tensor, orig_rows: int) -> torch.Tensor:
        """Restore grouped tensors to the original 2D shape and remove padding."""
        cols = x_grouped.shape[-1]
        return x_grouped.reshape(-1, cols)[:orig_rows]

    def _build_codebook(self) -> torch.Tensor:
        """Build unit-circle codebook directions from learnable phase_angles."""
        return torch.stack((torch.cos(self.phase_angles), torch.sin(self.phase_angles)), dim=-1)

    def forward(self, w_real: torch.Tensor, w_imag: torch.Tensor) -> TensorPair:
        # 1) Group rows and pad.
        wr, wi, mask, orig_rows = self._group_pair_rows(w_real, w_imag)

        # 2) Optionally rotate groups to improve directional quantization fit.
        if self.enable_rotation:
            cos_t = torch.cos(self.rotation_angles)
            sin_t = torch.sin(self.rotation_angles)
            wr_rot = wr * cos_t + wi * sin_t
            wi_rot = wi * cos_t - wr * sin_t
        else:
            cos_t = sin_t = None
            wr_rot, wi_rot = wr, wi

        # 3) Normalize and select the closest codebook direction by cosine similarity.
        inv_mag = torch.rsqrt((wr_rot.square() + wi_rot.square()).clamp_min(EPS))
        norm_real = wr_rot * inv_mag
        norm_imag = wi_rot * inv_mag

        codebook = self._build_codebook()
        codebook_real = codebook[:, 0].view(1, 1, 1, -1)
        codebook_imag = codebook[:, 1].view(1, 1, 1, -1)

        similarity = norm_real.unsqueeze(-1) * codebook_real + norm_imag.unsqueeze(-1) * codebook_imag
        code_idx = similarity.argmax(dim=-1)
        selected = codebook[code_idx]
        qdir_real = selected[..., 0]
        qdir_imag = selected[..., 1]

        # 4) Fit the main quantized component magnitude with group-level LS scaling.
        num = ((wr_rot * qdir_real + wi_rot * qdir_imag) * mask).sum(dim=(1, 2), keepdim=True)
        den = (((qdir_real.square() + qdir_imag.square()) * mask).sum(dim=(1, 2), keepdim=True)).clamp_min(EPS)
        alpha = num / den
        q1_real = alpha * qdir_real
        q1_imag = alpha * qdir_imag

        # 5) Optional residual compensation with signs and group-level magnitudes.
        if self.use_residual:
            residual_real = wr_rot - q1_real
            residual_imag = wi_rot - q1_imag
            valid_count = (mask.sum(dim=(1, 2), keepdim=True) * residual_real.shape[-1]).clamp_min(1.0)
            beta_real = (residual_real.abs() * mask).sum(dim=(1, 2), keepdim=True) / valid_count
            beta_imag = (residual_imag.abs() * mask).sum(dim=(1, 2), keepdim=True) / valid_count

            q2_real = beta_real * torch.where(residual_real >= 0, 1.0, -1.0).to(residual_real.dtype)
            q2_imag = beta_imag * torch.where(residual_imag >= 0, 1.0, -1.0).to(residual_imag.dtype)
            q_real_rot = q1_real + q2_real
            q_imag_rot = q1_imag + q2_imag
        else:
            q_real_rot = q1_real
            q_imag_rot = q1_imag

        # 6) Rotate back to the original coordinate system.
        if self.enable_rotation:
            q_real = q_real_rot * cos_t - q_imag_rot * sin_t
            q_imag = q_imag_rot * cos_t + q_real_rot * sin_t
        else:
            q_real, q_imag = q_real_rot, q_imag_rot

        # 7) Restore the grouped shape.
        q_real = self._ungroup_rows(q_real, orig_rows)
        q_imag = self._ungroup_rows(q_imag, orig_rows)

        # 8) STE: pass input weight gradients through while retaining quantizer parameter gradients.
        q_real = q_real + (w_real - w_real.detach())
        q_imag = q_imag + (w_imag - w_imag.detach())
        return q_real, q_imag


# Publicly supported quantization method names.
SUPPORTED_QUANT_METHODS = (
    "complex_phase_v1",
    "complex_phase_v2",
    "complex_phase_v3",
    "complex_phase_v4",
    "complex_phase_new_v2",
    "complex_phase_next_v1",
)

# Functional quantizer map, excluding the module-based next_v1 path.
_FUNCTION_QUANTIZERS: Dict[str, torch.autograd.Function] = {
    "complex_phase_v1": PhaseQuantSTE,
    "complex_phase_v2": PhaseQuantSTE_V2,
    "complex_phase_v3": PhaseQuantSTE_V3,
    "complex_phase_v4": PhaseQuantSTE_V4,
    "complex_phase_new_v2": NewPhaseQuantSTE_V2,
}


class ComplexWeightQuantizer(nn.Module):
    """
    Unified weight quantization entrypoint.

    - legacy methods call autograd.Function.apply
    - next_v1 calls the modular LearnableGroupedPhaseQuant path
    - optional global rotation applies a 2D rotate/unrotate around quantization
    """

    def __init__(
        self,
        quant_method: str = "complex_phase_v1",
        enable_rotation: bool = False,
        n_rows: int | None = None,
        group_size: int = 32,
    ):
        super().__init__()
        self.quant_method = quant_method
        self.enable_rotation = enable_rotation
        self.is_module_quantizer = quant_method == "complex_phase_next_v1"

        if quant_method not in SUPPORTED_QUANT_METHODS:
            raise ValueError(f"Unsupported quantization method: {quant_method}")

        if self.is_module_quantizer:
            if n_rows is None:
                raise ValueError("complex_phase_next_v1 requires n_rows.")
            self.quant_fn = LearnableGroupedPhaseQuant(
                n_rows=n_rows,
                group_size=group_size,
                n_codewords=4,
                enable_rotation=enable_rotation,
                use_residual=True,
                init_mode="quadrant",
            )
            self.register_parameter("rotation_angle", None)
            return

        self.quant_fn = _FUNCTION_QUANTIZERS[quant_method]
        if enable_rotation:
            self.rotation_angle = nn.Parameter(torch.zeros(1))
        else:
            self.register_parameter("rotation_angle", None)

    def forward(self, w_real: torch.Tensor, w_imag: torch.Tensor) -> TensorPair:
        if self.is_module_quantizer:
            return self.quant_fn(w_real, w_imag)

        if not self.enable_rotation:
            return self.quant_fn.apply(w_real, w_imag)

        # Global rotation helps weight directions fit the discrete codewords.
        cos_theta = torch.cos(self.rotation_angle)
        sin_theta = torch.sin(self.rotation_angle)

        w_real_rot = w_real * cos_theta + w_imag * sin_theta
        w_imag_rot = w_imag * cos_theta - w_real * sin_theta
        qw_real_rot, qw_imag_rot = self.quant_fn.apply(w_real_rot, w_imag_rot)

        qw_real = qw_real_rot * cos_theta - qw_imag_rot * sin_theta
        qw_imag = qw_imag_rot * cos_theta + qw_real_rot * sin_theta
        return qw_real, qw_imag


class ActivationQuantSTE(torch.autograd.Function):
    """Symmetric activation quantization over the last-dimension dynamic range."""

    @staticmethod
    def forward(ctx, x_real: torch.Tensor, x_imag: torch.Tensor, num_bits: int) -> TensorPair:
        # 1. Compute quantization extrema from the requested bit width.
        qmax = (1 << (num_bits - 1)) - 1  # 7 for 4-bit.
        qmin = -(1 << (num_bits - 1))     # -8 for 4-bit.

        # 2. Compute scale factors from qmax.
        real_scale = float(qmax) / x_real.abs().amax(dim=-1, keepdim=True).clamp_min(1e-5)
        imag_scale = float(qmax) / x_imag.abs().amax(dim=-1, keepdim=True).clamp_min(1e-5)
        
        # 3. Quantize, clamp, and dequantize.
        qx_real = (x_real * real_scale).round().clamp(qmin, qmax) / real_scale
        qx_imag = (x_imag * imag_scale).round().clamp(qmin, qmax) / imag_scale
        
        return qx_real, qx_imag

    @staticmethod
    def backward(ctx, grad_real: torch.Tensor, grad_imag: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, None]:
        # num_bits is not differentiable, so return None for its placeholder gradient.
        return grad_real, grad_imag, None


class ComplexActivationQuantizer(nn.Module):
    """Activation quantizer wrapper for optional use inside QATLinear."""
    
    def __init__(self, num_bits: int):
        super().__init__()
        self.num_bits = num_bits

    def forward(self, x_real: torch.Tensor, x_imag: torch.Tensor) -> TensorPair:
        # Pass the bit-width parameter to the underlying autograd function.
        return ActivationQuantSTE.apply(x_real, x_imag, self.num_bits)


class QATLinearComplexPhase(nn.Linear):
    """
    Complex-structured QAT linear layer.
    ...
    """

    def __init__(
        self,
        quant_method_u: str = "complex_phase_v1",
        quant_method_w: str = "complex_phase_v1",
        enable_act_quant: bool = True,
        act_num_bits: int = 4,       
        enable_rotation: bool = False,
        quant_group_size: int = 32,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if self.in_features % 2 != 0 or self.out_features % 2 != 0:
            raise ValueError("Complex-phase QAT requires even in/out features.")

        self.enable_act_quant = enable_act_quant
        n_complex_rows = self.out_features // 2

        self.quantizer_u = ComplexWeightQuantizer(
            quant_method=quant_method_u,
            enable_rotation=enable_rotation,
            n_rows=n_complex_rows,
            group_size=quant_group_size,
        )
        self.quantizer_w = ComplexWeightQuantizer(
            quant_method=quant_method_w,
            enable_rotation=enable_rotation,
            n_rows=n_complex_rows,
            group_size=quant_group_size,
        )
        self.act_quantizer = ComplexActivationQuantizer(num_bits=act_num_bits) if self.enable_act_quant else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1) Split weights into 2x2 real-valued blocks with complex structure.
        half_out = self.out_features // 2
        half_in = self.in_features // 2
        weight = self.weight

        a11 = weight[:half_out, :half_in]
        a12 = weight[:half_out, half_in:]
        a21 = weight[half_out:, :half_in]
        a22 = weight[half_out:, half_in:]

        u_real = 0.5 * (a11 + a22)
        u_imag = 0.5 * (a21 - a12)
        w_real = 0.5 * (a11 - a22)
        w_imag = 0.5 * (a12 + a21)

        # 2) Quantize U/W components independently, possibly with different methods.
        u_real_q, u_imag_q = self.quantizer_u(u_real, u_imag)
        w_real_q, w_imag_q = self.quantizer_w(w_real, w_imag)

        # 3) Reassemble the quantized linear-layer weight matrix.
        a11_q = w_real_q + u_real_q
        a12_q = w_imag_q - u_imag_q
        a21_q = w_imag_q + u_imag_q
        a22_q = -w_real_q + u_real_q
        q_weight = torch.cat(
            (
                torch.cat((a11_q, a12_q), dim=1),
                torch.cat((a21_q, a22_q), dim=1),
            ),
            dim=0,
        )

        # 4) Optionally quantize input activations with separate real/imag parts.
        if self.act_quantizer is not None:
            x_real = x[..., :half_in]
            x_imag = x[..., half_in:]
            qx_real, qx_imag = self.act_quantizer(x_real, x_imag)
            x = torch.cat((qx_real, qx_imag), dim=-1)

        # 5) Linear mapping.
        return F.linear(x, q_weight, self.bias)
