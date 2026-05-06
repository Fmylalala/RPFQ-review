import torch
import torch.nn as nn
import torch.nn.functional as F
from .quant_i import NewPhaseQuantSTE_V2,ComplexActivationQuantizer
from torch.library import Library
from pathlib import Path

try:
    import triton
    import triton.language as tl
    _TRITON_IMPORT_ERROR = None
except ModuleNotFoundError as exc:
    _TRITON_IMPORT_ERROR = exc

    class _MissingTritonTesting:
        @staticmethod
        def Benchmark(*args, **kwargs):
            return {"args": args, "kwargs": kwargs}

        @staticmethod
        def perf_report(configs):
            def decorator(fn):
                def wrapped(*args, **kwargs):
                    _require_triton()
                    return fn(*args, **kwargs)

                wrapped.run = lambda *args, **kwargs: _require_triton()
                return wrapped

            return decorator

        @staticmethod
        def do_bench(*args, **kwargs):
            _require_triton()

    class _MissingTriton:
        testing = _MissingTritonTesting()

        @staticmethod
        def Config(*args, **kwargs):
            return {"args": args, "kwargs": kwargs}

        @staticmethod
        def autotune(*args, **kwargs):
            def decorator(fn):
                return fn

            return decorator

        @staticmethod
        def jit(fn=None, **kwargs):
            if fn is None:
                return lambda wrapped: wrapped
            return fn

        @staticmethod
        def cdiv(a, b):
            return (a + b - 1) // b

        @staticmethod
        def next_power_of_2(x):
            return 1 << (x - 1).bit_length()

    class _MissingTritonLanguage:
        constexpr = object

        def __getattr__(self, name):
            _require_triton()

    triton = _MissingTriton()
    tl = _MissingTritonLanguage()


def _require_triton() -> None:
    if _TRITON_IMPORT_ERROR is not None:
        raise RuntimeError(
            "quant_models.kernel is an optional Triton/CUDA benchmark module. "
            "Install Triton in a CUDA-capable environment to use it."
        ) from _TRITON_IMPORT_ERROR
def get_cuda_autotune_config():
    return [
        triton.Config(
            {'BLOCK_SIZE':M},num_stages=s, num_warps=w
        )
        for M in [32,64,128]
        for s in [1,2,3,4,5]
        for w in [4,8,16]
    ]
    



@triton.autotune(
    configs=get_cuda_autotune_config(),
    key=['M','N'],
    reset_to_zero=['global_accum_ptr'] 
)  
@triton.jit
def weight_quant_global_kernel(
    real_ptr, imag_ptr, 
    out_real_ptr, out_imag_ptr, 
    # Buffer for accumulating global sum and count values (four scalars).
    global_accum_ptr, 
    M, N,
    stride_wm, stride_wn,
    SM_NUM :  tl.constexpr ,
    BLOCK_SIZE: tl.constexpr 
):
    # 1. Locate the block.
    SM_pid = tl.program_id(0)
    TASK_M = tl.cdiv(M,BLOCK_SIZE)
    TASK_N =  tl.cdiv(N,BLOCK_SIZE)
    TOTAL_TASK = TASK_N * TASK_M
    local_real_sum = 0.0
    local_real_count = 0.0
    local_imag_sum = 0.0
    local_imag_count = 0.0

    for pid in range(SM_pid,TOTAL_TASK,SM_NUM):
        pid_m = pid // TASK_N
        pid_n = pid % TASK_N
        
        
        real_block_ptr = tl.make_block_ptr(real_ptr,
                                           shape=[M,N],
                                           strides=[stride_wm,stride_wn],
                                           offsets=[pid_m*BLOCK_SIZE,pid_n*BLOCK_SIZE],
                                           block_shape=[BLOCK_SIZE,BLOCK_SIZE],
                                           order=[1,0])
        imag_block_ptr = tl.make_block_ptr(imag_ptr,
                                           shape=[M,N],
                                           strides=[stride_wm,stride_wn],
                                           offsets=[pid_m*BLOCK_SIZE,pid_n*BLOCK_SIZE],
                                           block_shape=[BLOCK_SIZE,BLOCK_SIZE],
                                           order=[1,0])
        
        

        # 2. Load values and evaluate geometry.
        real = tl.load(real_block_ptr)
        imag = tl.load(imag_block_ptr)
        
        # Convert to float32 for comparisons to avoid bf16 (i16) type issues.
        abs_real = tl.abs(real.to(tl.float32))
        abs_imag = tl.abs(imag.to(tl.float32))
        is_real_dominant = abs_real > abs_imag

        # 3. Quantize and store (1, -1, 0).
        # Comparisons also need conversion to float32.
        real_fp32 = real.to(tl.float32)
        imag_fp32 = imag.to(tl.float32)
        out_real = tl.where(is_real_dominant, tl.where(real_fp32 >= 0.0, 1.0, -1.0), 0.0)
        out_imag = tl.where(is_real_dominant, 0.0, tl.where(imag_fp32 >= 0.0, 1.0, -1.0))
        

        
        
        
        out_real_block_ptr = tl.make_block_ptr(out_real_ptr,
                                           shape=[M,N],
                                           strides=[stride_wm,stride_wn],
                                           offsets=[pid_m*BLOCK_SIZE,pid_n*BLOCK_SIZE],
                                           block_shape=[BLOCK_SIZE,BLOCK_SIZE],
                                           order=[1,0])
        out_imag_block_ptr = tl.make_block_ptr(out_imag_ptr,
                                           shape=[M,N],
                                           strides=[stride_wm,stride_wn],
                                           offsets=[pid_m*BLOCK_SIZE,pid_n*BLOCK_SIZE],
                                           block_shape=[BLOCK_SIZE,BLOCK_SIZE],
                                           order=[1,0])
        
        

        
        tl.store(out_real_block_ptr, out_real.to(out_real_ptr.dtype.element_ty))
        tl.store(out_imag_block_ptr, out_imag.to(out_imag_ptr.dtype.element_ty))

        # 4. Compute local contribution for this block.
        local_real_count += tl.sum(is_real_dominant.to(tl.float32))
        
        # Zero non-dominant real absolute values and keep the result in bf16.
        real_abs_bf16 = tl.where(is_real_dominant, abs_real, 0.0).to(tl.float32)
        local_real_sum += tl.sum(real_abs_bf16)
        
        # Imaginary-part statistics.
        is_imag_dominant = (tl.abs(out_imag) > 0.0)
        local_imag_count += tl.sum(is_imag_dominant.to(tl.float32))
        
        # Zero non-dominant imaginary absolute values and keep the result in bf16.
        imag_abs_bf16 = tl.where(is_imag_dominant, abs_imag, 0.0).to(tl.float32)
        local_imag_sum += tl.sum(imag_abs_bf16)

    # 5. Atomically accumulate into global memory.
    # global_accum_ptr is expected to point to an array of length 4:
    # [sum_real, count_real, sum_imag, count_imag]
    tl.atomic_add(global_accum_ptr + 0, local_real_sum)
    tl.atomic_add(global_accum_ptr + 1, local_real_count)
    tl.atomic_add(global_accum_ptr + 2, local_imag_sum)
    tl.atomic_add(global_accum_ptr + 3, local_imag_count)

def weight_quantizer(x_real, x_imag):
    _require_triton()
    M, N = x_real.shape

    
    # Output quantization result.
    out_real = torch.empty_like(x_real, dtype=torch.bfloat16)
    out_imag = torch.empty_like(x_imag, dtype=torch.bfloat16)
    
    # Prepare a global accumulator initialized to 0.
    # These four locations need to be allocated on the GPU.
    accum = torch.zeros(4, device=x_real.device, dtype=torch.float32)
    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count
    grid = (NUM_SMS,)
    weight_quant_global_kernel[grid](
        x_real, x_imag, out_real, out_imag,
        accum, 
        M, N, 
        x_real.stride(0), x_real.stride(1),
        NUM_SMS,
    )
    
    # Compute final averages on the host.
    # This introduces a very small synchronization while waiting for the kernel.
    scale_real = (accum[0] / max(accum[1], 1e-6)).to(torch.bfloat16)
    scale_imag = (accum[2] / max(accum[3], 1e-6)).to(torch.bfloat16)
    
    return out_real*scale_real, out_imag*scale_imag





@triton.autotune(
    configs=get_cuda_autotune_config(),
    key=['M','N'],
)  
@triton.jit
def fairytoi_split_kernel(A_ptr,U_re_ptr,U_im_ptr,W_re_ptr,W_im_ptr,A_row_stride,A_col_stride,M, N,SM_NUM :  tl.constexpr,BLOCK_SIZE: tl.constexpr ):
# A is a 2M x 2N matrix.
    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE)
    TOTAL_TASK = num_pid_m * num_pid_n    


    for pid in range(start_pid,TOTAL_TASK,SM_NUM):
        row_pid = pid // num_pid_n 
        col_pid = pid % num_pid_n
        
#A11, A12 = A[:n, :m], A[:n, m:]
#A21, A22 = A[n:, :m], A[n:, m:]

        A11_ptr = tl.make_block_ptr(A_ptr,shape=[2*M,2*N],
                                    strides=[A_row_stride,A_col_stride],
                                    offsets=[row_pid*BLOCK_SIZE,col_pid*BLOCK_SIZE],
                                    block_shape=[BLOCK_SIZE,BLOCK_SIZE],order=[1,0])
        A12_ptr = tl.make_block_ptr(A_ptr,shape=[2*M,2*N],
                                    strides=[A_row_stride,A_col_stride],
                                    offsets=[row_pid*BLOCK_SIZE,col_pid*BLOCK_SIZE+N],
                                    block_shape=[BLOCK_SIZE,BLOCK_SIZE],order=[1,0])
        
        A21_ptr = tl.make_block_ptr(A_ptr,shape=[2*M,2*N],
                                    strides=[A_row_stride,A_col_stride],
                                    offsets=[row_pid*BLOCK_SIZE+M,col_pid*BLOCK_SIZE],
                                    block_shape=[BLOCK_SIZE,BLOCK_SIZE],order=[1,0])
        
        A22_ptr = tl.make_block_ptr(A_ptr,shape=[2*M,2*N],
                                    strides=[A_row_stride,A_col_stride],
                                    offsets=[row_pid*BLOCK_SIZE+M,col_pid*BLOCK_SIZE+N],
                                    block_shape=[BLOCK_SIZE,BLOCK_SIZE],order=[1,0])
        
        
        A11 = tl.load(A11_ptr)
        A12 = tl.load(A12_ptr)
        A21 = tl.load(A21_ptr)  
        A22 = tl.load(A22_ptr) 
        
        
        U_re = 0.5 * (A11 + A22)
        U_im = 0.5 * (A21 - A12)
        W_re = 0.5 * (A11 - A22)
        W_im = 0.5 * (A12 + A21)     

        U_row_stride = A_row_stride//2
        U1_ptr = tl.make_block_ptr(U_re_ptr,shape=[M,N],
                                    strides=[U_row_stride,A_col_stride],
                                    offsets=[row_pid*BLOCK_SIZE,col_pid*BLOCK_SIZE],
                                    block_shape=[BLOCK_SIZE,BLOCK_SIZE],order=[1,0])
        U2_ptr = tl.make_block_ptr(U_im_ptr,shape=[M,N],
                                    strides=[U_row_stride,A_col_stride],
                                    offsets=[row_pid*BLOCK_SIZE,col_pid*BLOCK_SIZE],
                                    block_shape=[BLOCK_SIZE,BLOCK_SIZE],order=[1,0])
        
        W1_ptr = tl.make_block_ptr(W_re_ptr,shape=[M,N],
                                    strides=[U_row_stride,A_col_stride],
                                    offsets=[row_pid*BLOCK_SIZE,col_pid*BLOCK_SIZE],
                                    block_shape=[BLOCK_SIZE,BLOCK_SIZE],order=[1,0])
        
        W2_ptr = tl.make_block_ptr(W_im_ptr,shape=[M,N],
                                    strides=[U_row_stride,A_col_stride],
                                    offsets=[row_pid*BLOCK_SIZE,col_pid*BLOCK_SIZE],
                                    block_shape=[BLOCK_SIZE,BLOCK_SIZE],order=[1,0])
        
                        
        tl.store(U1_ptr, U_re.to(tl.bfloat16))
        tl.store(U2_ptr, U_im.to(tl.bfloat16))
        tl.store(W1_ptr, W_re.to(tl.bfloat16))
        tl.store(W2_ptr, W_im.to(tl.bfloat16))
        
        
        



@triton.autotune(
    configs=get_cuda_autotune_config(),
    key=['M','N'],
)  
@triton.jit
def fairytoi_combine_kernel(B_ptr,U_re_ptr,U_im_ptr,W_re_ptr,W_im_ptr,U_res_re_ptr,U_res_im_ptr,W_res_re_ptr,W_res_im_ptr,A_row_stride,A_col_stride,M, N,SM_NUM :  tl.constexpr,BLOCK_SIZE: tl.constexpr ):
# A is a 2M x 2N matrix.
    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE)
    TOTAL_TASK = num_pid_m * num_pid_n    


    for pid in range(start_pid,TOTAL_TASK,SM_NUM):
        row_pid = pid // num_pid_n 
        col_pid = pid % num_pid_n
        
#A11, A12 = A[:n, :m], A[:n, m:]
#A21, A22 = A[n:, :m], A[n:, m:]
        U_row_stride = A_row_stride // 2
        U1_ptr = tl.make_block_ptr(U_re_ptr,shape=[M,N],
                                    strides=[U_row_stride,A_col_stride],
                                    offsets=[row_pid*BLOCK_SIZE,col_pid*BLOCK_SIZE],
                                    block_shape=[BLOCK_SIZE,BLOCK_SIZE],order=[1,0])
        U2_ptr = tl.make_block_ptr(U_im_ptr,shape=[M,N],
                                    strides=[U_row_stride,A_col_stride],
                                    offsets=[row_pid*BLOCK_SIZE,col_pid*BLOCK_SIZE],
                                    block_shape=[BLOCK_SIZE,BLOCK_SIZE],order=[1,0])
        
        W1_ptr = tl.make_block_ptr(W_re_ptr,shape=[M,N],
                                    strides=[U_row_stride,A_col_stride],
                                    offsets=[row_pid*BLOCK_SIZE,col_pid*BLOCK_SIZE],
                                    block_shape=[BLOCK_SIZE,BLOCK_SIZE],order=[1,0])
        
        W2_ptr = tl.make_block_ptr(W_im_ptr,shape=[M,N],
                                    strides=[U_row_stride,A_col_stride],
                                    offsets=[row_pid*BLOCK_SIZE,col_pid*BLOCK_SIZE],
                                    block_shape=[BLOCK_SIZE,BLOCK_SIZE],order=[1,0])
        
        res_U1_ptr = tl.make_block_ptr(U_res_re_ptr,shape=[M,N],
                                    strides=[U_row_stride,A_col_stride],
                                    offsets=[row_pid*BLOCK_SIZE,col_pid*BLOCK_SIZE],
                                    block_shape=[BLOCK_SIZE,BLOCK_SIZE],order=[1,0])
        res_U2_ptr = tl.make_block_ptr(U_res_im_ptr,shape=[M,N],
                                    strides=[U_row_stride,A_col_stride],
                                    offsets=[row_pid*BLOCK_SIZE,col_pid*BLOCK_SIZE],
                                    block_shape=[BLOCK_SIZE,BLOCK_SIZE],order=[1,0])
        
        res_W1_ptr = tl.make_block_ptr(W_res_re_ptr,shape=[M,N],
                                    strides=[U_row_stride,A_col_stride],
                                    offsets=[row_pid*BLOCK_SIZE,col_pid*BLOCK_SIZE],
                                    block_shape=[BLOCK_SIZE,BLOCK_SIZE],order=[1,0])
        
        res_W2_ptr = tl.make_block_ptr(W_res_im_ptr,shape=[M,N],
                                    strides=[U_row_stride,A_col_stride],
                                    offsets=[row_pid*BLOCK_SIZE,col_pid*BLOCK_SIZE],
                                    block_shape=[BLOCK_SIZE,BLOCK_SIZE],order=[1,0])
        
        # Load from a bf16 tensor and explicitly convert to fp32 to match B_ptr.
        U_out_real = tl.load(U1_ptr).to(tl.float32) + tl.load(res_U1_ptr).to(tl.float32)
        
        U_out_imag = tl.load(U2_ptr).to(tl.float32) + tl.load(res_U2_ptr).to(tl.float32)
           
        W_out_real = tl.load(W1_ptr).to(tl.float32) + tl.load(res_W1_ptr).to(tl.float32)
        
        W_out_imag = tl.load(W2_ptr).to(tl.float32) + tl.load(res_W2_ptr).to(tl.float32)
        
        # The result is now fp32 and can be written safely to B.
        A11_q = W_out_real + U_out_real
        A12_q = W_out_imag - U_out_imag
        A21_q = W_out_imag + U_out_imag
        A22_q = -W_out_real + U_out_real
        
        
        A11_ptr = tl.make_block_ptr(B_ptr,shape=[2*M,2*N],
                                    strides=[A_row_stride,A_col_stride],
                                    offsets=[row_pid*BLOCK_SIZE,col_pid*BLOCK_SIZE],
                                    block_shape=[BLOCK_SIZE,BLOCK_SIZE],order=[1,0])
        A12_ptr = tl.make_block_ptr(B_ptr,shape=[2*M,2*N],
                                    strides=[A_row_stride,A_col_stride],
                                    offsets=[row_pid*BLOCK_SIZE,col_pid*BLOCK_SIZE+N],
                                    block_shape=[BLOCK_SIZE,BLOCK_SIZE],order=[1,0])
        
        A21_ptr = tl.make_block_ptr(B_ptr,shape=[2*M,2*N],
                                    strides=[A_row_stride,A_col_stride],
                                    offsets=[row_pid*BLOCK_SIZE+M,col_pid*BLOCK_SIZE],
                                    block_shape=[BLOCK_SIZE,BLOCK_SIZE],order=[1,0])
        
        A22_ptr = tl.make_block_ptr(B_ptr,shape=[2*M,2*N],
                                    strides=[A_row_stride,A_col_stride],
                                    offsets=[row_pid*BLOCK_SIZE+M,col_pid*BLOCK_SIZE+N],
                                    block_shape=[BLOCK_SIZE,BLOCK_SIZE],order=[1,0])
                 
        tl.store(A11_ptr, A11_q)
        tl.store(A12_ptr, A12_q)
        tl.store(A21_ptr, A21_q)
        tl.store(A22_ptr, A22_q)
        
        
        
def fairytoi_quant_V2(A:torch.Tensor):
        _require_triton()
        assert A.is_contiguous(), "Input tensor must be contiguous."
        M,N = A.shape[0] // 2, A.shape[1] // 2
        
        A_row_stride,A_col_stride = A.stride()
        
        
        U_re = torch.empty([M,N],dtype=torch.bfloat16,device=A.device)
        U_im = torch.empty([M,N],dtype=torch.bfloat16,device=A.device)
        W_re = torch.empty([M,N],dtype=torch.bfloat16,device=A.device)
        W_im = torch.empty([M,N],dtype=torch.bfloat16,device=A.device) 
        B = torch.empty_like(A)
        
        NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count

        grid = (NUM_SMS,)
        
        fairytoi_split_kernel[grid](A,U_re,U_im,W_re,W_im,A_row_stride,A_col_stride,M,N,NUM_SMS)
        
        U_quant_re,U_quant_im = weight_quantizer(U_re,U_im)
        U_res_re,U_res_im = weight_quantizer(U_re-U_quant_re,U_im-U_quant_im)
        
        W_quant_re,W_quant_im = weight_quantizer(W_re,W_im)
        W_res_re,W_res_im = weight_quantizer(W_re-W_quant_re,W_im-W_quant_im)
        
        fairytoi_combine_kernel[grid](B,U_quant_re,U_quant_im,W_quant_re,W_quant_im,U_res_re,U_res_im,W_res_re,W_res_im,A_row_stride,A_col_stride,M,N,NUM_SMS)        
        
        return B
    
    
    

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 2}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 4}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 8}, num_warps=4, num_stages=5),
        triton.Config({'BLOCK_M': 16}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 32}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 64}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 128}, num_warps=8, num_stages=5),
    ],
    key=['M', 'N'],
)
@triton.jit
def _fused_split_quant_cat_kernel(
    x_ptr, y_ptr,
    M, N, HALF_N,
    stride_x_m, stride_x_n,
    stride_y_m, stride_y_n,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    OUTPUT_IS_BF16: tl.constexpr
):
    pid_m = tl.program_id(0)
    
    x_block_ptr = tl.make_block_ptr(
        base=x_ptr, shape=(M, N), strides=(stride_x_m, stride_x_n),
        offsets=(pid_m * BLOCK_M, 0), block_shape=(BLOCK_M, BLOCK_N), order=(1, 0)
    )
    y_block_ptr = tl.make_block_ptr(
        base=y_ptr, shape=(M, N), strides=(stride_y_m, stride_y_n),
        offsets=(pid_m * BLOCK_M, 0), block_shape=(BLOCK_M, BLOCK_N), order=(1, 0)
    )
    

    x = tl.load(x_block_ptr, boundary_check=(0, 1), padding_option="zero")
    

    cols = tl.arange(0, BLOCK_N)
    mask_real = cols < HALF_N
    mask_imag = (cols >= HALF_N) & (cols < N)
    

    mask_real_2d = tl.expand_dims(mask_real, 0)
    mask_imag_2d = tl.expand_dims(mask_imag, 0)
    

    abs_x = tl.abs(x)
    x_real_abs = tl.where(mask_real_2d, abs_x, 0.0)
    x_imag_abs = tl.where(mask_imag_2d, abs_x, 0.0)
    

    max_real = tl.maximum(tl.max(x_real_abs, axis=1), 1e-5)
    max_imag = tl.maximum(tl.max(x_imag_abs, axis=1), 1e-5)
    

    scale_real = tl.expand_dims(127.0 / max_real, 1) # shape: (BLOCK_M, 1)
    scale_imag = tl.expand_dims(127.0 / max_imag, 1) # shape: (BLOCK_M, 1)
    

    scale = tl.where(mask_real_2d, scale_real, scale_imag)
    
    qx = x.to(tl.float32) * scale.to(tl.float32)
    qx = tl.extra.cuda.libdevice.round(qx)

    
    qx = tl.maximum(qx, -128.0)
    qx = tl.minimum(qx, 127.0)

    y = qx / scale
    # Convert according to output type: bf16 outputs need conversion, fp32 outputs can be stored directly.
    if OUTPUT_IS_BF16:
        y = y.to(tl.bfloat16)
    tl.store(y_block_ptr, y, boundary_check=(0, 1))  
    
def ActivationQuantSTE(x):
    _require_triton()
    # Make non-contiguous tensors contiguous to avoid assertion failures.
    if not x.is_contiguous():
        x = x.contiguous()
    original_shape = x.shape
        
    # Dimension parsing.
    N = original_shape[-1]
    M = x.numel() // N
    HALF_N = N // 2
    
    BLOCK_N = triton.next_power_of_2(N)
    
    x_2d = x.view(M, N)
    # Allocate output memory once to save substantial memory bandwidth.
    y_2d = torch.empty_like(x_2d) 
    
    # Check output type and pass it to the kernel.
    output_is_bf16 = (y_2d.dtype == torch.bfloat16)
    
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), )
    
    # Launch one kernel to complete all operations.
    _fused_split_quant_cat_kernel[grid](
        x_2d, y_2d,
        M, N, HALF_N,
        x_2d.stride(0), x_2d.stride(1),
        y_2d.stride(0), y_2d.stride(1),
        BLOCK_N=BLOCK_N,
        OUTPUT_IS_BF16=output_is_bf16
    )
    
    return y_2d.view(original_shape)   
    
    
    
    
class QATLinearComplexPhaseV2_autograd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias):
        # Assume fairytoi_quant_V2 and ActivationQuantSTE are defined in global scope.
        x_quant = ActivationQuantSTE(x)
        w_quant = fairytoi_quant_V2(weight)
        
        # Store quantized tensors for the backward pass.
        ctx.save_for_backward(x_quant, w_quant)
        ctx.had_bias = bias is not None
        
        # Forward computation: [B, L, I] @ [O, I]^T -> [B, L, O].
        return F.linear(x_quant, w_quant, bias)

    @staticmethod
    def backward(ctx, grad_output):
        # grad_output: [Batch(b), Length(l), Out(o)]
        x_quant, w_quant = ctx.saved_tensors
        
        # Use einsum to compute gradients for 3D inputs.
        # 1. grad_weight [o, i] = sum over b, l of (grad_out[b, l, o] * x_quant[b, l, i])
        grad_weight = torch.einsum('blo, bli -> oi', grad_output, x_quant)
        
        # 2. grad_x [b, l, i] = grad_out[b, l, o] @ w_quant[o, i]
        grad_x = torch.einsum('blo, oi -> bli', grad_output, w_quant)
        
        # 3. grad_bias
        grad_bias = grad_output.sum(dim=(0, 1)) if ctx.had_bias else None
        
        return grad_x, grad_weight, grad_bias
    

def QATLinearComplexPhaseV2_forward(x, weight, bias):
    _require_triton()
    # Call the custom Autograd Function.
    return QATLinearComplexPhaseV2_autograd.apply(x, weight, bias)


_quant_lib = Library("quant", "DEF")
_quant_lib.define("QATLinearComplexPhaseV2_forward(Tensor x, Tensor weight, Tensor? bias) -> Tensor")
_quant_lib.impl(
    "QATLinearComplexPhaseV2_forward",
    QATLinearComplexPhaseV2_forward,
    "CompositeExplicitAutograd",
)


class QATLinearComplexPhaseV2(nn.Linear):
    """Complex-Phase V2 QAT linear layer (1-step residual)"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.in_features % 2 != 0 or self.out_features % 2 != 0:
            raise ValueError("Complex-Phase QAT requires even in/out features for Linear layers.")

    def forward(self, x):
        return torch.ops.quant.QATLinearComplexPhaseV2_forward(x,self.weight,self.bias)


def naive(A,x):
    _require_triton()
    act_quantizer = ComplexActivationQuantizer(num_bits=8)
    n, m = A.shape[0] // 2, A.shape[1] // 2
    
    A11, A12 = A[:n, :m], A[:n, m:]
    A21, A22 = A[n:, :m], A[n:, m:]
    
    U_re = 0.5 * (A11 + A22)
    U_im = 0.5 * (A21 - A12)
    W_re = 0.5 * (A11 - A22)
    W_im = 0.5 * (A12 + A21)
    
    U_re_q, U_im_q = NewPhaseQuantSTE_V2.apply(U_re, U_im)
    W_re_q, W_im_q = NewPhaseQuantSTE_V2.apply(W_re, W_im)
    
    A11_q = W_re_q + U_re_q
    A12_q = W_im_q - U_im_q
    A21_q = W_im_q + U_im_q
    A22_q = -W_re_q + U_re_q
    
    A_quant_top = torch.cat([A11_q, A12_q], dim=1)
    A_quant_bottom = torch.cat([A21_q, A22_q], dim=1)
    A_quant = torch.cat([A_quant_top, A_quant_bottom], dim=0)

    
    mid = x.shape[-1] // 2
    x_real, x_imag = x[..., :mid], x[..., mid:]
    qx_real, qx_imag =act_quantizer(x_real, x_imag)
    x_in = torch.cat([qx_real, qx_imag], dim=-1)

    return F.linear(x_in, A_quant)



DEVICE = "cuda"
configs = []
configs = [
    triton.testing.Benchmark(
        x_names=["M"],  
        x_vals=[256 * i for i in range(6, 33)],  
        line_arg="provider",  
        line_vals=["complie+","naive"],  
        line_names=["complie+","naive"],  
        styles=[("blue","-"),("green", "-")], 
        ylabel='MS',  
        plot_name="quant-global-performance-MS", 
        args={},  
    )
]
@triton.testing.perf_report(configs)
def benchmark_MS(M, provider):
    _require_triton()
    N =  M
    A_real = torch.randn((M, N), device=DEVICE, dtype=torch.bfloat16)
    B_real = torch.randn((M, N), device=DEVICE, dtype=torch.bfloat16)

    quantiles = [0.5, 0.2, 0.8]

    if provider == 'naive':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: naive(A_real,B_real), quantiles=quantiles) 
    if provider == 'complie+':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: QATLinearComplexPhaseV2_forward(A_real,B_real), quantiles=quantiles)
    perf = lambda ms: ms
    
    return perf(ms), perf(max_ms), perf(min_ms)











    
if __name__ == "__main__":
    benchmark_dir = Path(__file__).resolve().parents[1] / "results" / "benchmarks" / "complex_quant"
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    benchmark_MS.run(save_path=str(benchmark_dir),show_plots=True, print_data=True)
    
    '''DEVICE = 'cuda'

    A = torch.rand((4096,4096),device=DEVICE,dtype=torch.bfloat16) / 40
    X = QATLinearComplexPhaseV2_forward(A,A,A)
    Y = naive(A,A)
    print(X-Y)
    print("Output difference (Frobenius norm):", torch.norm(X - Y).item())
    print("="*30)
    print("Test completed successfully.")
    print("="*30)'''
