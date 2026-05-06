import Foundation
import MLX
import MLXFast
import MLXNN

struct RPFQViTBackboneDebugTensors {
    let patchEmbed: MLXArray
    let block0Trace: RPFQViTBlockTraceDebugTensors?
    let block0: MLXArray?
    let block1: MLXArray?
    let finalNorm: MLXArray
}

struct RPFQViTAttentionDebugTensors {
    let qkvConcat: MLXArray
    let q: MLXArray
    let k: MLXArray
    let v: MLXArray
    let attnOutPreProj: MLXArray
    let attnProj: MLXArray
}

struct RPFQViTBlockTraceDebugTensors {
    let blockInput: MLXArray
    let norm1: MLXArray
    let qkvConcat: MLXArray
    let q: MLXArray
    let k: MLXArray
    let v: MLXArray
    let attnOutPreProj: MLXArray
    let attnProj: MLXArray
    let residual1: MLXArray
    let blockOutput: MLXArray
}

struct RPFQViTForwardDebugTensors {
    let patchEmbed: MLXArray
    let block0Trace: RPFQViTBlockTraceDebugTensors?
    let block0: MLXArray?
    let block1: MLXArray?
    let finalNorm: MLXArray
    let logits: MLXArray
}

public class RPFQViTPhaseLinear: Module {
    private static let decodeMetalKernel: MLXFast.MLXFastKernel = MLXFast.metalKernel(
        name: "rpfq_vit_decode_linear",
        inputNames: ["xRe", "xIm", "codes", "gammaReal", "gammaImag"],
        outputNames: ["outRe", "outIm"],
        source: RPFQViTMetalKernels.decodeSource,
        ensureRowContiguous: true
    )

    private static let prefillMetalKernel: MLXFast.MLXFastKernel = MLXFast.metalKernel(
        name: "rpfq_vit_prefill_linear",
        inputNames: ["xRe", "xIm", "codes", "gammaReal", "gammaImag"],
        outputNames: ["outRe", "outIm"],
        source: RPFQViTMetalKernels.prefillSource,
        ensureRowContiguous: true
    )

    let inDim: Int
    let outDim: Int
    let blockSize: Int
    let blocksPerRow: Int
    let packedPerRow: Int
    let codebook10IsPosImag: Bool

    @ParameterInfo(key: "codes") var codes: MLXArray
    @ParameterInfo(key: "gamma_real") var gammaReal: MLXArray
    @ParameterInfo(key: "gamma_imag") var gammaImag: MLXArray

    public init(inDim: Int, outDim: Int, blockSize: Int, codebook10IsPosImag: Bool = false) {
        self.inDim = inDim
        self.outDim = outDim
        self.blockSize = blockSize
        self.blocksPerRow = inDim / blockSize
        self.packedPerRow = inDim / 4
        self.codebook10IsPosImag = codebook10IsPosImag

        self._codes.wrappedValue = MLXArray.zeros([outDim, packedPerRow], dtype: .uint8)
        self._gammaReal.wrappedValue = MLXArray.zeros([outDim, blocksPerRow], dtype: .float16)
        self._gammaImag.wrappedValue = MLXArray.zeros([outDim, blocksPerRow], dtype: .float16)
    }

    private func metalDecode(_ xRe: MLXArray, _ xIm: MLXArray) -> (MLXArray, MLXArray) {
        let tgSize = RPFQViTMetalKernels.threadgroupSize
        let rowsPerGroup = RPFQViTMetalKernels.rowsPerGroup
        let numGroups = (outDim + rowsPerGroup - 1) / rowsPerGroup

        let outputs = Self.decodeMetalKernel(
            [
                xRe.reshaped(inDim),
                xIm.reshaped(inDim),
                codes.reshaped(outDim * packedPerRow),
                gammaReal.reshaped(outDim * blocksPerRow),
                gammaImag.reshaped(outDim * blocksPerRow),
            ],
            template: [
                ("QK", blockSize),
                ("BLOCKS_PER_ROW", blocksPerRow),
                ("PACKED_PER_BLOCK", blockSize / 4),
                ("PACKED_PER_ROW", packedPerRow),
                ("CODEBOOK_10_POS_IMAG", codebook10IsPosImag),
                ("SIMD_WIDTH", RPFQViTMetalKernels.simdWidth),
                ("PACKED_PER_THREAD", RPFQViTMetalKernels.packedPerThread),
                ("ROWS_PER_GROUP", rowsPerGroup),
                ("TG_SIZE", tgSize),
                ("OUT_DIM", outDim),
            ],
            grid: (numGroups * tgSize, 1, 1),
            threadGroup: (tgSize, 1, 1),
            outputShapes: [[outDim], [outDim]],
            outputDTypes: [.float16, .float16],
            stream: .gpu
        )

        return (outputs[0].reshaped(1, 1, outDim), outputs[1].reshaped(1, 1, outDim))
    }

    private func metalPrefill(_ xRe: MLXArray, _ xIm: MLXArray) -> (MLXArray, MLXArray) {
        let tokens = xRe.dim(1)
        let tgSize = RPFQViTMetalKernels.threadgroupSize
        let rowsPerGroup = RPFQViTMetalKernels.rowsPerGroup
        let numGroups = (outDim + rowsPerGroup - 1) / rowsPerGroup

        let outputs = Self.prefillMetalKernel(
            [
                xRe.reshaped(tokens * inDim),
                xIm.reshaped(tokens * inDim),
                codes.reshaped(outDim * packedPerRow),
                gammaReal.reshaped(outDim * blocksPerRow),
                gammaImag.reshaped(outDim * blocksPerRow),
            ],
            template: [
                ("QK", blockSize),
                ("BLOCKS_PER_ROW", blocksPerRow),
                ("PACKED_PER_BLOCK", blockSize / 4),
                ("PACKED_PER_ROW", packedPerRow),
                ("IN_DIM", inDim),
                ("OUT_DIM", outDim),
                ("CODEBOOK_10_POS_IMAG", codebook10IsPosImag),
                ("SIMD_WIDTH", RPFQViTMetalKernels.simdWidth),
                ("PACKED_PER_THREAD", RPFQViTMetalKernels.packedPerThread),
                ("ROWS_PER_GROUP", rowsPerGroup),
                ("TG_SIZE", tgSize),
            ],
            grid: (numGroups * tgSize, tokens, 1),
            threadGroup: (tgSize, 1, 1),
            outputShapes: [[tokens * outDim], [tokens * outDim]],
            outputDTypes: [.float16, .float16],
            stream: .gpu
        )

        return (outputs[0].reshaped(1, tokens, outDim), outputs[1].reshaped(1, tokens, outDim))
    }

    func dequantize() -> (wReal: MLXArray, wImag: MLXArray) {
        let c0 = (codes & 0x03).asType(.int32)
        let c1 = ((codes >> 2) & 0x03).asType(.int32)
        let c2 = ((codes >> 4) & 0x03).asType(.int32)
        let c3 = ((codes >> 6) & 0x03).asType(.int32)

        let allCodes = MLX.stacked([c0, c1, c2, c3], axis: -1)
            .reshaped(outDim, blocksPerRow, blockSize)

        let realSign = MLX.where(
            allCodes .== 0,
            -1.0,
            MLX.where(allCodes .== 1, 1.0, MLXArray(0.0))
        ).asType(.float16)

        let imagSign: MLXArray
        if codebook10IsPosImag {
            imagSign = MLX.where(
                allCodes .== 2,
                1.0,
                MLX.where(allCodes .== 3, -1.0, MLXArray(0.0))
            ).asType(.float16)
        } else {
            imagSign = MLX.where(
                allCodes .== 2,
                -1.0,
                MLX.where(allCodes .== 3, 1.0, MLXArray(0.0))
            ).asType(.float16)
        }

        let gr = gammaReal.expandedDimensions(axis: -1)
        let gi = gammaImag.expandedDimensions(axis: -1)
        let wReal = (realSign / gr).reshaped(outDim, inDim)
        let wImag = (imagSign / gi).reshaped(outDim, inDim)
        return (wReal, wImag)
    }

    public func callAsFunction(_ xRe: MLXArray, _ xIm: MLXArray) -> (MLXArray, MLXArray) {
        if xRe.ndim == 3, xRe.dim(0) == 1, xRe.dim(1) == 1, xRe.dtype == .float16 {
            return metalDecode(xRe, xIm)
        }
        if xRe.ndim == 3, xRe.dim(0) == 1, xRe.dim(1) > 1, xRe.dtype == .float16 {
            return metalPrefill(xRe, xIm)
        }

        let (wRe, wIm) = dequantize()
        let xr = xRe.asType(wRe.dtype)
        let xi = xIm.asType(wRe.dtype)
        let yRe = MLX.matmul(xr, wRe.T) + MLX.matmul(xi, wIm.T)
        let yIm = MLX.matmul(xr, wIm.T) - MLX.matmul(xi, wRe.T)
        return (yRe, yIm)
    }
}

public class RPFQViTWideLinear: Module {
    private static let fusedPrefillKernel: MLXFast.MLXFastKernel = MLXFast.metalKernel(
        name: "rpfq_vit_widelinear_prefill",
        inputNames: [
            "xRe", "xIm",
            "u0Codes", "u0GammaR", "u0GammaI",
            "u1Codes", "u1GammaR", "u1GammaI",
            "w0Codes", "w0GammaR", "w0GammaI",
            "w1Codes", "w1GammaR", "w1GammaI",
        ],
        outputNames: ["outRe", "outIm"],
        source: RPFQViTMetalKernels.widePrefillSource,
        ensureRowContiguous: true
    )

    let realInDim: Int
    let realOutDim: Int
    let complexInDim: Int
    let complexOutDim: Int
    let originalOutComplex: Int
    let blockSize: Int
    let blocksPerRow: Int
    let packedPerRow: Int

    @ModuleInfo(key: "u_s0") var uS0: RPFQViTPhaseLinear
    @ModuleInfo(key: "u_s1") var uS1: RPFQViTPhaseLinear
    @ModuleInfo(key: "w_s0") var wS0: RPFQViTPhaseLinear
    @ModuleInfo(key: "w_s1") var wS1: RPFQViTPhaseLinear
    @ParameterInfo(key: "bias") var bias: MLXArray

    public init(
        realInDim: Int,
        realOutDim: Int,
        complexInDim: Int,
        complexOutDim: Int,
        originalOutComplex: Int,
        blockSize: Int
    ) {
        self.realInDim = realInDim
        self.realOutDim = realOutDim
        self.complexInDim = complexInDim
        self.complexOutDim = complexOutDim
        self.originalOutComplex = originalOutComplex
        self.blockSize = blockSize
        self.blocksPerRow = complexInDim / blockSize
        self.packedPerRow = complexInDim / 4

        self._uS0.wrappedValue = RPFQViTPhaseLinear(
            inDim: complexInDim,
            outDim: complexOutDim,
            blockSize: blockSize
        )
        self._uS1.wrappedValue = RPFQViTPhaseLinear(
            inDim: complexInDim,
            outDim: complexOutDim,
            blockSize: blockSize
        )
        self._wS0.wrappedValue = RPFQViTPhaseLinear(
            inDim: complexInDim,
            outDim: complexOutDim,
            blockSize: blockSize
        )
        self._wS1.wrappedValue = RPFQViTPhaseLinear(
            inDim: complexInDim,
            outDim: complexOutDim,
            blockSize: blockSize
        )
        self._bias.wrappedValue = MLXArray.zeros([realOutDim], dtype: .float16)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let halfIn = realInDim / 2
        let xRe = x[0..., 0..., ..<halfIn].asType(.float16)
        let xIm = x[0..., 0..., halfIn...].asType(.float16)

        let paddedXRe: MLXArray
        let paddedXIm: MLXArray
        if complexInDim > halfIn {
            let padSize = complexInDim - halfIn
            var padShape = xRe.shape
            padShape[padShape.count - 1] = padSize
            let zeros = MLXArray.zeros(padShape, dtype: xRe.dtype)
            paddedXRe = MLX.concatenated([xRe, zeros], axis: -1)
            paddedXIm = MLX.concatenated([xIm, zeros], axis: -1)
        } else {
            paddedXRe = xRe
            paddedXIm = xIm
        }

        if paddedXRe.ndim == 3, paddedXRe.dim(0) == 1, paddedXRe.dtype == .float16 {
            return applyBias(fusedPrefillPath(paddedXRe, paddedXIm))
        }

        return applyBias(separatePath(paddedXRe, paddedXIm))
    }

    private func fusedPrefillPath(_ xRe: MLXArray, _ xIm: MLXArray) -> MLXArray {
        let tokens = xRe.dim(1)
        let tgSize = RPFQViTMetalKernels.threadgroupSize
        let rowsPerGroup = RPFQViTMetalKernels.rowsPerGroup
        let numGroups = (complexOutDim + rowsPerGroup - 1) / rowsPerGroup

        let outputs = Self.fusedPrefillKernel(
            [
                xRe.reshaped(tokens * complexInDim),
                xIm.reshaped(tokens * complexInDim),
                uS0.codes.reshaped(complexOutDim * packedPerRow),
                uS0.gammaReal.reshaped(complexOutDim * blocksPerRow),
                uS0.gammaImag.reshaped(complexOutDim * blocksPerRow),
                uS1.codes.reshaped(complexOutDim * packedPerRow),
                uS1.gammaReal.reshaped(complexOutDim * blocksPerRow),
                uS1.gammaImag.reshaped(complexOutDim * blocksPerRow),
                wS0.codes.reshaped(complexOutDim * packedPerRow),
                wS0.gammaReal.reshaped(complexOutDim * blocksPerRow),
                wS0.gammaImag.reshaped(complexOutDim * blocksPerRow),
                wS1.codes.reshaped(complexOutDim * packedPerRow),
                wS1.gammaReal.reshaped(complexOutDim * blocksPerRow),
                wS1.gammaImag.reshaped(complexOutDim * blocksPerRow),
            ],
            template: [
                ("QK", blockSize),
                ("BLOCKS_PER_ROW", blocksPerRow),
                ("PACKED_PER_BLOCK", blockSize / 4),
                ("PACKED_PER_ROW", packedPerRow),
                ("IN_DIM", complexInDim),
                ("OUT_DIM", complexOutDim),
                ("CODEBOOK_10_POS_IMAG", false),
                ("SIMD_WIDTH", RPFQViTMetalKernels.simdWidth),
                ("PACKED_PER_THREAD", RPFQViTMetalKernels.packedPerThread),
                ("ROWS_PER_GROUP", rowsPerGroup),
                ("TG_SIZE", tgSize),
            ],
            grid: (numGroups * tgSize, tokens, 1),
            threadGroup: (tgSize, 1, 1),
            outputShapes: [[tokens * complexOutDim], [tokens * complexOutDim]],
            outputDTypes: [.float16, .float16],
            stream: .gpu
        )

        var yRe = outputs[0].reshaped(1, tokens, complexOutDim)
        var yIm = outputs[1].reshaped(1, tokens, complexOutDim)

        if complexOutDim > originalOutComplex {
            yRe = yRe[0..., 0..., ..<originalOutComplex]
            yIm = yIm[0..., 0..., ..<originalOutComplex]
        }

        return MLX.concatenated([yRe, yIm], axis: -1)
    }

    private func separatePath(_ xRe: MLXArray, _ xIm: MLXArray) -> MLXArray {
        let negXIm = -xIm

        let (uRe0, uIm0) = uS0(xRe, negXIm)
        let (uRe1, uIm1) = uS1(xRe, negXIm)
        let (wRe0, wIm0) = wS0(xRe, xIm)
        let (wRe1, wIm1) = wS1(xRe, xIm)

        var yRe = uRe0 + uRe1 + wRe0 + wRe1
        var yIm = uIm0 + uIm1 + wIm0 + wIm1

        if complexOutDim > originalOutComplex {
            yRe = yRe[0..., 0..., ..<originalOutComplex]
            yIm = yIm[0..., 0..., ..<originalOutComplex]
        }

        return MLX.concatenated([yRe, yIm], axis: -1)
    }

    private func applyBias(_ y: MLXArray) -> MLXArray {
        y + bias.reshaped(1, 1, realOutDim).asType(y.dtype)
    }
}

public class RPFQViTLayerNorm: Module {
    let eps: Float

    @ParameterInfo(key: "weight") var weight: MLXArray
    @ParameterInfo(key: "bias") var bias: MLXArray

    public init(dims: Int, eps: Float) {
        self.eps = eps
        self._weight.wrappedValue = MLXArray.ones([dims], dtype: .float16)
        self._bias.wrappedValue = MLXArray.zeros([dims], dtype: .float16)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let x32 = x.asType(.float32)
        let mean = MLX.mean(x32, axis: -1, keepDims: true)
        let centered = x32 - mean
        let variance = MLX.mean(centered * centered, axis: -1, keepDims: true)
        let normalized = centered / MLX.sqrt(variance + eps)
        return (
            normalized * weight.asType(.float32) + bias.asType(.float32)
        ).asType(x.dtype)
    }
}

private func rpfq_vitGELU(_ x: MLXArray) -> MLXArray {
    let x32 = x.asType(.float32)
    let cubic = x32 * x32 * x32
    let coeff = MLXArray(Float(0.797_884_560_8)).asType(.float32)
    let inner = coeff * (x32 + 0.044_715 * cubic)
    return (0.5 * x32 * (1.0 + MLX.tanh(inner))).asType(x.dtype)
}

public class RPFQViTAttention: Module {
    let hiddenSize: Int
    let numHeads: Int
    let headDim: Int
    let scale: Float

    @ModuleInfo(key: "qkv") var qkv: RPFQViTWideLinear
    @ModuleInfo(key: "proj") var proj: RPFQViTWideLinear

    public init(_ config: RPFQViTConfiguration) {
        hiddenSize = config.hiddenSize
        numHeads = config.numAttentionHeads
        headDim = config.headDim
        scale = 1.0 / sqrt(Float(headDim))

        self._qkv.wrappedValue = RPFQViTWideLinear(
            realInDim: config.hiddenSize,
            realOutDim: config.hiddenSize * 3,
            complexInDim: config.hiddenComplexPadded,
            complexOutDim: config.qkvComplexPadded,
            originalOutComplex: config.qkvComplex,
            blockSize: config.blockSize
        )
        self._proj.wrappedValue = RPFQViTWideLinear(
            realInDim: config.hiddenSize,
            realOutDim: config.hiddenSize,
            complexInDim: config.hiddenComplexPadded,
            complexOutDim: config.hiddenComplexPadded,
            originalOutComplex: config.hiddenComplex,
            blockSize: config.blockSize
        )
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (out, _) = forwardWithDebug(x)
        return out
    }

    func forwardWithDebug(_ x: MLXArray) -> (MLXArray, RPFQViTAttentionDebugTensors) {
        let batch = x.dim(0)
        let tokens = x.dim(1)

        let qkvConcat = qkv(x)
        let q = qkvConcat[0..., 0..., ..<hiddenSize]
            .reshaped(batch, tokens, numHeads, headDim)
            .transposed(0, 2, 1, 3)
        let k = qkvConcat[0..., 0..., hiddenSize..<(hiddenSize * 2)]
            .reshaped(batch, tokens, numHeads, headDim)
            .transposed(0, 2, 1, 3)
        let v = qkvConcat[0..., 0..., (hiddenSize * 2)...]
            .reshaped(batch, tokens, numHeads, headDim)
            .transposed(0, 2, 1, 3)

        let attnScale = MLXArray(scale).asType(q.dtype)
        var attn = MLX.matmul(q, k.transposed(0, 1, 3, 2)) * attnScale
        attn = MLX.softmax(attn, axis: -1)

        let attnOutPreProj = MLX.matmul(attn, v)
            .transposed(0, 2, 1, 3)
            .reshaped(batch, tokens, hiddenSize)
        let attnProj = proj(attnOutPreProj)

        return (
            attnProj,
            RPFQViTAttentionDebugTensors(
                qkvConcat: qkvConcat,
                q: q,
                k: k,
                v: v,
                attnOutPreProj: attnOutPreProj,
                attnProj: attnProj
            )
        )
    }
}

public class RPFQViTMLP: Module {
    @ModuleInfo(key: "fc1") var fc1: RPFQViTWideLinear
    @ModuleInfo(key: "fc2") var fc2: RPFQViTWideLinear

    public init(_ config: RPFQViTConfiguration) {
        self._fc1.wrappedValue = RPFQViTWideLinear(
            realInDim: config.hiddenSize,
            realOutDim: config.intermediateSize,
            complexInDim: config.hiddenComplexPadded,
            complexOutDim: config.intermediateComplexPadded,
            originalOutComplex: config.intermediateComplex,
            blockSize: config.blockSize
        )
        self._fc2.wrappedValue = RPFQViTWideLinear(
            realInDim: config.intermediateSize,
            realOutDim: config.hiddenSize,
            complexInDim: config.intermediateComplexPadded,
            complexOutDim: config.hiddenComplexPadded,
            originalOutComplex: config.hiddenComplex,
            blockSize: config.blockSize
        )
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        fc2(rpfq_vitGELU(fc1(x)))
    }
}

public class RPFQViTBlock: Module {
    @ModuleInfo(key: "norm1") var norm1: RPFQViTLayerNorm
    @ModuleInfo(key: "attn") var attn: RPFQViTAttention
    @ModuleInfo(key: "norm2") var norm2: RPFQViTLayerNorm
    @ModuleInfo(key: "mlp") var mlp: RPFQViTMLP

    public init(_ config: RPFQViTConfiguration) {
        self._norm1.wrappedValue = RPFQViTLayerNorm(dims: config.hiddenSize, eps: config.layerNormEps)
        self._attn.wrappedValue = RPFQViTAttention(config)
        self._norm2.wrappedValue = RPFQViTLayerNorm(dims: config.hiddenSize, eps: config.layerNormEps)
        self._mlp.wrappedValue = RPFQViTMLP(config)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (out, _) = forwardWithDebug(x)
        return out
    }

    func forwardWithDebug(_ x: MLXArray) -> (MLXArray, RPFQViTBlockTraceDebugTensors) {
        let norm1Output = norm1(x)
        let (attnOut, attnDebug) = attn.forwardWithDebug(norm1Output)
        let residual1 = x + attnOut
        let mlpOut = mlp(norm2(residual1))
        let blockOutput = residual1 + mlpOut
        return (
            blockOutput,
            RPFQViTBlockTraceDebugTensors(
                blockInput: x,
                norm1: norm1Output,
                qkvConcat: attnDebug.qkvConcat,
                q: attnDebug.q,
                k: attnDebug.k,
                v: attnDebug.v,
                attnOutPreProj: attnDebug.attnOutPreProj,
                attnProj: attnDebug.attnProj,
                residual1: residual1,
                blockOutput: blockOutput
            )
        )
    }
}

public class RPFQViTBackbone: Module {
    let hiddenSize: Int

    @ModuleInfo(key: "patch_embed") var patchEmbed: Linear
    @ParameterInfo(key: "cls_token") var clsToken: MLXArray
    @ParameterInfo(key: "pos_embed") var posEmbed: MLXArray
    @ModuleInfo var layers: [RPFQViTBlock]
    @ModuleInfo var norm: RPFQViTLayerNorm

    public init(_ config: RPFQViTConfiguration) {
        hiddenSize = config.hiddenSize

        self._patchEmbed.wrappedValue = Linear(
            config.patchVectorSize,
            config.hiddenSize,
            bias: true
        )
        self._clsToken.wrappedValue = MLXArray.zeros([1, 1, config.hiddenSize], dtype: .float16)
        self._posEmbed.wrappedValue = MLXArray.zeros(
            [1, config.patchCount + 1, config.hiddenSize],
            dtype: .float16
        )
        self._layers.wrappedValue = (0 ..< config.numHiddenLayers).map { _ in
            RPFQViTBlock(config)
        }
        self._norm.wrappedValue = RPFQViTLayerNorm(dims: config.hiddenSize, eps: config.layerNormEps)
    }

    public func callAsFunction(_ patches: MLXArray) -> MLXArray {
        let (x, _) = forwardWithDebug(patches)
        return x
    }

    func forwardWithDebug(_ patches: MLXArray) -> (MLXArray, RPFQViTBackboneDebugTensors) {
        let patchEmbedOutput = patchEmbed(patches)
        var x = patchEmbedOutput
        let batch = x.dim(0)
        let cls = batch == 1
            ? clsToken.asType(x.dtype)
            : MLX.repeated(clsToken.asType(x.dtype), count: batch, axis: 0)

        x = MLX.concatenated([cls, x], axis: 1)
        let pos = posEmbed[0..., ..<x.dim(1), 0...].asType(x.dtype)
        x = x + pos

        var block0Trace: RPFQViTBlockTraceDebugTensors?
        var block0Output: MLXArray?
        var block1Output: MLXArray?
        for (index, layer) in layers.enumerated() {
            if index == 0 {
                let trace: RPFQViTBlockTraceDebugTensors
                (x, trace) = layer.forwardWithDebug(x)
                block0Trace = trace
                block0Output = x
            } else if index == 1 {
                x = layer(x)
                block1Output = x
            } else {
                x = layer(x)
            }
        }
        let finalNormOutput = norm(x)
        return (
            finalNormOutput,
            RPFQViTBackboneDebugTensors(
                patchEmbed: patchEmbedOutput,
                block0Trace: block0Trace,
                block0: block0Output,
                block1: block1Output,
                finalNorm: finalNormOutput
            )
        )
    }
}

public class RPFQViTClassifier: Module {
    let config: RPFQViTConfiguration

    @ModuleInfo(key: "model") var model: RPFQViTBackbone
    @ModuleInfo(key: "head") var head: Linear

    public init(_ config: RPFQViTConfiguration) {
        self.config = config
        self._model.wrappedValue = RPFQViTBackbone(config)
        self._head.wrappedValue = Linear(config.hiddenSize, config.numClasses, bias: true)
    }

    public func callAsFunction(_ patches: MLXArray) -> MLXArray {
        let (logits, _) = forwardWithDebug(patches)
        return logits
    }

    func forwardWithDebug(_ patches: MLXArray) -> (MLXArray, RPFQViTForwardDebugTensors) {
        var x: MLXArray
        let debugTensors: RPFQViTBackboneDebugTensors
        (x, debugTensors) = model.forwardWithDebug(patches)
        x = x[0..., ..<1, 0...]
        let logits = head(x).reshaped(1, config.numClasses)
        return (
            logits,
            RPFQViTForwardDebugTensors(
                patchEmbed: debugTensors.patchEmbed,
                block0Trace: debugTensors.block0Trace,
                block0: debugTensors.block0,
                block1: debugTensors.block1,
                finalNorm: debugTensors.finalNorm,
                logits: logits
            )
        )
    }
}
