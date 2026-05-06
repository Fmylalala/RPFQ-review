import Foundation

public struct RPFQViTPreprocessConfiguration: Codable, Sendable {
    public let imageSize: Int
    public let patchSize: Int
    public let numChannels: Int
    public let mean: [Float]
    public let std: [Float]
    public let interpolation: String
    public let cropPct: Float
    public let cropMode: String
    public let resizeMode: String

    enum CodingKeys: String, CodingKey {
        case imageSize = "image_size"
        case patchSize = "patch_size"
        case numChannels = "num_channels"
        case mean
        case std
        case interpolation
        case cropPct = "crop_pct"
        case cropMode = "crop_mode"
        case resizeMode = "resize_mode"
    }

    public init(
        imageSize: Int = 224,
        patchSize: Int = 16,
        numChannels: Int = 3,
        mean: [Float] = [0.485, 0.456, 0.406],
        std: [Float] = [0.229, 0.224, 0.225],
        interpolation: String = "bicubic",
        cropPct: Float = 0.875,
        cropMode: String = "center",
        resizeMode: String = "shortest_side"
    ) {
        self.imageSize = imageSize
        self.patchSize = patchSize
        self.numChannels = numChannels
        self.mean = mean
        self.std = std
        self.interpolation = interpolation
        self.cropPct = cropPct
        self.cropMode = cropMode
        self.resizeMode = resizeMode
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        imageSize = (try? c.decode(Int.self, forKey: .imageSize)) ?? 224
        patchSize = (try? c.decode(Int.self, forKey: .patchSize)) ?? 16
        numChannels = (try? c.decode(Int.self, forKey: .numChannels)) ?? 3
        mean = (try? c.decode([Float].self, forKey: .mean)) ?? [0.485, 0.456, 0.406]
        std = (try? c.decode([Float].self, forKey: .std)) ?? [0.229, 0.224, 0.225]
        interpolation = (try? c.decode(String.self, forKey: .interpolation)) ?? "bicubic"
        cropPct = (try? c.decode(Float.self, forKey: .cropPct)) ?? 0.875
        cropMode = (try? c.decode(String.self, forKey: .cropMode)) ?? "center"
        resizeMode = (try? c.decode(String.self, forKey: .resizeMode)) ?? "shortest_side"
    }
}

public struct RPFQViTConfiguration: Decodable, Sendable {
    public let modelType: String
    public let variant: String
    public let imageSize: Int
    public let patchSize: Int
    public let numChannels: Int
    public let hiddenSize: Int
    public let numHiddenLayers: Int
    public let numAttentionHeads: Int
    public let intermediateSize: Int
    public let numClasses: Int
    public let layerNormEps: Float

    public let hiddenComplex: Int
    public let hiddenComplexPadded: Int
    public let qkvComplex: Int
    public let qkvComplexPadded: Int
    public let intermediateComplex: Int
    public let intermediateComplexPadded: Int
    public let residualStages: Int
    public let blockSize: Int
    public let quantMethod: String

    public let preprocess: RPFQViTPreprocessConfiguration

    public var headDim: Int { hiddenSize / numAttentionHeads }
    public var patchVectorSize: Int { patchSize * patchSize * numChannels }
    public var patchCount: Int {
        let side = imageSize / patchSize
        return side * side
    }

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case variant
        case imageSize = "image_size"
        case patchSize = "patch_size"
        case numChannels = "num_channels"
        case hiddenSize = "hidden_size"
        case numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads"
        case intermediateSize = "intermediate_size"
        case numClasses = "num_classes"
        case layerNormEps = "layer_norm_eps"
        case rpfq_vit
        case preprocess
    }

    enum RPFQViTCodingKeys: String, CodingKey {
        case hiddenComplex = "hidden_complex"
        case hiddenComplexPadded = "hidden_complex_padded"
        case qkvComplex = "qkv_complex"
        case qkvComplexPadded = "qkv_complex_padded"
        case intermediateComplex = "intermediate_complex"
        case intermediateComplexPadded = "intermediate_complex_padded"
        case residualStages = "residual_stages"
        case blockSize = "block_size"
        case quantMethod = "quant_method"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)

        modelType = (try? c.decode(String.self, forKey: .modelType)) ?? "rpfq_vit"
        variant = (try? c.decode(String.self, forKey: .variant)) ?? "rpfq_vit_small_patch16_224_q"
        imageSize = (try? c.decode(Int.self, forKey: .imageSize)) ?? 224
        patchSize = (try? c.decode(Int.self, forKey: .patchSize)) ?? 16
        numChannels = (try? c.decode(Int.self, forKey: .numChannels)) ?? 3
        hiddenSize = try c.decode(Int.self, forKey: .hiddenSize)
        numHiddenLayers = try c.decode(Int.self, forKey: .numHiddenLayers)
        numAttentionHeads = try c.decode(Int.self, forKey: .numAttentionHeads)
        intermediateSize = try c.decode(Int.self, forKey: .intermediateSize)
        numClasses = try c.decode(Int.self, forKey: .numClasses)
        layerNormEps = (try? c.decode(Float.self, forKey: .layerNormEps)) ?? 1e-5

        if c.contains(.rpfq_vit),
           let rpfq_vit = try? c.nestedContainer(keyedBy: RPFQViTCodingKeys.self, forKey: .rpfq_vit)
        {
            hiddenComplex = (try? rpfq_vit.decode(Int.self, forKey: .hiddenComplex)) ?? (hiddenSize / 2)
            hiddenComplexPadded =
                (try? rpfq_vit.decode(Int.self, forKey: .hiddenComplexPadded)) ?? hiddenComplex
            qkvComplex = (try? rpfq_vit.decode(Int.self, forKey: .qkvComplex)) ?? (hiddenComplex * 3)
            qkvComplexPadded = (try? rpfq_vit.decode(Int.self, forKey: .qkvComplexPadded)) ?? qkvComplex
            intermediateComplex =
                (try? rpfq_vit.decode(Int.self, forKey: .intermediateComplex)) ?? (intermediateSize / 2)
            intermediateComplexPadded =
                (try? rpfq_vit.decode(Int.self, forKey: .intermediateComplexPadded)) ?? intermediateComplex
            residualStages = (try? rpfq_vit.decode(Int.self, forKey: .residualStages)) ?? 2
            blockSize = (try? rpfq_vit.decode(Int.self, forKey: .blockSize)) ?? 256
            quantMethod = (try? rpfq_vit.decode(String.self, forKey: .quantMethod)) ?? "complex_phase_v2"
        } else {
            hiddenComplex = hiddenSize / 2
            hiddenComplexPadded = hiddenComplex
            qkvComplex = hiddenComplex * 3
            qkvComplexPadded = qkvComplex
            intermediateComplex = intermediateSize / 2
            intermediateComplexPadded = intermediateComplex
            residualStages = 2
            blockSize = 256
            quantMethod = "complex_phase_v2"
        }

        preprocess =
            (try? c.decode(RPFQViTPreprocessConfiguration.self, forKey: .preprocess))
            ?? RPFQViTPreprocessConfiguration(
                imageSize: imageSize,
                patchSize: patchSize,
                numChannels: numChannels
            )
    }
}
