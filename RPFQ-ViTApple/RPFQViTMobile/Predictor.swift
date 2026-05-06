import CoreGraphics
import Foundation
import MLX
import MLXNN

struct PredictionResult: Identifiable, Sendable {
    let id = UUID()
    let index: Int
    let label: String
    let confidence: Float
}

struct InferenceProfile: Sendable {
    let preprocessMs: Double
    let inferenceMs: Double
}

private struct RPFQViTTensorSummary: Codable {
    let shape: [Int]
    let firstValues: [Float]
    let min: Float
    let max: Float
    let mean: Float
}

private struct RPFQViTDebugReport: Codable {
    let modelVariant: String
    let preprocess: RPFQViTPreprocessConfiguration
    let summaries: [String: RPFQViTTensorSummary]
}

fileprivate struct RPFQViTDebugConfiguration {
    let exportDirectory: URL?
    let exportPatches: Bool
    let enabledStages: Set<String>
    let maxTensorValues: Int
    let maxPatchValues: Int

    var isEnabled: Bool { exportDirectory != nil }

    /// Recompute each time so toggles via `UserDefaults` take effect without recompiling.
    static var current: RPFQViTDebugConfiguration { RPFQViTDebugConfiguration() }

    init(processInfo: ProcessInfo = .processInfo) {
        let defaults = UserDefaults.standard
        let environment = processInfo.environment

        let enabled = environment["RPFQ_VIT_DEBUG_EXPORT"] == "1" || defaults.bool(forKey: "RPFQ_VIT_DEBUG_EXPORT")
        let configuredPath =
            environment["RPFQ_VIT_DEBUG_EXPORT_DIR"]
            ?? defaults.string(forKey: "RPFQ_VIT_DEBUG_EXPORT_DIR")

        if let configuredPath, !configuredPath.isEmpty {
            exportDirectory = URL(fileURLWithPath: configuredPath, isDirectory: true)
        } else if enabled {
            let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first
            exportDirectory = (docs ?? FileManager.default.temporaryDirectory)
                .appendingPathComponent("RPFQViTDebug", isDirectory: true)
        } else {
            exportDirectory = nil
        }

        let rawStages =
            environment["RPFQ_VIT_DEBUG_STAGES"]
            ?? defaults.string(forKey: "RPFQ_VIT_DEBUG_STAGES")
            ?? "patch_embed,block0_input,block0_norm1,block0_qkv_concat,block0_q,block0_k,block0_v,block0_attn_out_pre_proj,block0_attn_proj,block0_residual1,block0,block1,final_norm,logits"
        enabledStages = Set(
            rawStages
                .split(separator: ",")
                .map { $0.trimmingCharacters(in: .whitespacesAndNewlines).lowercased() }
                .filter { !$0.isEmpty }
        )

        if let rawPatchFlag = environment["RPFQ_VIT_DEBUG_EXPORT_PATCHES"] {
            exportPatches = rawPatchFlag != "0"
        } else if defaults.object(forKey: "RPFQ_VIT_DEBUG_EXPORT_PATCHES") != nil {
            exportPatches = defaults.bool(forKey: "RPFQ_VIT_DEBUG_EXPORT_PATCHES")
        } else {
            exportPatches = true
        }

        // Exporting full tensors to JSON can be enormous and will stall the UI.
        // Default to a small prefix; users can override via env/defaults if needed.
        if let rawMax = environment["RPFQ_VIT_DEBUG_MAX_VALUES"], let parsed = Int(rawMax) {
            maxTensorValues = max(0, parsed)
        } else if defaults.object(forKey: "RPFQ_VIT_DEBUG_MAX_VALUES") != nil {
            maxTensorValues = max(0, defaults.integer(forKey: "RPFQ_VIT_DEBUG_MAX_VALUES"))
        } else {
            maxTensorValues = 2048
        }

        // Patches are relatively small (~150k floats for ViT-B/16) and are critical for parity checks.
        if let rawMax = environment["RPFQ_VIT_DEBUG_MAX_PATCH_VALUES"], let parsed = Int(rawMax) {
            maxPatchValues = max(0, parsed)
        } else if defaults.object(forKey: "RPFQ_VIT_DEBUG_MAX_PATCH_VALUES") != nil {
            maxPatchValues = max(0, defaults.integer(forKey: "RPFQ_VIT_DEBUG_MAX_PATCH_VALUES"))
        } else {
            maxPatchValues = 200_000
        }
    }

    func shouldExport(stage: String) -> Bool {
        let key = stage.lowercased()
        return enabledStages.contains("all") || enabledStages.contains(key)
    }
}

final class RPFQViTPredictor {
    let modelDirectory: URL
    let config: RPFQViTConfiguration
    let labels: [String]
    let preprocessor: ImagePreprocessor
    let model: RPFQViTClassifier

    private init(
        modelDirectory: URL,
        config: RPFQViTConfiguration,
        labels: [String],
        preprocessor: ImagePreprocessor,
        model: RPFQViTClassifier
    ) {
        self.modelDirectory = modelDirectory
        self.config = config
        self.labels = labels
        self.preprocessor = preprocessor
        self.model = model
    }

    static func load(from modelDirectory: URL) throws -> RPFQViTPredictor {
        let configURL = modelDirectory.appendingPathComponent("config.json")
        let weightsURL = modelDirectory.appendingPathComponent("model.safetensors")

        let configData = try Data(contentsOf: configURL)
        let config = try JSONDecoder().decode(RPFQViTConfiguration.self, from: configData)

        let preprocessURL = modelDirectory.appendingPathComponent("preprocess.json")
        let preprocessConfig: RPFQViTPreprocessConfiguration
        if FileManager.default.fileExists(atPath: preprocessURL.path) {
            let preprocessData = try Data(contentsOf: preprocessURL)
            preprocessConfig = try JSONDecoder().decode(
                RPFQViTPreprocessConfiguration.self,
                from: preprocessData
            )
        } else {
            preprocessConfig = config.preprocess
        }

        let labels = try loadLabels(
            modelDirectory: modelDirectory,
            numClasses: config.numClasses
        )

        let model = RPFQViTClassifier(config)
        let (weights, _) = try loadArraysAndMetadata(url: weightsURL)
        try model.update(
            parameters: ModuleParameters.unflattened(weights),
            verify: [.noUnusedKeys]
        )
        eval(model)

        return RPFQViTPredictor(
            modelDirectory: modelDirectory,
            config: config,
            labels: labels,
            preprocessor: ImagePreprocessor(config: preprocessConfig),
            model: model
        )
    }

    func predict(cgImage: CGImage, topK: Int = 5) throws -> ([PredictionResult], InferenceProfile) {
        // Read the latest debug configuration each inference so toggles take effect immediately.
        let debugConfiguration = RPFQViTDebugConfiguration.current

        let preprocessStart = CFAbsoluteTimeGetCurrent()
        let patches = try preprocessor.preparePatches(from: cgImage)
        let preprocessMs = (CFAbsoluteTimeGetCurrent() - preprocessStart) * 1000.0

        let inferenceStart = CFAbsoluteTimeGetCurrent()
        let logits: MLXArray
        let debugTensors: RPFQViTForwardDebugTensors?
        if debugConfiguration.isEnabled {
            let (debugLogits, captured) = model.forwardWithDebug(patches)
            logits = debugLogits.reshaped(config.numClasses)
            debugTensors = captured
        } else {
            logits = model(patches).reshaped(config.numClasses)
            debugTensors = nil
        }
        eval(logits)
        let probs = MLX.softmax(logits, axis: -1)
        eval(probs)

        let inferenceMs = (CFAbsoluteTimeGetCurrent() - inferenceStart) * 1000.0

        if let debugTensors {
            try exportDebugArtifacts(
                patches: patches,
                tensors: debugTensors,
                debugConfiguration: debugConfiguration
            )
        }

        let k = max(1, min(topK, config.numClasses))
        let probValues = tensorValues(probs, limit: nil)
        let top = topKIndices(values: probValues, k: k)
        let results: [PredictionResult] = top.map { index, confidence in
            let label = index < labels.count ? labels[index] : "Class \(index)"
            return PredictionResult(index: index, label: label, confidence: confidence)
        }
        let profile = InferenceProfile(preprocessMs: preprocessMs, inferenceMs: inferenceMs)
        return (results, profile)
    }

    private static func loadLabels(
        modelDirectory: URL,
        numClasses: Int
    ) throws -> [String] {
        let fm = FileManager.default

        let candidates = ["labels.txt"]

        for filename in candidates {
            let url = modelDirectory.appendingPathComponent(filename)
            guard fm.fileExists(atPath: url.path) else { continue }

            let raw = try String(contentsOf: url, encoding: .utf8)
            let parsed = raw
                .components(separatedBy: .newlines)
                .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
                .filter { !$0.isEmpty }

            if parsed.count >= numClasses {
                return Array(parsed.prefix(numClasses))
            }

            // If the file exists but is shorter than expected, keep searching.
        }

        return (0 ..< numClasses).map { "Class \($0)" }
    }

    private func topKIndices(values: [Float], k: Int) -> [(Int, Float)] {
        guard !values.isEmpty else { return [] }
        let kk = max(1, min(k, values.count))

        // Full sort is fine for ~1k classes and keeps code simple/robust.
        let sorted = values.enumerated().sorted { lhs, rhs in
            lhs.element > rhs.element
        }
        return sorted.prefix(kk).map { ($0.offset, $0.element) }
    }

    private func exportDebugArtifacts(
        patches: MLXArray,
        tensors: RPFQViTForwardDebugTensors,
        debugConfiguration: RPFQViTDebugConfiguration
    ) throws {
        guard let exportDirectory = debugConfiguration.exportDirectory else { return }

        let fm = FileManager.default
        try fm.createDirectory(at: exportDirectory, withIntermediateDirectories: true, attributes: nil)

        let formatter = DateFormatter()
        formatter.dateFormat = "yyyyMMdd-HHmmss"
        let runDirectory = exportDirectory.appendingPathComponent(
            "\(modelDirectory.lastPathComponent)-\(formatter.string(from: Date()))",
            isDirectory: true
        )
        try fm.createDirectory(at: runDirectory, withIntermediateDirectories: true, attributes: nil)
        UserDefaults.standard.set(runDirectory.path, forKey: "RPFQ_VIT_DEBUG_LAST_RUN_DIR")

        let maxValues = debugConfiguration.maxTensorValues
        let maxPatchValues = debugConfiguration.maxPatchValues

        if debugConfiguration.exportPatches {
            let patchLimit: Int? = maxPatchValues == 0 ? nil : maxPatchValues
            let patchPayload = tensorPayload(patches, limit: patchLimit)
            let patchData = try JSONSerialization.data(withJSONObject: patchPayload, options: [.prettyPrinted])
            try patchData.write(to: runDirectory.appendingPathComponent("patches.json"))
        }

        var summaries: [String: RPFQViTTensorSummary] = [:]
        var tensorPayloads: [String: [String: Any]] = [:]
        if debugConfiguration.shouldExport(stage: "patch_embed") {
            summaries["patch_embed"] = tensorSummary(tensors.patchEmbed)
            tensorPayloads["patch_embed"] = tensorPayload(tensors.patchEmbed, limit: maxValues)
        }
        if debugConfiguration.shouldExport(stage: "block0_input"), let trace = tensors.block0Trace {
            summaries["block0_input"] = tensorSummary(trace.blockInput)
            tensorPayloads["block0_input"] = tensorPayload(trace.blockInput, limit: maxValues)
        }
        if debugConfiguration.shouldExport(stage: "block0_norm1"), let trace = tensors.block0Trace {
            summaries["block0_norm1"] = tensorSummary(trace.norm1)
            tensorPayloads["block0_norm1"] = tensorPayload(trace.norm1, limit: maxValues)
        }
        if debugConfiguration.shouldExport(stage: "block0_qkv_concat"), let trace = tensors.block0Trace {
            summaries["block0_qkv_concat"] = tensorSummary(trace.qkvConcat)
            tensorPayloads["block0_qkv_concat"] = tensorPayload(trace.qkvConcat, limit: maxValues)
        }
        if debugConfiguration.shouldExport(stage: "block0_q"), let trace = tensors.block0Trace {
            summaries["block0_q"] = tensorSummary(trace.q)
            tensorPayloads["block0_q"] = tensorPayload(trace.q, limit: maxValues)
        }
        if debugConfiguration.shouldExport(stage: "block0_k"), let trace = tensors.block0Trace {
            summaries["block0_k"] = tensorSummary(trace.k)
            tensorPayloads["block0_k"] = tensorPayload(trace.k, limit: maxValues)
        }
        if debugConfiguration.shouldExport(stage: "block0_v"), let trace = tensors.block0Trace {
            summaries["block0_v"] = tensorSummary(trace.v)
            tensorPayloads["block0_v"] = tensorPayload(trace.v, limit: maxValues)
        }
        if debugConfiguration.shouldExport(stage: "block0_attn_out_pre_proj"), let trace = tensors.block0Trace {
            summaries["block0_attn_out_pre_proj"] = tensorSummary(trace.attnOutPreProj)
            tensorPayloads["block0_attn_out_pre_proj"] = tensorPayload(trace.attnOutPreProj, limit: maxValues)
        }
        if debugConfiguration.shouldExport(stage: "block0_attn_proj"), let trace = tensors.block0Trace {
            summaries["block0_attn_proj"] = tensorSummary(trace.attnProj)
            tensorPayloads["block0_attn_proj"] = tensorPayload(trace.attnProj, limit: maxValues)
        }
        if debugConfiguration.shouldExport(stage: "block0_residual1"), let trace = tensors.block0Trace {
            summaries["block0_residual1"] = tensorSummary(trace.residual1)
            tensorPayloads["block0_residual1"] = tensorPayload(trace.residual1, limit: maxValues)
        }
        if debugConfiguration.shouldExport(stage: "block0"), let block0 = tensors.block0 {
            summaries["block0"] = tensorSummary(block0)
            tensorPayloads["block0"] = tensorPayload(block0, limit: maxValues)
        }
        if debugConfiguration.shouldExport(stage: "block1"), let block1 = tensors.block1 {
            summaries["block1"] = tensorSummary(block1)
            tensorPayloads["block1"] = tensorPayload(block1, limit: maxValues)
        }
        if debugConfiguration.shouldExport(stage: "final_norm") {
            summaries["final_norm"] = tensorSummary(tensors.finalNorm)
            tensorPayloads["final_norm"] = tensorPayload(tensors.finalNorm, limit: maxValues)
        }
        if debugConfiguration.shouldExport(stage: "logits") {
            summaries["logits"] = tensorSummary(tensors.logits)
            tensorPayloads["logits"] = tensorPayload(tensors.logits, limit: maxValues)
        }

        let report = RPFQViTDebugReport(
            modelVariant: config.variant,
            preprocess: preprocessor.config,
            summaries: summaries
        )
        let reportData = try JSONEncoder().encode(report)
        try reportData.write(to: runDirectory.appendingPathComponent("summaries.json"))

        let tensorsData = try JSONSerialization.data(withJSONObject: tensorPayloads, options: [.prettyPrinted])
        try tensorsData.write(to: runDirectory.appendingPathComponent("tensors.json"))
    }

    private func tensorSummary(_ array: MLXArray) -> RPFQViTTensorSummary {
        // Compute global statistics with MLX reductions (fast, avoids huge host copies).
        let x32 = array.asType(.float32)
        let minValue: Float = MLX.min(x32).item()
        let maxValue: Float = MLX.max(x32).item()
        let meanValue: Float = MLX.mean(x32).item()

        let values = tensorValues(array, limit: 128)
        return RPFQViTTensorSummary(
            shape: array.shape,
            firstValues: Array(values.prefix(10)),
            min: minValue,
            max: maxValue,
            mean: meanValue
        )
    }

    private func tensorPayload(_ array: MLXArray, limit: Int?) -> [String: Any] {
        [
            "shape": array.shape,
            "values": tensorValues(array, limit: limit),
        ]
    }

    private func tensorValues(_ array: MLXArray, limit: Int? = nil) -> [Float] {
        let elementCount = max(array.shape.reduce(1, *), 1)
        let flat = array.asType(.float32).reshaped(elementCount)
        eval(flat)
        let totalCount = flat.shape.first ?? 0
        let count = min(limit ?? totalCount, totalCount)
        return (0 ..< count).map { index in
            let value: Float = flat[index].item()
            return value
        }
    }
}
