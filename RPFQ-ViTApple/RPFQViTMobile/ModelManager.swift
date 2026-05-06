import Foundation
import MLX
import Observation

struct RPFQViTModelBundle: Identifiable, Hashable, Sendable {
    let id: String
    let displayName: String
    let url: URL
    let exportVariant: String
    let checkpoint: String?
}

private struct RPFQViTBundleManifest: Decodable, Sendable {
    let modelType: String?
    let variant: String?
    let checkpoint: String?
    let exportFormat: String?
    let quantMethod: String?

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case variant
        case checkpoint
        case exportFormat = "export_format"
        case quantMethod = "quant_method"
    }
}

enum RPFQViTModelState: Equatable {
    case idle
    case loading(String)
    case ready
    case running(String)
    case error(String)

    var isReady: Bool {
        if case .ready = self { return true }
        return false
    }
}

@Observable
@MainActor
final class ModelManager {
    var state: RPFQViTModelState = .idle
    var predictor: RPFQViTPredictor?
    var selectedBundleID: String?
    var availableBundles: [RPFQViTModelBundle] = []
    var loadedBundle: RPFQViTModelBundle?
    var selectedImage: PlatformImage?
    var predictions: [PredictionResult] = []
    var profile: InferenceProfile?
    var debugExportEnabled: Bool = UserDefaults.standard.bool(forKey: "RPFQ_VIT_DEBUG_EXPORT")

    func loadModelIfNeeded() async {
        scanAvailableModels()
        if let runtimeFailure = InferenceRuntimeSupport.failureMessage() {
            state = .error(runtimeFailure)
            return
        }
        guard predictor == nil || loadedBundle?.id != selectedBundleID else { return }
        await reloadSelectedModel()
    }

    func scanAvailableModels() {
        let found = scanBundledModelDirectories()
        availableBundles = found

        if let selectedBundleID,
           found.contains(where: { $0.id == selectedBundleID })
        {
            return
        }

        selectedBundleID = found.first?.id
    }

    func reloadSelectedModel() async {
        scanAvailableModels()

        predictor = nil
        loadedBundle = nil
        predictions = []
        profile = nil

        if let runtimeFailure = InferenceRuntimeSupport.failureMessage() {
            state = .error(runtimeFailure)
            return
        }

        InferenceRuntimeSupport.resetGPUCacheIfSupported()

        guard let selectedBundleID,
              let bundle = availableBundles.first(where: { $0.id == selectedBundleID })
        else {
            state = .error("No model bundles found in app resources.")
            return
        }

        let modelURL = bundle.url
        guard FileManager.default.fileExists(atPath: modelURL.appendingPathComponent("config.json").path) else {
            state = .error(
                "Model bundle is missing config.json: \(modelURL.lastPathComponent)"
            )
            return
        }

        do {
            state = .loading("Loading \(bundle.displayName)...")
            predictor = try RPFQViTPredictor.load(from: modelURL)
            loadedBundle = bundle
            state = .ready
        } catch {
            state = .error("Failed to load RPFQViT bundle: \(error.localizedDescription)")
        }
    }

    func setDebugExportEnabled(_ enabled: Bool) {
        debugExportEnabled = enabled
        UserDefaults.standard.set(enabled, forKey: "RPFQ_VIT_DEBUG_EXPORT")
    }

    func setImage(_ image: PlatformImage) {
        selectedImage = image
        predictions = []
        profile = nil
    }

    func runPrediction() async {
        guard let predictor, let selectedImage else { return }

        do {
            state = .running("Running image classification...")
            let cgImage = try selectedImage.cgImageForInference()
            // Run inference off the main thread to keep UI responsive, especially when debug export is enabled.
            let (results, profile) = try await withCheckedThrowingContinuation { continuation in
                DispatchQueue.global(qos: .userInitiated).async {
                    do {
                        let out = try predictor.predict(cgImage: cgImage, topK: 5)
                        continuation.resume(returning: out)
                    } catch {
                        continuation.resume(throwing: error)
                    }
                }
            }
            self.predictions = results
            self.profile = profile
            state = .ready
        } catch {
            state = .error("Prediction failed: \(error.localizedDescription)")
        }
    }

    private func scanBundledModelDirectories() -> [RPFQViTModelBundle] {
        let fm = FileManager.default
        guard let resourceURL = Bundle.main.resourceURL else { return [] }

        let candidates: [URL]
        do {
            candidates = try fm.contentsOfDirectory(
                at: resourceURL,
                includingPropertiesForKeys: [.isDirectoryKey],
                options: [.skipsHiddenFiles]
            )
        } catch {
            return []
        }

        var bundles: [RPFQViTModelBundle] = []
        for url in candidates where url.pathExtension.lowercased() == "bundle" {
            let configURL = url.appendingPathComponent("config.json")
            guard fm.fileExists(atPath: configURL.path),
                  let configData = try? Data(contentsOf: configURL),
                  let config = try? JSONDecoder().decode(RPFQViTConfiguration.self, from: configData)
            else { continue }

            let manifestURL = url.appendingPathComponent("manifest.json")
            let manifest: RPFQViTBundleManifest?
            if fm.fileExists(atPath: manifestURL.path),
               let manifestData = try? Data(contentsOf: manifestURL),
               let decoded = try? JSONDecoder().decode(RPFQViTBundleManifest.self, from: manifestData)
            {
                manifest = decoded
            } else {
                manifest = nil
            }

            // Stable identity: bundle folder name.
            let id = url.lastPathComponent

            // Prefer a human-friendly tag derived from the checkpoint path.
            let tag: String? = {
                guard let checkpoint = manifest?.checkpoint, !checkpoint.isEmpty else { return nil }
                let parts = checkpoint.split(separator: "/")
                // ".../rpfq_vit-b16-q4-rot/.../model_best.pth.tar" -> "rpfq_vit-b16-q4-rot"
                if let parent = parts.dropLast().last { return String(parent) }
                return nil
            }()

            let baseName: String = config.variant.contains("base") ? "RPFQViT Base/16" :
                (config.variant.contains("small") ? "RPFQViT Small/16" : config.variant)

            let displayName = tag.map { "\(baseName) (\($0))" } ?? baseName
            bundles.append(
                RPFQViTModelBundle(
                    id: id,
                    displayName: displayName,
                    url: url,
                    exportVariant: config.variant,
                    checkpoint: manifest?.checkpoint
                )
            )
        }

        return bundles.sorted { $0.displayName.localizedStandardCompare($1.displayName) == .orderedAscending }
    }
}
