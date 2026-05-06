import Foundation
import Observation
import SwiftUI

struct ContentView: View {
    @State private var modelManager = ModelManager()

    var body: some View {
        platformContentView
            .task {
                await modelManager.loadModelIfNeeded()
            }
    }

    @ViewBuilder
    private var platformContentView: some View {
        #if os(iOS)
        IOSContentView(modelManager: modelManager)
        #elseif os(macOS)
        MacOSContentView(modelManager: modelManager)
        #endif
    }
}

struct StatusCard: View {
    let modelManager: ModelManager

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Status")
                .font(.headline)

            Text(modelManager.loadedBundle?.displayName ?? "No model selected")
                .font(.subheadline)
                .foregroundStyle(.secondary)

            switch modelManager.state {
            case .idle:
                Text("Idle")
                    .foregroundStyle(.secondary)
            case .loading(let message):
                HStack(spacing: 10) {
                    ProgressView()
                    Text(message)
                        .foregroundStyle(.secondary)
                }
            case .ready:
                Text("Model ready")
                    .foregroundStyle(.green)
            case .running(let message):
                HStack(spacing: 10) {
                    ProgressView()
                    Text(message)
                        .foregroundStyle(.secondary)
                }
            case .error(let message):
                Text(message)
                    .foregroundStyle(.red)
            }

            if !modelManager.availableBundles.isEmpty {
                Text("Available bundles: \(modelManager.availableBundles.map(\.displayName).joined(separator: ", "))")
                    .font(.footnote)
                    .foregroundStyle(.secondary)
            }
        }
        .contentCardStyle()
    }
}

struct PreviewCard: View {
    let image: PlatformImage?

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Image")
                .font(.headline)

            Group {
                if let image {
                    Image(platformImage: image)
                        .resizable()
                        .scaledToFit()
                        .frame(maxWidth: .infinity)
                        .frame(height: 260)
                        .clipShape(RoundedRectangle(cornerRadius: 18))
                } else {
                    RoundedRectangle(cornerRadius: 18)
                        .fill(Color.secondary.opacity(0.12))
                        .frame(height: 260)
                        .overlay {
                            VStack(spacing: 8) {
                                Image(systemName: "photo.on.rectangle.angled")
                                    .font(.system(size: 32))
                                    .foregroundStyle(.secondary)
                                Text("Choose an image to run RPFQViT classification")
                                    .foregroundStyle(.secondary)
                            }
                        }
                }
            }
        }
        .contentCardStyle()
    }
}

struct ModelPickerControl: View {
    @Bindable var modelManager: ModelManager

    var body: some View {
        if modelManager.availableBundles.count > 1 {
            if modelManager.availableBundles.count <= 3 {
                Picker("Model", selection: Binding(
                    get: { modelManager.selectedBundleID ?? "" },
                    set: { newValue in modelManager.selectedBundleID = newValue }
                )) {
                    ForEach(modelManager.availableBundles) { bundle in
                        Text(bundle.displayName).tag(bundle.id)
                    }
                }
                .pickerStyle(.segmented)
                .onChange(of: modelManager.selectedBundleID, initial: false) {
                    Task { await modelManager.reloadSelectedModel() }
                }
            } else {
                Picker("Model", selection: Binding(
                    get: { modelManager.selectedBundleID ?? "" },
                    set: { newValue in modelManager.selectedBundleID = newValue }
                )) {
                    ForEach(modelManager.availableBundles) { bundle in
                        Text(bundle.displayName).tag(bundle.id)
                    }
                }
                .pickerStyle(.menu)
                .onChange(of: modelManager.selectedBundleID, initial: false) {
                    Task { await modelManager.reloadSelectedModel() }
                }
            }
        }
    }
}

struct ResultCard: View {
    let modelManager: ModelManager

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Result")
                .font(.headline)

            if !modelManager.predictions.isEmpty {
                ForEach(Array(modelManager.predictions.prefix(5).enumerated()), id: \.element.id) { rank, prediction in
                    HStack(alignment: .firstTextBaseline, spacing: 10) {
                        Text("#\(rank + 1)")
                            .font(.footnote.weight(.semibold))
                            .foregroundStyle(.secondary)
                            .frame(width: 34, alignment: .leading)

                        VStack(alignment: .leading, spacing: 2) {
                            Text(prediction.label)
                                .font(rank == 0 ? .title3.weight(.semibold) : .body.weight(.semibold))
                            Text("Class \(prediction.index) · \(String(format: "%.2f%%", prediction.confidence * 100))")
                                .font(.footnote)
                                .foregroundStyle(.secondary)
                        }
                    }
                }

                if let profile = modelManager.profile {
                    Divider()
                    Text(String(format: "Preprocess %.1f ms", profile.preprocessMs))
                        .foregroundStyle(.secondary)
                    Text(String(format: "Inference %.1f ms", profile.inferenceMs))
                        .foregroundStyle(.secondary)
                }

                if modelManager.debugExportEnabled,
                   let lastDir = UserDefaults.standard.string(forKey: "RPFQ_VIT_DEBUG_LAST_RUN_DIR")
                {
                    Divider()
                    Text("Debug export directory")
                        .font(.footnote.weight(.semibold))
                    Text(lastDir)
                        .font(.footnote)
                        .foregroundStyle(.secondary)
                        .textSelection(.enabled)
                }
            } else {
                Text("No prediction yet.")
                    .foregroundStyle(.secondary)
            }
        }
        .contentCardStyle()
    }
}

extension View {
    func contentCardStyle() -> some View {
        frame(maxWidth: .infinity, alignment: .leading)
            .padding(18)
            .background(.thinMaterial, in: RoundedRectangle(cornerRadius: 20))
    }
}

#Preview {
    ContentView()
}
