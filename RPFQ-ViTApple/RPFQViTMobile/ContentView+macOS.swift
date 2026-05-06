#if os(macOS)
import AppKit
import Foundation
import Observation
import SwiftUI
import UniformTypeIdentifiers

private enum MacImageImportError: LocalizedError {
    case failedToDecode

    var errorDescription: String? {
        switch self {
        case .failedToDecode:
            return "The selected file could not be decoded as an image."
        }
    }
}

struct MacOSContentView: View {
    @Bindable var modelManager: ModelManager
    @State private var isImporterPresented = false

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 20) {
                    StatusCard(modelManager: modelManager)
                    PreviewCard(image: modelManager.selectedImage)
                    actionsCard
                    ResultCard(modelManager: modelManager)
                }
                .padding(20)
                .frame(maxWidth: 760)
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
            .navigationTitle("RPFQViT for macOS")
            .fileImporter(
                isPresented: $isImporterPresented,
                allowedContentTypes: [.image],
                allowsMultipleSelection: false
            ) { result in
                handleImport(result)
            }
        }
    }

    private var actionsCard: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text("Actions")
                .font(.headline)

            ModelPickerControl(modelManager: modelManager)

            Button {
                isImporterPresented = true
            } label: {
                Label("Choose Image File", systemImage: "folder.badge.plus")
                    .frame(maxWidth: .infinity)
            }
            .buttonStyle(.borderedProminent)

            Button {
                Task { await modelManager.runPrediction() }
            } label: {
                Label("Run Again", systemImage: "play.circle.fill")
                    .frame(maxWidth: .infinity)
            }
            .buttonStyle(.bordered)
            .disabled(modelManager.selectedImage == nil || !modelManager.state.isReady)
        }
        .contentCardStyle()
    }

    private func handleImport(_ result: Result<[URL], Error>) {
        switch result {
        case .success(let urls):
            guard let url = urls.first else { return }

            do {
                let image = try loadImage(from: url)
                modelManager.setImage(image)
                Task { await modelManager.runPrediction() }
            } catch {
                modelManager.state = .error("Failed to load selected image: \(error.localizedDescription)")
            }
        case .failure(let error):
            modelManager.state = .error("Failed to load selected image: \(error.localizedDescription)")
        }
    }

    private func loadImage(from url: URL) throws -> NSImage {
        let isAccessingScopedResource = url.startAccessingSecurityScopedResource()
        defer {
            if isAccessingScopedResource {
                url.stopAccessingSecurityScopedResource()
            }
        }

        guard let image = NSImage(contentsOf: url) else {
            throw MacImageImportError.failedToDecode
        }

        return image
    }
}
#endif
