#if os(iOS)
import Foundation
import Observation
import PhotosUI
import SwiftUI
import UIKit

struct IOSContentView: View {
    @Bindable var modelManager: ModelManager
    @State private var pickerItem: PhotosPickerItem?

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
            }
            .navigationTitle("RPFQViT Mobile")
            .task(id: pickerItem) {
                guard let pickerItem else { return }

                do {
                    if let data = try await pickerItem.loadTransferable(type: Data.self),
                       let image = UIImage(data: data)
                    {
                        modelManager.setImage(image)
                        await modelManager.runPrediction()
                    } else {
                        modelManager.state = .error("The selected photo could not be decoded.")
                    }
                } catch {
                    modelManager.state = .error("Failed to load selected image: \(error.localizedDescription)")
                }
            }
        }
    }

    private var actionsCard: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text("Actions")
                .font(.headline)

            ModelPickerControl(modelManager: modelManager)

            PhotosPicker(selection: $pickerItem, matching: .images) {
                Label("Choose from Photos", systemImage: "photo.badge.plus")
                    .frame(maxWidth: .infinity)
            }
            .buttonStyle(.borderedProminent)

            Toggle("Export debug tensors", isOn: Binding(
                get: { modelManager.debugExportEnabled },
                set: { newValue in modelManager.setDebugExportEnabled(newValue) }
            ))
            .disabled(!modelManager.state.isReady)

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
}
#endif
