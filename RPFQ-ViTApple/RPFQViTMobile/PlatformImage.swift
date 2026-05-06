import CoreGraphics
import Foundation
#if os(iOS)
import UIKit
#elseif os(macOS)
import AppKit
#endif
import SwiftUI

enum PlatformImageError: LocalizedError {
    case failedToCreateCGImage

    var errorDescription: String? {
        switch self {
        case .failedToCreateCGImage:
            return "Failed to convert the selected image into a CGImage."
        }
    }
}

#if os(iOS)
typealias PlatformImage = UIImage

extension PlatformImage {
    func cgImageForInference() throws -> CGImage {
        if imageOrientation == .up, let cgImage {
            return cgImage
        }

        let imageSize = size
        guard imageSize.width > 0, imageSize.height > 0 else {
            throw PlatformImageError.failedToCreateCGImage
        }

        let format = UIGraphicsImageRendererFormat.default()
        format.scale = 1

        let renderer = UIGraphicsImageRenderer(size: imageSize, format: format)
        let normalizedImage = renderer.image { _ in
            draw(in: CGRect(origin: .zero, size: imageSize))
        }

        guard let cgImage = normalizedImage.cgImage else {
            throw PlatformImageError.failedToCreateCGImage
        }

        return cgImage
    }
}

extension Image {
    init(platformImage: PlatformImage) {
        self.init(uiImage: platformImage)
    }
}
#elseif os(macOS)
typealias PlatformImage = NSImage

extension PlatformImage {
    func cgImageForInference() throws -> CGImage {
        var proposedRect = CGRect(origin: .zero, size: size)
        if let cgImage = cgImage(forProposedRect: &proposedRect, context: nil, hints: nil) {
            return cgImage
        }

        guard
            let tiffRepresentation,
            let bitmapRepresentation = NSBitmapImageRep(data: tiffRepresentation),
            let cgImage = bitmapRepresentation.cgImage
        else {
            throw PlatformImageError.failedToCreateCGImage
        }

        return cgImage
    }
}

extension Image {
    init(platformImage: PlatformImage) {
        self.init(nsImage: platformImage)
    }
}
#endif
