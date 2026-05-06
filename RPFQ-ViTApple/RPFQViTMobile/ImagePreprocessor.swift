import CoreGraphics
import Foundation
import MLX

enum ImagePreprocessorError: LocalizedError {
    case failedToCreateContext
    case invalidConfiguration(String)

    var errorDescription: String? {
        switch self {
        case .failedToCreateContext:
            return "Failed to create an RGBA bitmap context."
        case .invalidConfiguration(let message):
            return message
        }
    }
}

final class ImagePreprocessor {
    let config: RPFQViTPreprocessConfiguration

    init(config: RPFQViTPreprocessConfiguration) {
        self.config = config
    }

    func preparePatches(from cgImage: CGImage) throws -> MLXArray {
        try validateConfiguration()

        let width = config.imageSize
        let height = config.imageSize
        let bytesPerPixel = 4
        let bytesPerRow = width * bytesPerPixel
        var rgba = [UInt8](repeating: 0, count: width * height * bytesPerPixel)

        let colorSpace = CGColorSpaceCreateDeviceRGB()
        // Use a well-defined and widely supported pixel format on Apple platforms:
        // BGRA (byteOrder32Little + premultipliedFirst). We'll explicitly swizzle to RGB below.
        let bitmapInfo =
            CGImageAlphaInfo.premultipliedFirst.rawValue | CGBitmapInfo.byteOrder32Little.rawValue
        guard let context = CGContext(
            data: &rgba,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: bytesPerRow,
            space: colorSpace,
            bitmapInfo: bitmapInfo
        ) else {
            throw ImagePreprocessorError.failedToCreateContext
        }

        context.interpolationQuality = interpolationQuality(for: config.interpolation)
        context.setFillColor(CGColor(red: 0, green: 0, blue: 0, alpha: 1))
        context.fill(CGRect(x: 0, y: 0, width: width, height: height))
        context.translateBy(x: 0, y: CGFloat(height))
        context.scaleBy(x: 1, y: -1)

        let drawRect = preprocessingDrawRect(
            sourceSize: CGSize(width: cgImage.width, height: cgImage.height),
            targetSize: CGSize(width: width, height: height)
        )
        context.draw(cgImage, in: drawRect)

        var chw = [Float](repeating: 0, count: config.numChannels * width * height)
        for y in 0 ..< height {
            for x in 0 ..< width {
                let rgbaIndex = (y * width + x) * bytesPerPixel
                // BGRA memory layout for the chosen CGContext bitmapInfo.
                let b = Float(rgba[rgbaIndex]) / 255.0
                let g = Float(rgba[rgbaIndex + 1]) / 255.0
                let r = Float(rgba[rgbaIndex + 2]) / 255.0
                let pixelIndex = y * width + x

                chw[pixelIndex] = (r - config.mean[0]) / config.std[0]
                chw[width * height + pixelIndex] = (g - config.mean[1]) / config.std[1]
                chw[2 * width * height + pixelIndex] = (b - config.mean[2]) / config.std[2]
            }
        }

        let patch = config.patchSize
        let grid = width / patch
        let patchArea = patch * patch
        let patchVectorSize = patchArea * config.numChannels
        var patchBuffer = [Float](repeating: 0, count: grid * grid * patchVectorSize)

        for py in 0 ..< grid {
            for px in 0 ..< grid {
                let patchIndex = py * grid + px
                let patchOffset = patchIndex * patchVectorSize

                for channel in 0 ..< config.numChannels {
                    let channelOffset = channel * width * height
                    let dstChannelOffset = patchOffset + channel * patchArea

                    for dy in 0 ..< patch {
                        for dx in 0 ..< patch {
                            let srcY = py * patch + dy
                            let srcX = px * patch + dx
                            let srcIndex = channelOffset + srcY * width + srcX
                            let dstIndex = dstChannelOffset + dy * patch + dx
                            patchBuffer[dstIndex] = chw[srcIndex]
                        }
                    }
                }
            }
        }

        return MLXArray(patchBuffer).reshaped(1, grid * grid, patchVectorSize).asType(.float16)
    }

    private func validateConfiguration() throws {
        guard config.imageSize > 0 else {
            throw ImagePreprocessorError.invalidConfiguration("Image size must be positive.")
        }

        guard config.patchSize > 0 else {
            throw ImagePreprocessorError.invalidConfiguration("Patch size must be positive.")
        }

        guard config.imageSize % config.patchSize == 0 else {
            throw ImagePreprocessorError.invalidConfiguration(
                "Image size must be divisible by patch size."
            )
        }

        guard config.numChannels == 3 else {
            throw ImagePreprocessorError.invalidConfiguration(
                "Only 3-channel RGB preprocessing is currently supported."
            )
        }

        guard config.mean.count >= 3, config.std.count >= 3 else {
            throw ImagePreprocessorError.invalidConfiguration(
                "Preprocess mean/std must contain RGB channel values."
            )
        }

        guard config.cropPct > 0 else {
            throw ImagePreprocessorError.invalidConfiguration("crop_pct must be positive.")
        }
    }

    private func preprocessingDrawRect(sourceSize: CGSize, targetSize: CGSize) -> CGRect {
        let resizeMode = config.resizeMode.lowercased()
        let cropMode = config.cropMode.lowercased()

        if resizeMode == "shortest_side", cropMode == "center" {
            let shortestSide = min(sourceSize.width, sourceSize.height)
            let targetShortest = (targetSize.width / CGFloat(config.cropPct)).rounded()
            let scale = targetShortest / shortestSide
            let resizedSize = CGSize(
                width: sourceSize.width * scale,
                height: sourceSize.height * scale
            )
            return CGRect(
                x: (targetSize.width - resizedSize.width) / 2.0,
                y: (targetSize.height - resizedSize.height) / 2.0,
                width: resizedSize.width,
                height: resizedSize.height
            )
        }

        let scale = max(targetSize.width / sourceSize.width, targetSize.height / sourceSize.height)
        let scaledSize = CGSize(
            width: sourceSize.width * scale,
            height: sourceSize.height * scale
        )
        return CGRect(
            x: (targetSize.width - scaledSize.width) / 2.0,
            y: (targetSize.height - scaledSize.height) / 2.0,
            width: scaledSize.width,
            height: scaledSize.height
        )
    }

    private func interpolationQuality(for interpolation: String) -> CGInterpolationQuality {
        switch interpolation.lowercased() {
        case "nearest":
            return .none
        case "bilinear":
            return .medium
        case "bicubic":
            return .high
        default:
            return .default
        }
    }
}
