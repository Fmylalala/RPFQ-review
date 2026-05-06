import MLX
import Metal

enum InferenceRuntimeSupport {
    static let cacheLimitBytes = 256 * 1024 * 1024

    static func failureMessage() -> String? {
        #if os(iOS) && targetEnvironment(simulator)
        return "iOS Simulator is supported for UI validation only. Run on a real iPhone or iPad for MLX inference."
        #else
        guard MTLCreateSystemDefaultDevice() != nil else {
            return "Metal is unavailable on this device. RPFQViT requires a Metal-capable iPhone or iPad, or an Apple Silicon Mac."
        }

        return nil
        #endif
    }

    static func configureGPUCacheIfSupported() {
        guard failureMessage() == nil else { return }
        GPU.set(cacheLimit: cacheLimitBytes)
    }

    static func resetGPUCacheIfSupported() {
        guard failureMessage() == nil else { return }
        GPU.clearCache()
        GPU.set(cacheLimit: cacheLimitBytes)
    }
}
