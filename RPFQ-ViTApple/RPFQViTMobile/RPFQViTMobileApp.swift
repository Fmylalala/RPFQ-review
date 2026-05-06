import SwiftUI

@main
struct RPFQViTMobileApp: App {
    init() {
        InferenceRuntimeSupport.configureGPUCacheIfSupported()
    }

    var body: some Scene {
        WindowGroup {
            ContentView()
        }
    }
}
