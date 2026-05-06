package com.rpfq_vit.android

import android.graphics.Bitmap
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import com.rpfq_vit.android.data.InferenceProfile
import com.rpfq_vit.android.data.PredictionResult
import com.rpfq_vit.android.model.AssetBundleInstaller
import com.rpfq_vit.android.model.LabelLoader
import com.rpfq_vit.android.model.ModelMetadata
import com.rpfq_vit.android.preprocess.ImagePreprocessor
import com.rpfq_vit.android.preprocess.PreprocessConfig
import com.rpfq_vit.android.runtime.NativeRPFQViTEngine
import com.rpfq_vit.android.ui.HomeScreen
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import kotlin.math.exp

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            RPFQViTAndroidApp()
        }
    }
}

@Composable
private fun RPFQViTAndroidApp() {
    val context = LocalContext.current
    val scope = rememberCoroutineScope()
    val engine = remember { NativeRPFQViTEngine() }

    var bundleDir by remember { mutableStateOf<File?>(null) }
    var metadata by remember { mutableStateOf<ModelMetadata?>(null) }
    var preprocessConfig by remember { mutableStateOf<PreprocessConfig?>(null) }
    var labels by remember { mutableStateOf(emptyList<String>()) }
    var selectedBitmap by remember { mutableStateOf<Bitmap?>(null) }
    var predictions by remember { mutableStateOf(emptyList<PredictionResult>()) }
    var profile by remember { mutableStateOf<InferenceProfile?>(null) }
    var debugEnabled by remember { mutableStateOf(false) }
    var status by remember { mutableStateOf("Loading model bundle...") }
    var running by remember { mutableStateOf(false) }

    DisposableEffect(Unit) {
        onDispose { engine.close() }
    }

    LaunchedEffect(Unit) {
        runCatching {
            withContext(Dispatchers.IO) {
                val installed = AssetBundleInstaller.ensureInstalled(context)
                val meta = AssetBundleInstaller.loadMetadata(installed)
                val preprocess = AssetBundleInstaller.loadPreprocessConfig(installed)
                val loadedLabels = LabelLoader.load(installed)
                check(engine.load(installed.absolutePath)) { "Native runtime rejected the model bundle." }
                LoadedBundle(installed, meta, preprocess, loadedLabels)
            }
        }.onSuccess { loaded ->
            val (installed, meta, preprocess, loadedLabels) = loaded
            bundleDir = installed
            metadata = meta
            preprocessConfig = preprocess
            labels = loadedLabels
            status = "Ready"
        }.onFailure {
            status = "Load failed: ${it.message ?: it::class.java.simpleName}"
        }
    }

    val picker = rememberLauncherForActivityResult(ActivityResultContracts.PickVisualMedia()) { uri ->
        if (uri == null) return@rememberLauncherForActivityResult
        scope.launch {
            runCatching {
                withContext(Dispatchers.IO) {
                    ImagePreprocessor.decodeBitmap(context.contentResolver, uri)
                }
            }.onSuccess {
                selectedBitmap = it
                predictions = emptyList()
                profile = null
                status = "Ready"
            }.onFailure {
                status = "Image decode failed: ${it.message ?: it::class.java.simpleName}"
            }
        }
    }

    MaterialTheme {
        Surface(modifier = Modifier.fillMaxSize(), color = MaterialTheme.colorScheme.background) {
            HomeScreen(
                metadata = metadata,
                selectedBitmap = selectedBitmap,
                predictions = predictions,
                profile = profile,
                debugEnabled = debugEnabled,
                status = status,
                running = running,
                onDebugChanged = { debugEnabled = it },
                onSelectImage = {
                    picker.launch(PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly))
                },
                onRun = {
                    val bitmap = selectedBitmap
                    val currentBundle = bundleDir
                    val currentPreprocess = preprocessConfig
                    if (bitmap == null || currentBundle == null || currentPreprocess == null) {
                        status = "Select an image after the model is ready."
                    } else {
                        scope.launch {
                            running = true
                            status = "Running classification..."
                            runCatching {
                                withContext(Dispatchers.Default) {
                                    val totalStart = System.nanoTime()
                                    val preprocessStart = System.nanoTime()
                                    val input = ImagePreprocessor.preprocess(bitmap, currentPreprocess)
                                    val preprocessMs = elapsedMs(preprocessStart)

                                    val inferenceStart = System.nanoTime()
                                    val debugDir = File(context.filesDir, "debug").absolutePath
                                    engine.setDebug(debugEnabled, debugDir)
                                    val logits = engine.predict(input)
                                    val inferenceMs = elapsedMs(inferenceStart)

                                    val postStart = System.nanoTime()
                                    val top = topK(logits, labels, 5)
                                    val postMs = elapsedMs(postStart)
                                    top to InferenceProfile(
                                        preprocessMs = preprocessMs,
                                        inferenceMs = inferenceMs,
                                        postprocessMs = postMs,
                                        totalMs = elapsedMs(totalStart)
                                    )
                                }
                            }.onSuccess { (top, timing) ->
                                predictions = top
                                profile = timing
                                status = if (debugEnabled) {
                                    "Done. Debug tensors: ${File(currentBundle.parentFile, "debug").absolutePath}"
                                } else {
                                    "Done"
                                }
                            }.onFailure {
                                status = "Prediction failed: ${it.message ?: it::class.java.simpleName}"
                            }
                            running = false
                        }
                    }
                }
            )
        }
    }
}

private data class LoadedBundle(
    val bundleDir: File,
    val metadata: ModelMetadata,
    val preprocessConfig: PreprocessConfig,
    val labels: List<String>
)

private fun elapsedMs(startNs: Long): Long {
    return (System.nanoTime() - startNs) / 1_000_000L
}

private fun topK(
    logits: FloatArray,
    labels: List<String>,
    k: Int
): List<PredictionResult> {
    if (logits.isEmpty()) return emptyList()
    val max = logits.maxOrNull() ?: 0f
    val probs = FloatArray(logits.size)
    var sum = 0.0
    logits.forEachIndexed { index, value ->
        val p = exp((value - max).toDouble())
        probs[index] = p.toFloat()
        sum += p
    }
    if (sum <= 0.0) sum = 1.0
    for (i in probs.indices) probs[i] = (probs[i] / sum).toFloat()

    return probs.indices
        .sortedByDescending { probs[it] }
        .take(k)
        .map { index ->
            PredictionResult(
                index = index,
                label = labels.getOrNull(index) ?: "Class $index",
                score = probs[index]
            )
        }
}
