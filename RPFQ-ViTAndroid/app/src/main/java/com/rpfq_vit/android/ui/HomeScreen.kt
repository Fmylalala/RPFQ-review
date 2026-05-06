package com.rpfq_vit.android.ui

import android.graphics.Bitmap
import androidx.compose.foundation.Image
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import com.rpfq_vit.android.data.InferenceProfile
import com.rpfq_vit.android.data.PredictionResult
import com.rpfq_vit.android.model.ModelMetadata
import java.util.Locale

@Composable
fun HomeScreen(
    metadata: ModelMetadata?,
    selectedBitmap: Bitmap?,
    predictions: List<PredictionResult>,
    profile: InferenceProfile?,
    debugEnabled: Boolean,
    status: String,
    running: Boolean,
    onDebugChanged: (Boolean) -> Unit,
    onSelectImage: () -> Unit,
    onRun: () -> Unit
) {
    Column(
        modifier = Modifier
            .fillMaxSize()
            .verticalScroll(rememberScrollState())
            .padding(20.dp),
        verticalArrangement = Arrangement.spacedBy(14.dp)
    ) {
        Text(
            text = "RPFQ-ViT Android Demo",
            style = MaterialTheme.typography.headlineSmall,
            fontWeight = FontWeight.SemiBold
        )

        ModelInfoPanel(metadata)

        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(10.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            OutlinedButton(onClick = onSelectImage, modifier = Modifier.weight(1f)) {
                Text("Select Image")
            }
            Button(
                onClick = onRun,
                enabled = selectedBitmap != null && metadata != null && !running,
                modifier = Modifier.weight(1f)
            ) {
                if (running) {
                    CircularProgressIndicator(
                        modifier = Modifier
                            .height(18.dp)
                            .width(18.dp),
                        strokeWidth = 2.dp
                    )
                } else {
                    Text("Run Classification")
                }
            }
        }

        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically
        ) {
            Text("Debug Export", style = MaterialTheme.typography.bodyLarge)
            Switch(checked = debugEnabled, onCheckedChange = onDebugChanged)
        }

        Text(
            text = status,
            style = MaterialTheme.typography.bodyMedium,
            color = MaterialTheme.colorScheme.onSurfaceVariant
        )

        PreviewPanel(selectedBitmap)
        ResultPanel(predictions)
        RuntimePanel(profile)
    }
}

@Composable
private fun ModelInfoPanel(metadata: ModelMetadata?) {
    Card(shape = RoundedCornerShape(8.dp), colors = CardDefaults.cardColors()) {
        Column(modifier = Modifier.padding(14.dp), verticalArrangement = Arrangement.spacedBy(6.dp)) {
            InfoRow("Variant", metadata?.variant ?: "-")
            InfoRow("Quant", metadata?.quantMethod ?: "-")
            InfoRow("Backend", metadata?.backend ?: "-")
            InfoRow("Input", metadata?.let { "1x3x${it.inputSize}x${it.inputSize}" } ?: "-")
            InfoRow("Bundle", metadata?.bundleName ?: "-")
        }
    }
}

@Composable
private fun PreviewPanel(bitmap: Bitmap?) {
    Card(shape = RoundedCornerShape(8.dp)) {
        Column(modifier = Modifier.padding(14.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
            Text("Preview", style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.Medium)
            if (bitmap == null) {
                Text("-", color = MaterialTheme.colorScheme.onSurfaceVariant)
            } else {
                Image(
                    bitmap = bitmap.asImageBitmap(),
                    contentDescription = null,
                    modifier = Modifier
                        .fillMaxWidth()
                        .aspectRatio(1.25f),
                    contentScale = ContentScale.Crop
                )
            }
        }
    }
}

@Composable
private fun ResultPanel(predictions: List<PredictionResult>) {
    Card(shape = RoundedCornerShape(8.dp)) {
        Column(modifier = Modifier.padding(14.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
            Text("Results", style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.Medium)
            if (predictions.isEmpty()) {
                Text("-", color = MaterialTheme.colorScheme.onSurfaceVariant)
                return@Column
            }
            predictions.forEachIndexed { rank, result ->
                Column(verticalArrangement = Arrangement.spacedBy(2.dp)) {
                    Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
                        Text(
                            text = "${rank + 1}. ${result.label}",
                            modifier = Modifier.weight(1f),
                            maxLines = 1,
                            overflow = TextOverflow.Ellipsis
                        )
                        Text(formatScore(result.score), fontWeight = FontWeight.Medium)
                    }
                }
                if (rank != predictions.lastIndex) HorizontalDivider()
            }
        }
    }
}

@Composable
private fun RuntimePanel(profile: InferenceProfile?) {
    Card(shape = RoundedCornerShape(8.dp)) {
        Column(modifier = Modifier.padding(14.dp), verticalArrangement = Arrangement.spacedBy(6.dp)) {
            Text("Runtime", style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.Medium)
            InfoRow("Preprocess", profile?.let { "${it.preprocessMs} ms" } ?: "-")
            InfoRow("Inference", profile?.let { "${it.inferenceMs} ms" } ?: "-")
            InfoRow("Postprocess", profile?.let { "${it.postprocessMs} ms" } ?: "-")
            InfoRow("Total", profile?.let { "${it.totalMs} ms" } ?: "-")
        }
    }
}

@Composable
private fun InfoRow(label: String, value: String) {
    Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
        Text(label, color = MaterialTheme.colorScheme.onSurfaceVariant)
        Spacer(Modifier.width(12.dp))
        Text(value, maxLines = 1, overflow = TextOverflow.Ellipsis)
    }
}

private fun formatScore(score: Float): String {
    return String.format(Locale.US, "%.2f%%", score * 100f)
}
