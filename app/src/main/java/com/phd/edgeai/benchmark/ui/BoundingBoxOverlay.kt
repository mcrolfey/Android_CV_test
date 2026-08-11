package com.phd.edgeai.benchmark.ui

import androidx.compose.foundation.Canvas
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.nativeCanvas
import com.phd.edgeai.benchmark.inference.Detection

// Assumes PreviewView.ScaleType.FIT_CENTER so the analysis frame and this canvas share aspect ratio.
@Composable
fun BoundingBoxOverlay(
    detections: List<Detection>,
    frameWidth: Int,
    frameHeight: Int,
    modifier: Modifier = Modifier
) {
    if (frameWidth == 0 || frameHeight == 0) return

    Canvas(modifier = modifier) {
        val scaleX = size.width / frameWidth.toFloat()
        val scaleY = size.height / frameHeight.toFloat()

        val paint = android.graphics.Paint().apply {
            color = android.graphics.Color.GREEN
            textSize = 32f
            isAntiAlias = true
        }

        detections.forEach { detection ->
            val box = detection.boundingBox
            val topLeft = Offset(box.left * scaleX, box.top * scaleY)
            val boxSize = Size((box.right - box.left) * scaleX, (box.bottom - box.top) * scaleY)

            drawRect(color = Color.Green, topLeft = topLeft, size = boxSize, style = Stroke(width = 4f))

            val label = buildString {
                append(detection.roiLabel)
                append(" %.0f%%".format(detection.roiConfidence * 100))
                if (detection.classificationLabel != null) {
                    append(" | ")
                    append(detection.classificationLabel)
                    append(" %.0f%%".format((detection.classificationConfidence ?: 0f) * 100))
                }
            }

            drawContext.canvas.nativeCanvas.drawText(
                label,
                topLeft.x,
                (topLeft.y - 8f).coerceAtLeast(20f),
                paint
            )
        }
    }
}
