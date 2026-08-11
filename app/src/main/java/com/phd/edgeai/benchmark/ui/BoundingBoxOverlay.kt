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
import com.phd.edgeai.benchmark.inference.DetectionKind

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

        detections.forEach { detection ->
            val color = if (detection.kind == DetectionKind.ROI) Color.Blue else Color.Red
            val nativeColor = if (detection.kind == DetectionKind.ROI) {
                android.graphics.Color.BLUE
            } else {
                android.graphics.Color.RED
            }

            val box = detection.boundingBox
            val topLeft = Offset(box.left * scaleX, box.top * scaleY)
            val boxSize = Size((box.right - box.left) * scaleX, (box.bottom - box.top) * scaleY)

            drawRect(color = color, topLeft = topLeft, size = boxSize, style = Stroke(width = 4f))

            val label = "${detection.label} %.2f".format(detection.confidence)
            val paint = android.graphics.Paint().apply {
                this.color = nativeColor
                textSize = 32f
                isAntiAlias = true
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
