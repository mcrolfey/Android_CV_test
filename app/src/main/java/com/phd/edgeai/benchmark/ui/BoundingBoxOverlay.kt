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
import kotlin.math.min

@Composable
fun BoundingBoxOverlay(
    detections: List<Detection>,
    frameWidth: Int,
    frameHeight: Int,
    modifier: Modifier = Modifier
) {
    if (frameWidth == 0 || frameHeight == 0) return

    Canvas(modifier = modifier) {
        // PreviewView.ScaleType.FIT_CENTER is a "contain" fit: uniform scale (the constraining
        // axis is whichever leaves the other with slack), centered, with letterbox bars on the
        // slack axis -- it does NOT stretch the frame to fill the view. Independent scaleX/scaleY
        // (stretch-to-fill) only lines boxes up when the frame's aspect ratio happens to exactly
        // match the view's, which is why they were drifting off the real content on-device.
        val scale = min(size.width / frameWidth, size.height / frameHeight)
        val offsetX = (size.width - frameWidth * scale) / 2f
        val offsetY = (size.height - frameHeight * scale) / 2f

        detections.forEach { detection ->
            val color = if (detection.kind == DetectionKind.ROI) Color.Blue else Color.Red
            val nativeColor = if (detection.kind == DetectionKind.ROI) {
                android.graphics.Color.BLUE
            } else {
                android.graphics.Color.RED
            }

            val box = detection.boundingBox
            val topLeft = Offset(offsetX + box.left * scale, offsetY + box.top * scale)
            val boxSize = Size((box.right - box.left) * scale, (box.bottom - box.top) * scale)

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
