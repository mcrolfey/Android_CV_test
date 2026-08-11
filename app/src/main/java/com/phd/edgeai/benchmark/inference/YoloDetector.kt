package com.phd.edgeai.benchmark.inference

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.RectF
import java.nio.FloatBuffer
import kotlin.math.max
import kotlin.math.min

/**
 * Wraps a YOLOv11 ONNX export (Ultralytics detect head: output shape [1, 4+numClasses, numAnchors],
 * no separate objectness channel). Used standalone for Architecture B and as the ROI stage for
 * Architecture A.
 */
class YoloDetector(
    context: Context,
    modelAssetPath: String = "models/roi_detector.onnx",
    private val labels: List<String> = listOf("asbestos"),
    private val inputSize: Int = 640,
    private val confThreshold: Float = 0.25f,
    private val iouThreshold: Float = 0.45f
) {
    private val ortEnv: OrtEnvironment = OrtEnvironment.getEnvironment()
    private val session: OrtSession

    init {
        val modelBytes = context.assets.open(modelAssetPath).readBytes()
        session = ortEnv.createSession(modelBytes, OrtSession.SessionOptions())
    }

    fun detect(bitmap: Bitmap): List<Detection> {
        val preprocessed = preprocess(bitmap)
        preprocessed.tensor.use { tensor ->
            val inputName = session.inputNames.iterator().next()
            session.run(mapOf(inputName to tensor)).use { results ->
                @Suppress("UNCHECKED_CAST")
                val rawOutput = results[0].value as Array<Array<FloatArray>>
                return postprocess(rawOutput, preprocessed, bitmap.width, bitmap.height)
            }
        }
    }

    fun close() {
        session.close()
    }

    private data class Preprocessed(val tensor: OnnxTensor, val scale: Float, val padX: Float, val padY: Float)

    private fun preprocess(bitmap: Bitmap): Preprocessed {
        val scale = min(inputSize.toFloat() / bitmap.width, inputSize.toFloat() / bitmap.height)
        val scaledW = (bitmap.width * scale).toInt().coerceAtLeast(1)
        val scaledH = (bitmap.height * scale).toInt().coerceAtLeast(1)
        val padX = (inputSize - scaledW) / 2f
        val padY = (inputSize - scaledH) / 2f

        val resized = Bitmap.createScaledBitmap(bitmap, scaledW, scaledH, true)
        val letterboxed = Bitmap.createBitmap(inputSize, inputSize, Bitmap.Config.ARGB_8888)
        Canvas(letterboxed).apply {
            drawColor(Color.rgb(114, 114, 114))
            drawBitmap(resized, padX, padY, null)
        }

        val channelSize = inputSize * inputSize
        val pixels = IntArray(channelSize)
        letterboxed.getPixels(pixels, 0, inputSize, 0, 0, inputSize, inputSize)

        val rArr = FloatArray(channelSize)
        val gArr = FloatArray(channelSize)
        val bArr = FloatArray(channelSize)
        for (i in pixels.indices) {
            val p = pixels[i]
            rArr[i] = ((p shr 16) and 0xFF) / 255f
            gArr[i] = ((p shr 8) and 0xFF) / 255f
            bArr[i] = (p and 0xFF) / 255f
        }

        val buffer = FloatBuffer.allocate(3 * channelSize)
        buffer.put(rArr)
        buffer.put(gArr)
        buffer.put(bArr)
        buffer.rewind()

        val shape = longArrayOf(1, 3, inputSize.toLong(), inputSize.toLong())
        return Preprocessed(OnnxTensor.createTensor(ortEnv, buffer, shape), scale, padX, padY)
    }

    private fun postprocess(
        output: Array<Array<FloatArray>>,
        pre: Preprocessed,
        origWidth: Int,
        origHeight: Int
    ): List<Detection> {
        val predictions = output[0]
        val numClasses = predictions.size - 4
        val numAnchors = predictions[0].size

        val candidates = mutableListOf<Detection>()
        for (i in 0 until numAnchors) {
            var bestClassId = -1
            var bestScore = 0f
            for (c in 0 until numClasses) {
                val score = predictions[4 + c][i]
                if (score > bestScore) {
                    bestScore = score
                    bestClassId = c
                }
            }
            if (bestScore < confThreshold) continue

            val cx = predictions[0][i]
            val cy = predictions[1][i]
            val w = predictions[2][i]
            val h = predictions[3][i]

            val left = (cx - w / 2f - pre.padX) / pre.scale
            val top = (cy - h / 2f - pre.padY) / pre.scale
            val right = (cx + w / 2f - pre.padX) / pre.scale
            val bottom = (cy + h / 2f - pre.padY) / pre.scale

            candidates.add(
                Detection(
                    boundingBox = RectF(
                        left.coerceIn(0f, origWidth.toFloat()),
                        top.coerceIn(0f, origHeight.toFloat()),
                        right.coerceIn(0f, origWidth.toFloat()),
                        bottom.coerceIn(0f, origHeight.toFloat())
                    ),
                    roiConfidence = bestScore,
                    roiClassId = bestClassId,
                    roiLabel = labels.getOrElse(bestClassId) { "class_$bestClassId" }
                )
            )
        }
        return nonMaxSuppression(candidates)
    }

    private fun nonMaxSuppression(detections: List<Detection>): List<Detection> {
        val sorted = detections.sortedByDescending { it.roiConfidence }.toMutableList()
        val kept = mutableListOf<Detection>()
        while (sorted.isNotEmpty()) {
            val best = sorted.removeAt(0)
            kept.add(best)
            sorted.removeAll { iou(best.boundingBox, it.boundingBox) > iouThreshold }
        }
        return kept
    }

    private fun iou(a: RectF, b: RectF): Float {
        val interLeft = max(a.left, b.left)
        val interTop = max(a.top, b.top)
        val interRight = min(a.right, b.right)
        val interBottom = min(a.bottom, b.bottom)
        val interArea = max(0f, interRight - interLeft) * max(0f, interBottom - interTop)
        val union = a.width() * a.height() + b.width() * b.height() - interArea
        return if (union <= 0f) 0f else interArea / union
    }
}
