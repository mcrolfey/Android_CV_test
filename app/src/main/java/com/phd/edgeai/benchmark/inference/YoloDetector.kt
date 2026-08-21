package com.phd.edgeai.benchmark.inference

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Matrix
import android.graphics.Paint
import android.graphics.RectF
import android.util.Log
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import kotlin.math.max
import kotlin.math.min

data class RawDetection(val boundingBox: RectF, val confidence: Float, val classId: Int, val label: String)

/**
 * Wraps a single-stage YOLO ONNX export (Ultralytics detect head: output shape
 * [1, 4+numClasses, numAnchors], no separate objectness channel). Used for the ROI stage
 * (cement_roi_model), the PA stage (pa_detector), and Architecture B's independently-trained
 * YOLO-nano model -- each call site supplies its own labels, conf/iou thresholds, and
 * maxDetections (ROI: conf 0.25, iou 0.5, top-1; PA: conf 0.10, iou 0.5, up to 100).
 *
 * Preprocessing reuses a fixed inputSize x inputSize letterbox bitmap/canvas/pixel buffer and a
 * single direct FloatBuffer across calls instead of allocating fresh ones every frame -- on
 * Android those repeated large allocations otherwise trigger frequent GC pauses that show up as
 * uneven frame latency.
 *
 * Tries to enable the NNAPI execution provider so supported ops run on the device's GPU/DSP/NPU
 * instead of ONNX Runtime's plain CPU EP -- without it, Android has no hardware acceleration path
 * at all here, unlike an iOS build running the same models through CoreML (which gets Neural
 * Engine acceleration by default). ORT partitions unsupported ops back to CPU automatically, and
 * this falls back to CPU-only entirely if NNAPI itself isn't available on the device.
 */
class YoloDetector(
    context: Context,
    modelAssetPath: String,
    private val labels: List<String> = listOf("object"),
    private val inputSize: Int = 640,
    private val confThreshold: Float = 0.25f,
    private val iouThreshold: Float = 0.5f
) {
    private val ortEnv: OrtEnvironment = OrtEnvironment.getEnvironment()
    private val session: OrtSession
    private val logTag = "Yolo:" + modelAssetPath.substringAfterLast('/')

    private val channelSize = inputSize * inputSize
    private val letterboxBitmap = Bitmap.createBitmap(inputSize, inputSize, Bitmap.Config.ARGB_8888)
    private val letterboxCanvas = Canvas(letterboxBitmap)
    private val letterboxPaint = Paint(Paint.FILTER_BITMAP_FLAG)
    private val letterboxMatrix = Matrix()
    private val pixels = IntArray(channelSize)
    private val inputBuffer: FloatBuffer = ByteBuffer
        .allocateDirect(3 * channelSize * 4)
        .order(ByteOrder.nativeOrder())
        .asFloatBuffer()
    private val inputShape = longArrayOf(1, 3, inputSize.toLong(), inputSize.toLong())

    init {
        val modelBytes = context.assets.open(modelAssetPath).readBytes()
        val options = OrtSession.SessionOptions()
        try {
            options.addNnapi()
        } catch (e: Exception) {
            Log.w(logTag, "NNAPI execution provider unavailable, falling back to CPU", e)
        }
        session = ortEnv.createSession(modelBytes, options)
    }

    fun detect(bitmap: Bitmap, maxDetections: Int = Int.MAX_VALUE): List<RawDetection> {
        val (scale, padX, padY) = preprocess(bitmap)
        OnnxTensor.createTensor(ortEnv, inputBuffer, inputShape).use { tensor ->
            val inputName = session.inputNames.iterator().next()
            session.run(mapOf(inputName to tensor)).use { results ->
                @Suppress("UNCHECKED_CAST")
                val rawOutput = results[0].value as Array<Array<FloatArray>>
                val detections = postprocess(rawOutput, scale, padX, padY, bitmap.width, bitmap.height)
                return detections.take(maxDetections)
            }
        }
    }

    fun close() {
        session.close()
    }

    private data class LetterboxParams(val scale: Float, val padX: Float, val padY: Float)

    private fun preprocess(bitmap: Bitmap): LetterboxParams {
        val scale = min(inputSize.toFloat() / bitmap.width, inputSize.toFloat() / bitmap.height)
        val scaledW = bitmap.width * scale
        val scaledH = bitmap.height * scale
        val padX = (inputSize - scaledW) / 2f
        val padY = (inputSize - scaledH) / 2f

        letterboxMatrix.reset()
        letterboxMatrix.postScale(scale, scale)
        letterboxMatrix.postTranslate(padX, padY)

        letterboxCanvas.drawColor(Color.rgb(114, 114, 114))
        letterboxCanvas.drawBitmap(bitmap, letterboxMatrix, letterboxPaint)

        letterboxBitmap.getPixels(pixels, 0, inputSize, 0, 0, inputSize, inputSize)

        inputBuffer.clear()
        for (i in 0 until channelSize) {
            val p = pixels[i]
            inputBuffer.put(i, ((p shr 16) and 0xFF) / 255f)
            inputBuffer.put(channelSize + i, ((p shr 8) and 0xFF) / 255f)
            inputBuffer.put(2 * channelSize + i, (p and 0xFF) / 255f)
        }

        return LetterboxParams(scale, padX, padY)
    }

    private fun postprocess(
        output: Array<Array<FloatArray>>,
        scale: Float,
        padX: Float,
        padY: Float,
        origWidth: Int,
        origHeight: Int
    ): List<RawDetection> {
        val predictions = output[0]
        val numClasses = predictions.size - 4
        val numAnchors = predictions[0].size

        var maxScoreSeen = 0f
        var maxScoreClassId = -1
        val candidates = mutableListOf<RawDetection>()
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
            if (bestScore > maxScoreSeen) {
                maxScoreSeen = bestScore
                maxScoreClassId = bestClassId
            }
            if (bestScore < confThreshold) continue

            val cx = predictions[0][i]
            val cy = predictions[1][i]
            val w = predictions[2][i]
            val h = predictions[3][i]

            val left = (cx - w / 2f - padX) / scale
            val top = (cy - h / 2f - padY) / scale
            val right = (cx + w / 2f - padX) / scale
            val bottom = (cy + h / 2f - padY) / scale

            candidates.add(
                RawDetection(
                    boundingBox = RectF(
                        left.coerceIn(0f, origWidth.toFloat()),
                        top.coerceIn(0f, origHeight.toFloat()),
                        right.coerceIn(0f, origWidth.toFloat()),
                        bottom.coerceIn(0f, origHeight.toFloat())
                    ),
                    confidence = bestScore,
                    classId = bestClassId,
                    label = labels.getOrElse(bestClassId) { "class_$bestClassId" }
                )
            )
        }

        val topLabel = labels.getOrElse(maxScoreClassId) { "class_$maxScoreClassId" }
        Log.d(
            logTag,
            "maxScore=%.3f (%s) threshold=%.2f aboveThreshold=%d/%d".format(
                maxScoreSeen, topLabel, confThreshold, candidates.size, numAnchors
            )
        )

        // Sorted descending by confidence, so take(maxDetections) upstream reproduces
        // Ultralytics' post-NMS max_det truncation (e.g. top-1 for the ROI stage).
        return nonMaxSuppression(candidates)
    }

    private fun nonMaxSuppression(detections: List<RawDetection>): List<RawDetection> {
        val sorted = detections.sortedByDescending { it.confidence }.toMutableList()
        val kept = mutableListOf<RawDetection>()
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
