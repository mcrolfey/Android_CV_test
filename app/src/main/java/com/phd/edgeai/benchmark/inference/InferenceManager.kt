package com.phd.edgeai.benchmark.inference

import android.content.Context
import android.graphics.Bitmap
import androidx.camera.core.ImageProxy

data class InferenceResult(
    val detections: List<Detection>,
    val latencyMs: Double,
    val frameWidth: Int,
    val frameHeight: Int
)

/**
 * Routes a camera frame through Architecture A (YOLO ROI -> per-box ResNet classification) or
 * Architecture B (YOLO only), timed end-to-end with System.nanoTime(). Intended to be called from
 * a coroutine on Dispatchers.Default so the calling thread (CameraX's analysis executor / main
 * thread) is never blocked by inference.
 */
class InferenceManager(context: Context) {

    private val yoloDetector = YoloDetector(context)
    private val resNetClassifier = ResNetClassifier(context)

    fun runInference(imageProxy: ImageProxy, architecture: Architecture): InferenceResult {
        val startTime = System.nanoTime()

        val bitmap = ImageUtils.imageProxyToBitmap(imageProxy)
        imageProxy.close()

        val detections = when (architecture) {
            Architecture.ARCHITECTURE_B_SINGLESTAGE -> runArchitectureB(bitmap)
            Architecture.ARCHITECTURE_A_MULTISTAGE -> runArchitectureA(bitmap)
        }

        val latencyMs = (System.nanoTime() - startTime) / 1_000_000.0
        return InferenceResult(detections, latencyMs, bitmap.width, bitmap.height)
    }

    fun close() {
        yoloDetector.close()
        resNetClassifier.close()
    }

    private fun runArchitectureB(bitmap: Bitmap): List<Detection> {
        return yoloDetector.detect(bitmap)
    }

    private fun runArchitectureA(bitmap: Bitmap): List<Detection> {
        val roiDetections = yoloDetector.detect(bitmap)
        return roiDetections.map { detection ->
            val crop = ImageUtils.cropBitmap(bitmap, detection.boundingBox)
            val classification = resNetClassifier.classify(crop)
            detection.copy(
                classificationLabel = classification.label,
                classificationConfidence = classification.confidence
            )
        }
    }
}
