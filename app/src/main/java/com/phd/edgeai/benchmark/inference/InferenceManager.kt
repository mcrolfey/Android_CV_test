package com.phd.edgeai.benchmark.inference

import android.content.Context
import android.graphics.Bitmap
import android.graphics.RectF
import androidx.camera.core.ImageProxy

data class InferenceResult(
    val detections: List<Detection>,
    val latencyMs: Double,
    val frameWidth: Int,
    val frameHeight: Int
)

/**
 * Routes a camera frame through Architecture A or B, timed end-to-end with System.nanoTime().
 * Intended to be called from a coroutine on Dispatchers.Default so the calling thread (CameraX's
 * analysis executor / main thread) is never blocked by inference.
 *
 * Architecture A reproduces the reference batch pipeline's cascade exactly:
 *   ROI YOLO (top-1, conf 0.25/iou 0.5) on the full frame
 *     -> crop -> PA YOLO (up to 100, conf 0.10/iou 0.5) on the ROI crop
 *       -> crop each PA box -> binary ResNet (threshold 0.4) -> subtype ResNet (min conf 0.25)
 *
 * Architecture B is the standalone ROI detector only, uncapped, as the fast single-stage
 * detection baseline described in the original spec.
 */
class InferenceManager(context: Context) {

    private val roiDetector = YoloDetector(
        context,
        modelAssetPath = "models/roi_detector.onnx",
        confThreshold = 0.25f,
        iouThreshold = 0.5f
    )

    private val paDetector = YoloDetector(
        context,
        modelAssetPath = "models/pa_detector.onnx",
        confThreshold = 0.10f,
        iouThreshold = 0.5f
    )

    private val classifier = AsbestosClassifier(context)

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
        roiDetector.close()
        paDetector.close()
        classifier.close()
    }

    private fun runArchitectureB(bitmap: Bitmap): List<Detection> {
        return roiDetector.detect(bitmap).map {
            Detection(it.boundingBox, DetectionKind.ROI, "ROI", it.confidence)
        }
    }

    private fun runArchitectureA(bitmap: Bitmap): List<Detection> {
        val roiRaw = roiDetector.detect(bitmap, maxDetections = 1).firstOrNull() ?: return emptyList()

        val detections = mutableListOf<Detection>()
        detections.add(Detection(roiRaw.boundingBox, DetectionKind.ROI, "ROI", roiRaw.confidence))

        val roiCrop = ImageUtils.cropBitmap(bitmap, roiRaw.boundingBox) ?: return detections
        val offsetX = roiRaw.boundingBox.left
        val offsetY = roiRaw.boundingBox.top

        val paDetections = paDetector.detect(roiCrop, maxDetections = 100)
        for (pa in paDetections) {
            val crop = ImageUtils.cropBitmap(roiCrop, pa.boundingBox) ?: continue
            val classification = classifier.classify(crop)

            val fullFrameBox = RectF(
                offsetX + pa.boundingBox.left,
                offsetY + pa.boundingBox.top,
                offsetX + pa.boundingBox.right,
                offsetY + pa.boundingBox.bottom
            )

            detections.add(Detection(fullFrameBox, DetectionKind.FIBER, classification.label, classification.confidence))
        }

        return detections
    }
}
