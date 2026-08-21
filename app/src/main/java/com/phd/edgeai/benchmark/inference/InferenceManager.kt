package com.phd.edgeai.benchmark.inference

import android.content.Context
import android.graphics.Bitmap
import android.graphics.RectF
import android.util.Log
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
 * Architecture B is a separately-trained, single-stage YOLO-nano object detector (same dataset,
 * no cascade) -- the point of the app is comparing this single model's on-device latency/FPS/
 * thermal behavior directly against Architecture A's 4-stage pipeline.
 */
class InferenceManager(context: Context) {

    private val roiDetector = YoloDetector(
        context,
        modelAssetPath = "models/roi_detector.onnx",
        labels = listOf("CS"),
        inputSize = 640,
        confThreshold = 0.25f,
        iouThreshold = 0.5f
    )

    // Trained/exported at imgsz=1024 (the YOLOm PA model), unlike the other 640-trained detectors.
    private val paDetector = YoloDetector(
        context,
        modelAssetPath = "models/pa_detector.onnx",
        labels = listOf("pa"),
        inputSize = 1024,
        confThreshold = 0.10f,
        iouThreshold = 0.5f
    )

    private val architectureBDetector = YoloDetector(
        context,
        modelAssetPath = "models/yolo_nano_detector.onnx",
        labels = listOf("A-AM", "A-CF", "A-COF", "A-CP", "A-CRO", "NA-CS", "NA-OF"),
        inputSize = 640,
        confThreshold = 0.25f,
        iouThreshold = 0.5f
    )

    private val classifier = AsbestosClassifier(context)

    /**
     * Runs one throwaway inference through every model on a blank frame. ONNX Runtime (and NNAPI
     * underneath it) does lazy setup on a session's first real call -- memory arena growth, kernel
     * selection, NNAPI compilation -- so without this, whichever detector/classifier hasn't been
     * used yet pays that one-time cost on the first frame after switching architecture, which is
     * exactly the stutter felt when flipping between Architecture A and B. Call once, off the main
     * thread, before the user can interact with the architecture toggle.
     */
    fun warmUp() {
        try {
            val dummy640 = Bitmap.createBitmap(640, 640, Bitmap.Config.ARGB_8888)
            val dummy1024 = Bitmap.createBitmap(1024, 1024, Bitmap.Config.ARGB_8888)
            val dummy224 = Bitmap.createBitmap(224, 224, Bitmap.Config.ARGB_8888)

            roiDetector.detect(dummy640, maxDetections = 1)
            paDetector.detect(dummy1024, maxDetections = 100)
            architectureBDetector.detect(dummy640)
            classifier.classify(dummy224)
        } catch (e: Exception) {
            Log.w("InferenceManager", "Warm-up pass failed", e)
        }
    }

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
        architectureBDetector.close()
        classifier.close()
    }

    private fun runArchitectureB(bitmap: Bitmap): List<Detection> {
        return architectureBDetector.detect(bitmap).map {
            Detection(it.boundingBox, DetectionKind.FIBER, it.label, it.confidence)
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
