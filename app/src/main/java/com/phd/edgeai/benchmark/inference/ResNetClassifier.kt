package com.phd.edgeai.benchmark.inference

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.graphics.Bitmap
import java.nio.FloatBuffer
import kotlin.math.exp

/**
 * Second-stage classifier for Architecture A: runs on each YOLO ROI crop.
 * Assumes a standard ImageNet-normalized ResNet export, softmax applied client-side on raw logits.
 */
class ResNetClassifier(
    context: Context,
    modelAssetPath: String = "models/resnet_classifier.onnx",
    private val labels: List<String> = listOf("non_asbestos", "asbestos"),
    private val inputSize: Int = 224,
    private val mean: FloatArray = floatArrayOf(0.485f, 0.456f, 0.406f),
    private val std: FloatArray = floatArrayOf(0.229f, 0.224f, 0.225f)
) {
    private val ortEnv: OrtEnvironment = OrtEnvironment.getEnvironment()
    private val session: OrtSession

    init {
        val modelBytes = context.assets.open(modelAssetPath).readBytes()
        session = ortEnv.createSession(modelBytes, OrtSession.SessionOptions())
    }

    data class ClassificationResult(val label: String, val confidence: Float)

    fun classify(bitmap: Bitmap): ClassificationResult {
        preprocess(bitmap).use { tensor ->
            val inputName = session.inputNames.iterator().next()
            session.run(mapOf(inputName to tensor)).use { results ->
                @Suppress("UNCHECKED_CAST")
                val logits = (results[0].value as Array<FloatArray>)[0]
                val probs = softmax(logits)

                var bestIdx = 0
                var bestProb = probs[0]
                for (i in probs.indices) {
                    if (probs[i] > bestProb) {
                        bestProb = probs[i]
                        bestIdx = i
                    }
                }
                return ClassificationResult(labels.getOrElse(bestIdx) { "class_$bestIdx" }, bestProb)
            }
        }
    }

    fun close() {
        session.close()
    }

    private fun preprocess(bitmap: Bitmap): OnnxTensor {
        val resized = Bitmap.createScaledBitmap(bitmap, inputSize, inputSize, true)
        val channelSize = inputSize * inputSize
        val pixels = IntArray(channelSize)
        resized.getPixels(pixels, 0, inputSize, 0, 0, inputSize, inputSize)

        val rArr = FloatArray(channelSize)
        val gArr = FloatArray(channelSize)
        val bArr = FloatArray(channelSize)
        for (i in pixels.indices) {
            val p = pixels[i]
            rArr[i] = (((p shr 16) and 0xFF) / 255f - mean[0]) / std[0]
            gArr[i] = (((p shr 8) and 0xFF) / 255f - mean[1]) / std[1]
            bArr[i] = ((p and 0xFF) / 255f - mean[2]) / std[2]
        }

        val buffer = FloatBuffer.allocate(3 * channelSize)
        buffer.put(rArr)
        buffer.put(gArr)
        buffer.put(bArr)
        buffer.rewind()

        val shape = longArrayOf(1, 3, inputSize.toLong(), inputSize.toLong())
        return OnnxTensor.createTensor(ortEnv, buffer, shape)
    }

    private fun softmax(logits: FloatArray): FloatArray {
        val maxLogit = logits.max()
        val exps = FloatArray(logits.size) { exp((logits[it] - maxLogit).toDouble()).toFloat() }
        val sum = exps.sum()
        return FloatArray(exps.size) { exps[it] / sum }
    }
}
