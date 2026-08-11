package com.phd.edgeai.benchmark.inference

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.graphics.Bitmap
import kotlin.math.exp

/**
 * Two-stage ResNet-18 classification cascade applied to a single PA crop, mirroring
 * classify_patch() from the reference batch pipeline exactly:
 *
 *  1. Binary model (2-class: A=0, NA-OF=1). If P(A) < binaryAThreshold, stop and return "NA-OF".
 *  2. Otherwise run the subtype model (3-class: A-AM=0, A-C=1, A-CRO=2). If its top confidence is
 *     below subtypeMinConfidence, fall back to the generic "A" label (confidence = P(A) from the
 *     binary stage); otherwise return the argmax subtype label.
 */
class AsbestosClassifier(
    context: Context,
    binaryModelAssetPath: String = "models/resnet_binary.onnx",
    subtypeModelAssetPath: String = "models/resnet_subtype.onnx",
    private val inputSize: Int = 224,
    private val binaryAThreshold: Float = 0.4f,
    private val subtypeMinConfidence: Float = 0.25f
) {
    private val ortEnv: OrtEnvironment = OrtEnvironment.getEnvironment()
    private val binarySession: OrtSession
    private val subtypeSession: OrtSession

    private val mean = floatArrayOf(0.485f, 0.456f, 0.406f)
    private val std = floatArrayOf(0.229f, 0.224f, 0.225f)

    private val subtypeLabels = mapOf(0 to "A-AM", 1 to "A-C", 2 to "A-CRO")

    init {
        binarySession = ortEnv.createSession(context.assets.open(binaryModelAssetPath).readBytes(), OrtSession.SessionOptions())
        subtypeSession = ortEnv.createSession(context.assets.open(subtypeModelAssetPath).readBytes(), OrtSession.SessionOptions())
    }

    data class ClassificationResult(val label: String, val confidence: Float)

    fun classify(bitmap: Bitmap): ClassificationResult {
        val tensor = preprocess(bitmap)
        try {
            val binaryProbs = runSoftmax(binarySession, tensor)
            val pA = binaryProbs[0]
            val pNa = binaryProbs[1]
            if (pA < binaryAThreshold) {
                return ClassificationResult("NA-OF", pNa)
            }

            val subtypeProbs = runSoftmax(subtypeSession, tensor)
            var bestIdx = 0
            var bestProb = subtypeProbs[0]
            for (i in subtypeProbs.indices) {
                if (subtypeProbs[i] > bestProb) {
                    bestProb = subtypeProbs[i]
                    bestIdx = i
                }
            }

            return if (bestProb < subtypeMinConfidence) {
                ClassificationResult("A", pA)
            } else {
                ClassificationResult(subtypeLabels.getValue(bestIdx), bestProb)
            }
        } finally {
            tensor.close()
        }
    }

    fun close() {
        binarySession.close()
        subtypeSession.close()
    }

    private fun runSoftmax(session: OrtSession, tensor: OnnxTensor): FloatArray {
        val inputName = session.inputNames.iterator().next()
        session.run(mapOf(inputName to tensor)).use { results ->
            @Suppress("UNCHECKED_CAST")
            val logits = (results[0].value as Array<FloatArray>)[0]
            return softmax(logits)
        }
    }

    // Bitmap.createScaledBitmap (bilinear) approximates but doesn't exactly match cv2's
    // INTER_AREA resize used by the reference pipeline for downscaling crops to 224x224.
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

        val buffer = java.nio.FloatBuffer.allocate(3 * channelSize)
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
