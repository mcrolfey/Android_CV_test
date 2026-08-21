package com.phd.edgeai.benchmark.inference

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Matrix
import android.graphics.Paint
import android.util.Log
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import kotlin.math.exp

/**
 * Two-stage ResNet-18 classification cascade applied to a single PA crop, mirroring
 * classify_patch() from the reference batch pipeline exactly:
 *
 *  1. Binary model (2-class: A=0, NA-OF=1). If P(A) < binaryAThreshold, stop and return "NA-OF".
 *  2. Otherwise run the subtype model (3-class: A-AM=0, A-C=1, A-CRO=2). If its top confidence is
 *     below subtypeMinConfidence, fall back to the generic "A" label (confidence = P(A) from the
 *     binary stage); otherwise return the argmax subtype label.
 *
 * Reuses a fixed inputSize x inputSize scaling canvas, pixel buffer, and direct FloatBuffer across
 * calls instead of allocating fresh ones per crop (see YoloDetector for why that matters on
 * Android).
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

    private val channelSize = inputSize * inputSize
    private val scaledBitmap = Bitmap.createBitmap(inputSize, inputSize, Bitmap.Config.ARGB_8888)
    private val scaledCanvas = Canvas(scaledBitmap)
    private val scalePaint = Paint(Paint.FILTER_BITMAP_FLAG)
    private val scaleMatrix = Matrix()
    private val pixels = IntArray(channelSize)
    private val inputBuffer: FloatBuffer = ByteBuffer
        .allocateDirect(3 * channelSize * 4)
        .order(ByteOrder.nativeOrder())
        .asFloatBuffer()
    private val inputShape = longArrayOf(1, 3, inputSize.toLong(), inputSize.toLong())

    init {
        binarySession = ortEnv.createSession(context.assets.open(binaryModelAssetPath).readBytes(), accelerated())
        subtypeSession = ortEnv.createSession(context.assets.open(subtypeModelAssetPath).readBytes(), accelerated())
    }

    // See YoloDetector for why: gives Android a hardware-acceleration path comparable to iOS's
    // CoreML/Neural Engine instead of running everything on ONNX Runtime's plain CPU EP.
    private fun accelerated(): OrtSession.SessionOptions {
        val options = OrtSession.SessionOptions()
        try {
            options.addNnapi()
        } catch (e: Exception) {
            Log.w("AsbestosClassifier", "NNAPI execution provider unavailable, falling back to CPU", e)
        }
        return options
    }

    data class ClassificationResult(val label: String, val confidence: Float)

    fun classify(bitmap: Bitmap): ClassificationResult {
        preprocess(bitmap)
        OnnxTensor.createTensor(ortEnv, inputBuffer, inputShape).use { tensor ->
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

    // Bitmap scaling here (bilinear) approximates but doesn't exactly match cv2's INTER_AREA
    // resize used by the reference pipeline for downscaling crops to 224x224.
    private fun preprocess(bitmap: Bitmap) {
        scaleMatrix.reset()
        scaleMatrix.setScale(inputSize.toFloat() / bitmap.width, inputSize.toFloat() / bitmap.height)
        scaledCanvas.drawBitmap(bitmap, scaleMatrix, scalePaint)
        scaledBitmap.getPixels(pixels, 0, inputSize, 0, 0, inputSize, inputSize)

        inputBuffer.clear()
        for (i in 0 until channelSize) {
            val p = pixels[i]
            inputBuffer.put(i, (((p shr 16) and 0xFF) / 255f - mean[0]) / std[0])
            inputBuffer.put(channelSize + i, (((p shr 8) and 0xFF) / 255f - mean[1]) / std[1])
            inputBuffer.put(2 * channelSize + i, ((p and 0xFF) / 255f - mean[2]) / std[2])
        }
    }

    private fun softmax(logits: FloatArray): FloatArray {
        val maxLogit = logits.max()
        val exps = FloatArray(logits.size) { exp((logits[it] - maxLogit).toDouble()).toFloat() }
        val sum = exps.sum()
        return FloatArray(exps.size) { exps[it] / sum }
    }
}
