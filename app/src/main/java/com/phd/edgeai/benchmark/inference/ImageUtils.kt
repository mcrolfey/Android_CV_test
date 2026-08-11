package com.phd.edgeai.benchmark.inference

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Matrix
import android.graphics.RectF
import androidx.camera.core.ImageProxy

/**
 * Converts camera frames straight from YUV_420_888 to an ARGB Bitmap with fixed-point BT.601
 * math -- no JPEG encode/decode round trip. That round trip (the common "quick" approach: pack
 * into NV21, YuvImage.compressToJpeg, BitmapFactory.decodeByteArray) is what was causing uneven
 * frame times: JPEG compression time depends on frame content/entropy, so latency swings frame to
 * frame independent of the model. This path is a fixed per-pixel cost.
 *
 * Not thread-safe: the pixel buffer and output bitmaps are cached and reused across calls on the
 * assumption that frames are processed one at a time by a single camera analysis stream (true here
 * since CameraX's STRATEGY_KEEP_ONLY_LATEST + closing the ImageProxy inside InferenceManager gates
 * the next frame until the current one is done).
 */
object ImageUtils {

    private var pixelBuffer: IntArray? = null
    private var sourceBitmap: Bitmap? = null
    private var rotatedBitmap: Bitmap? = null
    private var rotatedCanvas: Canvas? = null
    private val rotationMatrix = Matrix()

    fun imageProxyToBitmap(imageProxy: ImageProxy): Bitmap {
        val width = imageProxy.width
        val height = imageProxy.height

        val pixels = obtainPixelBuffer(width, height)
        yuv420888ToArgbPixels(imageProxy, pixels)

        val source = obtainSourceBitmap(width, height)
        source.setPixels(pixels, 0, width, 0, 0, width, height)

        val degrees = imageProxy.imageInfo.rotationDegrees
        return if (degrees == 0) source else rotate(source, degrees)
    }

    // Mirrors the reference pipeline's crop_xyxy: clamps to bounds, returns null on a degenerate box.
    fun cropBitmap(source: Bitmap, box: RectF): Bitmap? {
        val w = source.width
        val h = source.height
        val left = box.left.toInt().coerceIn(0, w - 1)
        val top = box.top.toInt().coerceIn(0, h - 1)
        val right = box.right.toInt().coerceIn(0, w)
        val bottom = box.bottom.toInt().coerceIn(0, h)
        if (right <= left || bottom <= top) return null
        return Bitmap.createBitmap(source, left, top, right - left, bottom - top)
    }

    private fun obtainPixelBuffer(width: Int, height: Int): IntArray {
        val needed = width * height
        var buffer = pixelBuffer
        if (buffer == null || buffer.size != needed) {
            buffer = IntArray(needed)
            pixelBuffer = buffer
        }
        return buffer
    }

    private fun obtainSourceBitmap(width: Int, height: Int): Bitmap {
        var bmp = sourceBitmap
        if (bmp == null || bmp.width != width || bmp.height != height) {
            bmp = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
            sourceBitmap = bmp
        }
        return bmp
    }

    private fun rotate(source: Bitmap, degrees: Int): Bitmap {
        val swap = degrees == 90 || degrees == 270
        val outWidth = if (swap) source.height else source.width
        val outHeight = if (swap) source.width else source.height

        var out = rotatedBitmap
        if (out == null || out.width != outWidth || out.height != outHeight) {
            out = Bitmap.createBitmap(outWidth, outHeight, Bitmap.Config.ARGB_8888)
            rotatedBitmap = out
            rotatedCanvas = Canvas(out)
        }

        rotationMatrix.reset()
        rotationMatrix.postTranslate(-source.width / 2f, -source.height / 2f)
        rotationMatrix.postRotate(degrees.toFloat())
        rotationMatrix.postTranslate(outWidth / 2f, outHeight / 2f)

        rotatedCanvas!!.drawBitmap(source, rotationMatrix, null)
        return out
    }

    // Fixed-point BT.601 YUV->RGB (coefficients scaled by 1024) -- avoids float math in the
    // per-pixel hot loop. Handles both fully-planar (pixelStride==1) and semi-planar (==2) layouts.
    private fun yuv420888ToArgbPixels(image: ImageProxy, outPixels: IntArray) {
        val width = image.width
        val height = image.height

        val yPlane = image.planes[0]
        val uPlane = image.planes[1]
        val vPlane = image.planes[2]
        val yBuffer = yPlane.buffer
        val uBuffer = uPlane.buffer
        val vBuffer = vPlane.buffer
        val yRowStride = yPlane.rowStride
        val uRowStride = uPlane.rowStride
        val vRowStride = vPlane.rowStride
        val uPixelStride = uPlane.pixelStride
        val vPixelStride = vPlane.pixelStride

        for (row in 0 until height) {
            val yRowStart = row * yRowStride
            val uvRow = row shr 1
            val uRowStart = uvRow * uRowStride
            val vRowStart = uvRow * vRowStride
            var outIndex = row * width

            for (col in 0 until width) {
                val y = yBuffer.get(yRowStart + col).toInt() and 0xFF
                val uvCol = col shr 1
                val u = (uBuffer.get(uRowStart + uvCol * uPixelStride).toInt() and 0xFF) - 128
                val v = (vBuffer.get(vRowStart + uvCol * vPixelStride).toInt() and 0xFF) - 128

                val yShifted = (y - 16).coerceAtLeast(0)
                var r = (1192 * yShifted + 1634 * v) shr 10
                var g = (1192 * yShifted - 833 * v - 400 * u) shr 10
                var b = (1192 * yShifted + 2066 * u) shr 10
                r = r.coerceIn(0, 255)
                g = g.coerceIn(0, 255)
                b = b.coerceIn(0, 255)

                outPixels[outIndex++] = (0xFF shl 24) or (r shl 16) or (g shl 8) or b
            }
        }
    }
}
