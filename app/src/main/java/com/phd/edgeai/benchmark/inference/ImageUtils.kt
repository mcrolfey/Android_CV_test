package com.phd.edgeai.benchmark.inference

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.ImageFormat
import android.graphics.Matrix
import android.graphics.Rect
import android.graphics.RectF
import android.graphics.YuvImage
import androidx.camera.core.ImageProxy
import java.io.ByteArrayOutputStream

object ImageUtils {

    fun imageProxyToBitmap(imageProxy: ImageProxy): Bitmap {
        val nv21 = yuv420888ToNv21(imageProxy)
        val yuvImage = YuvImage(nv21, ImageFormat.NV21, imageProxy.width, imageProxy.height, null)
        val out = ByteArrayOutputStream()
        yuvImage.compressToJpeg(Rect(0, 0, imageProxy.width, imageProxy.height), 90, out)
        val jpegBytes = out.toByteArray()
        val bitmap = BitmapFactory.decodeByteArray(jpegBytes, 0, jpegBytes.size)
        return rotateBitmap(bitmap, imageProxy.imageInfo.rotationDegrees)
    }

    fun cropBitmap(source: Bitmap, box: RectF): Bitmap {
        val left = box.left.toInt().coerceIn(0, source.width - 1)
        val top = box.top.toInt().coerceIn(0, source.height - 1)
        val right = box.right.toInt().coerceIn(left + 1, source.width)
        val bottom = box.bottom.toInt().coerceIn(top + 1, source.height)
        return Bitmap.createBitmap(source, left, top, right - left, bottom - top)
    }

    private fun rotateBitmap(bitmap: Bitmap, degrees: Int): Bitmap {
        if (degrees == 0) return bitmap
        val matrix = Matrix().apply { postRotate(degrees.toFloat()) }
        return Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
    }

    // Standard semi-planar (pixelStride==2) vs fully-planar (pixelStride==1) YUV_420_888 handling.
    private fun yuv420888ToNv21(image: ImageProxy): ByteArray {
        val yBuffer = image.planes[0].buffer
        val uBuffer = image.planes[1].buffer
        val vBuffer = image.planes[2].buffer

        val ySize = yBuffer.remaining()
        val uSize = uBuffer.remaining()
        val vSize = vBuffer.remaining()
        val nv21 = ByteArray(ySize + uSize + vSize)

        yBuffer.get(nv21, 0, ySize)

        val vPixelStride = image.planes[2].pixelStride
        if (vPixelStride == 1) {
            vBuffer.get(nv21, ySize, vSize)
            uBuffer.get(nv21, ySize + vSize, uSize)
        } else {
            val vArray = ByteArray(vSize)
            val uArray = ByteArray(uSize)
            vBuffer.get(vArray)
            uBuffer.get(uArray)

            var offset = ySize
            var i = 0
            while (i < vArray.size && i < uArray.size) {
                nv21[offset++] = vArray[i]
                nv21[offset++] = uArray[i]
                i += vPixelStride
            }
        }
        return nv21
    }
}
