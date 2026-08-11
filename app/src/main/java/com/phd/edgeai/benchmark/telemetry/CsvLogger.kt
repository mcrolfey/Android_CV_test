package com.phd.edgeai.benchmark.telemetry

import android.content.ContentValues
import android.content.Context
import android.os.Build
import android.os.Environment
import android.provider.MediaStore
import android.util.Log
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.launch
import java.io.BufferedWriter
import java.io.File
import java.io.FileOutputStream
import java.io.OutputStreamWriter
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/**
 * Thread-safe, non-blocking CSV logger. `log()` is a non-suspending, non-blocking send onto an
 * unbounded Channel; a single dedicated coroutine on Dispatchers.IO drains it and does the actual
 * file I/O, so the inference/analysis loop is never blocked waiting on disk writes.
 *
 * On API 29+, writes go through MediaStore into the public Downloads collection (no storage
 * permission needed) so the CSV can be pulled over USB after the run.
 */
class CsvLogger(private val context: Context) {

    private val writerScope = CoroutineScope(SupervisorJob() + Dispatchers.IO)
    private val queue = Channel<FrameMetrics>(capacity = Channel.UNLIMITED)

    private var writer: BufferedWriter? = null
    private var isRunning = false

    private val header = "Timestamp,Frame_ID,Architecture,Detections_Count,Latency_ms,FPS,Thermal_State,Memory_MB"

    fun start(fileName: String = defaultFileName()) {
        if (isRunning) return
        isRunning = true

        writer = openWriter(fileName)
        writer?.appendLine(header)
        writer?.flush()

        writerScope.launch {
            for (metrics in queue) {
                writeLine(metrics)
            }
        }
    }

    fun log(metrics: FrameMetrics) {
        if (!isRunning) return
        queue.trySend(metrics)
    }

    fun stop() {
        if (!isRunning) return
        isRunning = false
        writer?.flush()
        writer?.close()
        writer = null
    }

    private fun writeLine(metrics: FrameMetrics) {
        val line = buildString {
            append(metrics.timestamp).append(',')
            append(metrics.frameId).append(',')
            append(metrics.architecture).append(',')
            append(metrics.detectionsCount).append(',')
            append("%.2f".format(Locale.US, metrics.latencyMs)).append(',')
            append("%.2f".format(Locale.US, metrics.fps)).append(',')
            append(metrics.thermalState).append(',')
            append("%.2f".format(Locale.US, metrics.memoryMb))
        }
        try {
            writer?.appendLine(line)
            writer?.flush()
        } catch (e: Exception) {
            Log.e("CsvLogger", "Failed to write CSV row", e)
        }
    }

    private fun defaultFileName(): String {
        val timestamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())
        return "benchmark_log_$timestamp.csv"
    }

    private fun openWriter(fileName: String): BufferedWriter {
        return if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            val values = ContentValues().apply {
                put(MediaStore.Downloads.DISPLAY_NAME, fileName)
                put(MediaStore.Downloads.MIME_TYPE, "text/csv")
                put(MediaStore.Downloads.RELATIVE_PATH, Environment.DIRECTORY_DOWNLOADS)
            }
            val uri = context.contentResolver.insert(MediaStore.Downloads.EXTERNAL_CONTENT_URI, values)
                ?: throw IllegalStateException("Failed to create CSV entry in Downloads")
            val outputStream = context.contentResolver.openOutputStream(uri, "wt")
                ?: throw IllegalStateException("Failed to open output stream for CSV")
            BufferedWriter(OutputStreamWriter(outputStream))
        } else {
            @Suppress("DEPRECATION")
            val downloadsDir = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS)
            if (!downloadsDir.exists()) downloadsDir.mkdirs()
            val file = File(downloadsDir, fileName)
            BufferedWriter(OutputStreamWriter(FileOutputStream(file)))
        }
    }
}
