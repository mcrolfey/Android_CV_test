package com.phd.edgeai.benchmark.camera

import android.content.Context
import android.util.Size
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.core.resolutionselector.ResolutionStrategy
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import androidx.lifecycle.LifecycleOwner
import com.phd.edgeai.benchmark.inference.Architecture
import com.phd.edgeai.benchmark.inference.InferenceManager
import com.phd.edgeai.benchmark.inference.InferenceResult
import com.phd.edgeai.benchmark.telemetry.CsvLogger
import com.phd.edgeai.benchmark.telemetry.FrameMetrics
import com.phd.edgeai.benchmark.telemetry.MemoryMonitor
import com.phd.edgeai.benchmark.telemetry.ThermalMonitor
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.launch
import java.util.concurrent.atomic.AtomicLong

/**
 * Binds CameraX Preview + ImageAnalysis, routes each frame to InferenceManager on
 * Dispatchers.Default, and reports per-frame results/metrics back to the UI layer.
 * STRATEGY_KEEP_ONLY_LATEST drops backed-up frames instead of queueing, so FPS/latency numbers
 * reflect true inference throughput rather than an analyzer backlog.
 *
 * The analysis stream is capped at ANALYSIS_TARGET_RESOLUTION -- left unconstrained, CameraX
 * defaults to a much larger frame (often close to full sensor resolution), which meant every
 * frame paid for YUV conversion, letterboxing, and NMS over far more pixels than any of the
 * models actually need, and made per-frame latency more sensitive to whatever resolution the
 * device happened to pick.
 */
class CameraController(
    private val context: Context,
    private val lifecycleOwner: LifecycleOwner,
    private val inferenceManager: InferenceManager,
    private val csvLogger: CsvLogger,
    private val onFrameResult: (InferenceResult, FrameMetrics) -> Unit
) {
    private val thermalMonitor = ThermalMonitor(context)
    private val analysisScope = CoroutineScope(SupervisorJob() + Dispatchers.Default)
    private val frameCounter = AtomicLong(0)
    private var lastFrameTimestampNs = 0L

    @Volatile
    var currentArchitecture: Architecture = Architecture.ARCHITECTURE_B_SINGLESTAGE

    @Volatile
    var isLogging: Boolean = false

    private var cameraProvider: ProcessCameraProvider? = null

    companion object {
        private val ANALYSIS_TARGET_RESOLUTION = Size(1280, 720)
    }

    fun start(previewView: PreviewView) {
        val providerFuture = ProcessCameraProvider.getInstance(context)
        providerFuture.addListener({
            val provider = providerFuture.get()
            cameraProvider = provider
            bindUseCases(provider, previewView)
        }, ContextCompat.getMainExecutor(context))
    }

    fun stop() {
        cameraProvider?.unbindAll()
    }

    private fun bindUseCases(provider: ProcessCameraProvider, previewView: PreviewView) {
        provider.unbindAll()

        val preview = Preview.Builder().build().also {
            it.setSurfaceProvider(previewView.surfaceProvider)
        }

        val resolutionSelector = ResolutionSelector.Builder()
            .setResolutionStrategy(
                ResolutionStrategy(ANALYSIS_TARGET_RESOLUTION, ResolutionStrategy.FALLBACK_RULE_CLOSEST_HIGHER_THEN_LOWER)
            )
            .build()

        val imageAnalysis = ImageAnalysis.Builder()
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .setResolutionSelector(resolutionSelector)
            .build()

        imageAnalysis.setAnalyzer(ContextCompat.getMainExecutor(context)) { imageProxy ->
            analysisScope.launch { processFrame(imageProxy) }
        }

        provider.bindToLifecycle(
            lifecycleOwner,
            CameraSelector.DEFAULT_BACK_CAMERA,
            preview,
            imageAnalysis
        )
    }

    private fun processFrame(imageProxy: ImageProxy) {
        val architecture = currentArchitecture

        // InferenceManager closes imageProxy once the frame has been copied to a Bitmap.
        val result = inferenceManager.runInference(imageProxy, architecture)

        val nowNs = System.nanoTime()
        val deltaMs = if (lastFrameTimestampNs == 0L) result.latencyMs
        else (nowNs - lastFrameTimestampNs) / 1_000_000.0
        lastFrameTimestampNs = nowNs
        val fps = if (deltaMs > 0) 1000.0 / deltaMs else 0.0

        val metrics = FrameMetrics(
            timestamp = System.currentTimeMillis(),
            frameId = frameCounter.incrementAndGet(),
            architecture = architecture.name,
            detectionsCount = result.detections.size,
            latencyMs = result.latencyMs,
            fps = fps,
            thermalState = thermalMonitor.currentThermalState(),
            memoryMb = MemoryMonitor.usedMemoryMb()
        )

        if (isLogging) {
            csvLogger.log(metrics)
        }

        onFrameResult(result, metrics)
    }
}
