package com.phd.edgeai.benchmark.ui

import androidx.lifecycle.ViewModel
import com.phd.edgeai.benchmark.inference.Architecture
import com.phd.edgeai.benchmark.inference.Detection
import com.phd.edgeai.benchmark.telemetry.FrameMetrics
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow

data class BenchmarkUiState(
    val architecture: Architecture = Architecture.ARCHITECTURE_B_SINGLESTAGE,
    val detections: List<Detection> = emptyList(),
    val fps: Double = 0.0,
    val latencyMs: Double = 0.0,
    val thermalState: String = "UNKNOWN",
    val memoryMb: Double = 0.0,
    val isStressTesting: Boolean = false,
    val stressTestFramesRemaining: Int = 0,
    val frameWidth: Int = 0,
    val frameHeight: Int = 0
)

class BenchmarkViewModel : ViewModel() {

    private val _uiState = MutableStateFlow(BenchmarkUiState())
    val uiState: StateFlow<BenchmarkUiState> = _uiState.asStateFlow()

    fun setArchitecture(architecture: Architecture) {
        _uiState.value = _uiState.value.copy(architecture = architecture)
    }

    fun updateFrame(detections: List<Detection>, metrics: FrameMetrics, frameWidth: Int, frameHeight: Int) {
        val current = _uiState.value
        val remaining = if (current.isStressTesting) (current.stressTestFramesRemaining - 1).coerceAtLeast(0) else 0
        _uiState.value = current.copy(
            detections = detections,
            fps = metrics.fps,
            latencyMs = metrics.latencyMs,
            thermalState = metrics.thermalState,
            memoryMb = metrics.memoryMb,
            frameWidth = frameWidth,
            frameHeight = frameHeight,
            stressTestFramesRemaining = remaining,
            isStressTesting = current.isStressTesting && remaining > 0
        )
    }

    fun startStressTest(frameCount: Int) {
        _uiState.value = _uiState.value.copy(isStressTesting = true, stressTestFramesRemaining = frameCount)
    }

    fun stopStressTest() {
        _uiState.value = _uiState.value.copy(isStressTesting = false, stressTestFramesRemaining = 0)
    }
}
