package com.phd.edgeai.benchmark.telemetry

data class FrameMetrics(
    val timestamp: Long,
    val frameId: Long,
    val architecture: String,
    val detectionsCount: Int,
    val latencyMs: Double,
    val fps: Double,
    val thermalState: String,
    val memoryMb: Double
)
