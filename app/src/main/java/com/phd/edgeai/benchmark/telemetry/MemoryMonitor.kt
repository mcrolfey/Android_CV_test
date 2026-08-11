package com.phd.edgeai.benchmark.telemetry

object MemoryMonitor {
    fun usedMemoryMb(): Double {
        val runtime = Runtime.getRuntime()
        val usedBytes = runtime.totalMemory() - runtime.freeMemory()
        return usedBytes / (1024.0 * 1024.0)
    }
}
