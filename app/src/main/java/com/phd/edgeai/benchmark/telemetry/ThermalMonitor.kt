package com.phd.edgeai.benchmark.telemetry

import android.content.Context
import android.os.Build
import android.os.PowerManager

class ThermalMonitor(context: Context) {

    private val powerManager = context.getSystemService(Context.POWER_SERVICE) as PowerManager

    fun currentThermalState(): String {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.Q) return "UNSUPPORTED"
        return mapStatus(powerManager.currentThermalStatus)
    }

    fun registerListener(onChanged: (String) -> Unit): PowerManager.OnThermalStatusChangedListener? {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.Q) return null
        val listener = PowerManager.OnThermalStatusChangedListener { status -> onChanged(mapStatus(status)) }
        powerManager.addThermalStatusListener(listener)
        return listener
    }

    fun unregisterListener(listener: PowerManager.OnThermalStatusChangedListener?) {
        if (listener != null && Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            powerManager.removeThermalStatusListener(listener)
        }
    }

    private fun mapStatus(status: Int): String = when (status) {
        PowerManager.THERMAL_STATUS_NONE -> "NOMINAL"
        PowerManager.THERMAL_STATUS_LIGHT -> "LIGHT"
        PowerManager.THERMAL_STATUS_MODERATE -> "MODERATE"
        PowerManager.THERMAL_STATUS_SEVERE -> "SEVERE"
        PowerManager.THERMAL_STATUS_CRITICAL -> "CRITICAL"
        PowerManager.THERMAL_STATUS_EMERGENCY -> "EMERGENCY"
        PowerManager.THERMAL_STATUS_SHUTDOWN -> "SHUTDOWN"
        else -> "UNKNOWN"
    }
}
