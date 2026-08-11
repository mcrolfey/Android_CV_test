package com.phd.edgeai.benchmark.ui

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.unit.dp

@Composable
fun HudOverlay(uiState: BenchmarkUiState, modifier: Modifier = Modifier) {
    Column(modifier = modifier.padding(16.dp), verticalArrangement = Arrangement.Top) {
        Column(
            modifier = Modifier
                .background(Color.Black.copy(alpha = 0.55f))
                .padding(8.dp)
        ) {
            Text("Architecture: ${uiState.architecture.label}", color = Color.White)
            Text("FPS: %.1f (inst. %.1f)".format(uiState.displayFps, uiState.fps), color = Color.White)
            Text("Latency: %.1f ms".format(uiState.latencyMs), color = Color.White)
            Text("Thermal: ${uiState.thermalState}", color = Color.White)
            Text("Memory: %.1f MB".format(uiState.memoryMb), color = Color.White)
            if (uiState.isStressTesting) {
                Text("Stress test frames remaining: ${uiState.stressTestFramesRemaining}", color = Color.Yellow)
            }
        }
    }
}
