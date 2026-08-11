package com.phd.edgeai.benchmark.ui

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.wrapContentSize
import androidx.compose.material3.Button
import androidx.compose.material3.RadioButton
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.unit.dp
import com.phd.edgeai.benchmark.inference.Architecture

@Composable
fun ControlPanel(
    uiState: BenchmarkUiState,
    onArchitectureSelected: (Architecture) -> Unit,
    onStartStressTest: (Int) -> Unit,
    onStopStressTest: () -> Unit,
    modifier: Modifier = Modifier,
    stressTestFrameCount: Int = 300
) {
    Column(modifier = modifier.wrapContentSize(Alignment.BottomCenter).padding(16.dp)) {
        Column(
            modifier = Modifier
                .background(Color.Black.copy(alpha = 0.55f))
                .padding(12.dp)
        ) {
            Architecture.entries.forEach { architecture ->
                Row {
                    RadioButton(
                        selected = uiState.architecture == architecture,
                        onClick = { onArchitectureSelected(architecture) }
                    )
                    Text(
                        architecture.label,
                        color = Color.White,
                        modifier = Modifier.padding(top = 12.dp)
                    )
                }
            }

            Button(
                onClick = {
                    if (uiState.isStressTesting) onStopStressTest() else onStartStressTest(stressTestFrameCount)
                }
            ) {
                Text(if (uiState.isStressTesting) "Stop Stress Test" else "Start Stress Test ($stressTestFrameCount frames)")
            }
        }
    }
}
