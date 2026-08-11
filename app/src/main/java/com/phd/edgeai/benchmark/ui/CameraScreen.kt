package com.phd.edgeai.benchmark.ui

import android.view.ViewGroup
import androidx.camera.view.PreviewView
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.remember
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalLifecycleOwner
import androidx.compose.ui.viewinterop.AndroidView
import androidx.lifecycle.viewmodel.compose.viewModel
import com.phd.edgeai.benchmark.camera.CameraController
import com.phd.edgeai.benchmark.inference.InferenceManager
import com.phd.edgeai.benchmark.telemetry.CsvLogger

@Composable
fun BenchmarkScreen(viewModel: BenchmarkViewModel = viewModel()) {
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current
    val uiState by viewModel.uiState.collectAsState()

    val inferenceManager = remember { InferenceManager(context) }
    val csvLogger = remember { CsvLogger(context) }

    val cameraController = remember {
        CameraController(
            context = context,
            lifecycleOwner = lifecycleOwner,
            inferenceManager = inferenceManager,
            csvLogger = csvLogger
        ) { result, metrics ->
            viewModel.updateFrame(
                detections = result.detections,
                metrics = metrics,
                frameWidth = result.frameWidth,
                frameHeight = result.frameHeight
            )
        }
    }

    cameraController.currentArchitecture = uiState.architecture
    cameraController.isLogging = uiState.isStressTesting

    Box(modifier = Modifier.fillMaxSize()) {
        AndroidView(
            modifier = Modifier.fillMaxSize(),
            factory = { ctx ->
                PreviewView(ctx).apply {
                    layoutParams = ViewGroup.LayoutParams(
                        ViewGroup.LayoutParams.MATCH_PARENT,
                        ViewGroup.LayoutParams.MATCH_PARENT
                    )
                    scaleType = PreviewView.ScaleType.FIT_CENTER
                    cameraController.start(this)
                }
            }
        )

        BoundingBoxOverlay(
            detections = uiState.detections,
            frameWidth = uiState.frameWidth,
            frameHeight = uiState.frameHeight,
            modifier = Modifier.fillMaxSize()
        )

        HudOverlay(uiState = uiState, modifier = Modifier.fillMaxSize())

        ControlPanel(
            uiState = uiState,
            onArchitectureSelected = viewModel::setArchitecture,
            onStartStressTest = { frameCount ->
                csvLogger.start()
                viewModel.startStressTest(frameCount)
            },
            onStopStressTest = {
                csvLogger.stop()
                viewModel.stopStressTest()
            },
            modifier = Modifier.fillMaxSize()
        )
    }
}
