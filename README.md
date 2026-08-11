# Android_CV_test

Headless, benchmark-focused Android app for PhD field-testing of on-device asbestos detection.
Measures real-world mobile inference constraints — latency, FPS, thermal throttling, memory — for
two vision pipelines, and logs every frame to a CSV for offline analysis.

## Architectures

- **Architecture B (single-stage):** camera frame -> YOLOv11n -> bounding boxes + confidence.
- **Architecture A (multi-stage ROI):** camera frame -> YOLOv11n ROI detector -> crop each box ->
  ResNet classifier per crop -> combined boxes + classification labels.

Toggle between them live from the on-screen control panel; the CSV log records which architecture
produced each row.

## Directory structure

```
Android_CV_test/
├── app/
│   ├── build.gradle.kts
│   ├── proguard-rules.pro
│   └── src/main/
│       ├── AndroidManifest.xml
│       ├── assets/models/           # drop roi_detector.onnx / resnet_classifier.onnx here
│       ├── res/values/              # strings.xml, themes.xml
│       └── java/com/phd/edgeai/benchmark/
│           ├── MainActivity.kt      # camera permission + Compose host
│           ├── inference/
│           │   ├── Architecture.kt      # A/B enum
│           │   ├── Detection.kt         # box + ROI conf/label + optional classification
│           │   ├── ImageUtils.kt        # ImageProxy -> Bitmap (YUV_420_888 -> NV21), crop
│           │   ├── YoloDetector.kt      # ONNX Runtime YOLOv11 wrapper, letterbox + NMS
│           │   ├── ResNetClassifier.kt  # ONNX Runtime ResNet wrapper for ROI crops
│           │   └── InferenceManager.kt  # A/B routing, System.nanoTime() timing
│           ├── telemetry/
│           │   ├── FrameMetrics.kt      # one CSV row
│           │   ├── ThermalMonitor.kt    # PowerManager.getCurrentThermalStatus()
│           │   ├── MemoryMonitor.kt     # heap usage via Runtime
│           │   └── CsvLogger.kt         # Channel + dedicated IO coroutine, non-blocking
│           ├── camera/
│           │   └── CameraController.kt  # CameraX Preview + ImageAnalysis binding
│           └── ui/
│               ├── BenchmarkViewModel.kt  # StateFlow<BenchmarkUiState>
│               ├── CameraScreen.kt        # PreviewView + overlays + control panel
│               ├── BoundingBoxOverlay.kt  # Canvas draw of boxes/labels
│               ├── HudOverlay.kt          # FPS / latency / thermal / memory text
│               └── ControlPanel.kt        # architecture toggle + stress test button
├── build.gradle.kts / settings.gradle.kts / gradle.properties
└── gradle/wrapper/gradle-wrapper.properties
```

## Building

Open the project root in Android Studio (Koala or newer) — it will offer to generate the Gradle
wrapper jar/scripts on first sync using the pinned Gradle 8.7 in
`gradle/wrapper/gradle-wrapper.properties`. Requires JDK 17.

If you'd rather use the command line, run `gradle wrapper` once with a local Gradle 8.7+ install
to generate `gradlew`/`gradlew.bat`, then `./gradlew assembleDebug`.

## Adding your model weights

The app expects two ONNX models in `app/src/main/assets/models/` (gitignored — see the README
there for export commands and expected input shapes):

- `roi_detector.onnx` — YOLOv11n, used by both architectures
- `resnet_classifier.onnx` — ResNet classifier, used only by Architecture A

`YoloDetector` and `ResNetClassifier` assume Ultralytics' standard detect-head export
(`[1, 4+numClasses, numAnchors]`, no objectness channel) and a plain ImageNet-normalized
classifier respectively. If your exports differ (class list, input size, normalization), update
the constructor defaults in those two files.

Swapping to PyTorch Mobile instead of ONNX Runtime means replacing the `ai.onnxruntime.*` calls in
`YoloDetector`/`ResNetClassifier` with `org.pytorch.Module`/`Tensor` equivalents and the
`onnxruntime-android` dependency in `app/build.gradle.kts` with `org.pytorch:pytorch_android_lite`
— the rest of the pipeline (`InferenceManager`, `CameraController`, telemetry, UI) is unaffected.

## CSV output

Press "Start Stress Test" to begin logging; each processed frame appends a row to
`benchmark_log_<timestamp>.csv` in the device's Downloads folder
(`Timestamp,Frame_ID,Architecture,Detections_Count,Latency_ms,FPS,Thermal_State,Memory_MB`).
Writes go through `CsvLogger`'s dedicated IO coroutine so the inference loop is never blocked on
disk I/O. Pull the file over USB from `Downloads/` once the run finishes.
