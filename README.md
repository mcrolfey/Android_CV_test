# Android_CV_test

Headless, benchmark-focused Android app for PhD field-testing of on-device asbestos detection.
Measures real-world mobile inference constraints — latency, FPS, thermal throttling, memory — for
two vision pipelines, and logs every frame to a CSV for offline analysis.

## Architectures

The app exists to compare these two independently-trained pipelines' on-device behavior directly.

- **Architecture B (single-stage baseline):** camera frame -> a separately-trained YOLO-nano
  object detector (same dataset, no cascade) -> all boxes above threshold, uncapped, conf 0.25 /
  iou 0.5. The fast single-model detection latency/FPS/thermal baseline.
- **Architecture A (4-stage cascade):** reproduces the reference batch-video pipeline exactly:
  1. ROI YOLO on the full frame, conf 0.25 / iou 0.5, **top-1 box only**.
  2. Crop that ROI, run PA YOLO on the crop, conf 0.10 / iou 0.5, up to 100 boxes.
  3. Crop each PA box, run the binary ResNet (A vs NA-OF). If P(A) < 0.4, label "NA-OF" and stop.
  4. Otherwise run the subtype ResNet (A-AM / A-C / A-CRO). If its top confidence is below 0.25,
     fall back to the generic "A" label; otherwise report the subtype.

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
│       ├── assets/models/           # drop roi_detector.onnx / pa_detector.onnx / resnet_*.onnx here
│       ├── res/values/              # strings.xml, themes.xml
│       └── java/com/phd/edgeai/benchmark/
│           ├── MainActivity.kt      # camera permission + Compose host
│           ├── inference/
│           │   ├── Architecture.kt        # A/B enum
│           │   ├── Detection.kt           # box + kind (ROI/FIBER) + label + confidence
│           │   ├── ImageUtils.kt          # ImageProxy -> Bitmap (YUV_420_888 -> NV21), crop
│           │   ├── YoloDetector.kt        # ONNX Runtime YOLO wrapper, letterbox + NMS, reused
│           │   │                          #   for the ROI/PA stages and Architecture B's model
│           │   ├── AsbestosClassifier.kt  # binary + subtype ResNet cascade with thresholds
│           │   └── InferenceManager.kt    # A/B routing + cascade wiring, System.nanoTime() timing
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

The app expects five ONNX models in `app/src/main/assets/models/` (gitignored — see the README
there for export commands, thresholds, and expected input shapes):

- `roi_detector.onnx` — from `cement_roi_model.pt`, Architecture A's ROI stage
- `pa_detector.onnx` — from `pa_detector.pt`, Architecture A's PA stage
- `resnet_binary.onnx` — from `resnet_A_NA_best.pt` (2-class), Architecture A's binary stage
- `resnet_subtype.onnx` — from `resnet_A_3class_best.pt` (3-class), Architecture A's subtype stage
- `yolo_nano_detector.onnx` — Architecture B's standalone, separately-trained detector

`YoloDetector` assumes Ultralytics' standard detect-head export
(`[1, 4+numClasses, numAnchors]`, no objectness channel). `AsbestosClassifier` assumes plain
ImageNet-normalized ResNet-18 exports. If your exports differ (class order, input size,
normalization), update the constructor defaults in those two files.

Swapping to PyTorch Mobile instead of ONNX Runtime means replacing the `ai.onnxruntime.*` calls in
`YoloDetector`/`AsbestosClassifier` with `org.pytorch.Module`/`Tensor` equivalents and the
`onnxruntime-android` dependency in `app/build.gradle.kts` with `org.pytorch:pytorch_android_lite`
— the rest of the pipeline (`InferenceManager`, `CameraController`, telemetry, UI) is unaffected.

## CSV output

Press "Start Stress Test" to begin logging; each processed frame appends a row to
`benchmark_log_<timestamp>.csv` in the device's Downloads folder
(`Timestamp,Frame_ID,Architecture,Detections_Count,Latency_ms,FPS,Thermal_State,Memory_MB`).
Writes go through `CsvLogger`'s dedicated IO coroutine so the inference loop is never blocked on
disk I/O. Pull the file over USB from `Downloads/` once the run finishes.
