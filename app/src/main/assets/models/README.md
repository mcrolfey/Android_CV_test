# Model weights

Drop your exported ONNX models here (gitignored — they don't get committed, along with the source
`.pt` checkpoints). Names, input sizes, and thresholds below match what's actually baked into the
checkpoints and the reference batch pipeline (see `InferenceManager`/`AsbestosClassifier` for where
each threshold is set).

- `roi_detector.onnx` — YOLO11n, single class (`CS`), **imgsz 640**. Architecture A only,
  conf 0.25 / iou 0.5, top-1 box.
- `pa_detector.onnx` — YOLO11m, single class (`pa`), **imgsz 1024** (larger than the other two
  detectors — check this if you retrain it). Run on the ROI crop, not the full frame. Architecture
  A only, conf 0.10 / iou 0.5, up to 100 boxes.
- `resnet_binary.onnx` — ResNet-18, 2-class (A=0, NA-OF=1), input 1x3x224x224, ImageNet mean/std.
  Architecture A only. If P(A) < 0.4 the cascade stops here and reports "NA-OF".
- `resnet_subtype.onnx` — ResNet-18, 3-class (A-AM=0, A-C=1, A-CRO=2), same input/normalization.
  Architecture A only. Only runs when P(A) >= 0.4; falls back to the generic "A" label if its top
  confidence is below 0.25.
- `yolo_nano_detector.onnx` — YOLO11n, **7 classes** (`A-AM, A-CF, A-COF, A-CP, A-CRO, NA-CS,
  NA-OF`, in that index order), imgsz 640. Architecture B's model: a separately-trained,
  single-stage detector (same dataset, no cascade), used as the direct on-device comparison point
  against the full 4-stage pipeline. conf 0.25 / iou 0.5, uncapped.

`pa_detector_nano.pt` (if present) isn't wired into the app — it's a smaller/faster PA-stage
candidate that isn't currently used anywhere in `InferenceManager`.

## Export commands used

YOLO checkpoints (`ultralytics` 8.4.x, needs the `onnx` + `onnxslim` packages; export will also
auto-install `onnxruntime-gpu` if missing):

```python
from ultralytics import YOLO
YOLO("roi_detector.pt").export(format="onnx", imgsz=640, opset=12, simplify=True, dynamic=False)
YOLO("pa_detector.pt").export(format="onnx", imgsz=1024, opset=12, simplify=True, dynamic=False)
YOLO("yolo_nano_detector.pt").export(format="onnx", imgsz=640, opset=12, simplify=True, dynamic=False)
```

ResNet-18 state dicts (`torch` 2.11+ needs `dynamo=False` to get the legacy exporter and an actual
opset-12 graph — the new dynamo exporter silently upgrades to opset 18 and needs `onnxscript`):

```python
import torch, torch.nn as nn, torchvision.models as models

model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, num_classes)  # 2 for binary, 3 for subtype
model.load_state_dict(torch.load("resnet_binary.pt", map_location="cpu"))
model.eval()

torch.onnx.export(
    model, torch.randn(1, 3, 224, 224), "resnet_binary.onnx",
    input_names=["input"], output_names=["logits"], opset_version=12, dynamo=False,
)
```

If you retrain any of these with a different input size, class list, or normalization, update the
matching constructor defaults in `YoloDetector.kt` / `AsbestosClassifier.kt` / the detector
instantiations in `InferenceManager.kt`.
