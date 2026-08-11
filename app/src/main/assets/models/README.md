# Model weights

Drop your exported ONNX models here (gitignored — they don't get committed). Names and thresholds
for the Architecture A cascade match the reference batch pipeline exactly (see
`InferenceManager`/`AsbestosClassifier` for where each threshold is set):

- `roi_detector.onnx` — from `cement_roi_model.pt`. YOLO ROI detector, Architecture A only,
  conf 0.25, iou 0.5, top-1 box.
  Export: `yolo export model=cement_roi_model.pt format=onnx imgsz=640 opset=12`
- `pa_detector.onnx` — from `pa_detector.pt`. YOLO "potential asbestos" detector, run on the ROI
  crop (not the full frame), Architecture A only. conf 0.10, iou 0.5, up to 100 boxes.
  Export: `yolo export model=pa_detector.pt format=onnx imgsz=640 opset=12`
- `resnet_binary.onnx` — from `resnet_A_NA_best.pt`. ResNet-18, 2-class (A=0, NA-OF=1), input
  1x3x224x224, ImageNet mean/std, Architecture A only. If P(A) < 0.4 the cascade stops here and
  reports "NA-OF".
- `resnet_subtype.onnx` — from `resnet_A_3class_best.pt`. ResNet-18, 3-class
  (A-AM=0, A-C=1, A-CRO=2), same input/normalization, Architecture A only. Only runs when
  P(A) >= 0.4; falls back to the generic "A" label if its top confidence is below 0.25.
- `yolo_nano_detector.onnx` — Architecture B's model: a separately-trained, single-stage YOLO-nano
  detector (trained on the same dataset as the cascade above), used as the direct on-device
  comparison point against the full 4-stage pipeline. conf 0.25, iou 0.5, uncapped. Assumed to
  output the same final classes as Architecture A (`NA-OF`, `A-AM`, `A-C`, `A-CRO`) — if its class
  order/names differ, update the `labels` list passed to `architectureBDetector` in
  `InferenceManager.kt`.
  Export: `yolo export model=<your_yolo_nano>.pt format=onnx imgsz=640 opset=12`

Export the two ResNets with `torch.onnx.export`, e.g.:

```python
import torch, torch.nn as nn, torchvision.models as models

model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, num_classes)  # 2 for binary, 3 for subtype
model.load_state_dict(torch.load("resnet_A_NA_best.pt", map_location="cpu"))
model.eval()

torch.onnx.export(
    model, torch.randn(1, 3, 224, 224), "resnet_binary.onnx",
    input_names=["input"], output_names=["logits"], opset_version=12
)
```

If your exports use different input sizes, class order, or normalization, update the constructor
defaults in `YoloDetector.kt` / `AsbestosClassifier.kt` accordingly.
