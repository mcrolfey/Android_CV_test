# Model weights

Drop your exported ONNX models here (gitignored — they don't get committed):

- `roi_detector.onnx` — YOLOv11n exported from `best.pt`, used as the ROI/single-stage detector
  by both Architecture A and Architecture B. Ultralytics export:
  `yolo export model=best.pt format=onnx imgsz=640 opset=12`
- `resnet_classifier.onnx` — ResNet classifier used only by Architecture A, applied to each
  cropped ROI. Export with `torch.onnx.export`, input `1x3x224x224`, normalized with ImageNet
  mean/std.

`YoloDetector` and `ResNetClassifier` (in `inference/`) read these paths by default and assume
these shapes/preprocessing. Update the constructor defaults there if your export differs
(input size, class labels, normalization).
