package com.phd.edgeai.benchmark.inference

enum class Architecture(val label: String) {
    ARCHITECTURE_B_SINGLESTAGE("Architecture B (YOLO-nano, single-stage)"),
    ARCHITECTURE_A_MULTISTAGE("Architecture A (ROI → PA → ResNet cascade)")
}
