package com.phd.edgeai.benchmark.inference

enum class Architecture(val label: String) {
    ARCHITECTURE_B_SINGLESTAGE("Architecture B (YOLO only)"),
    ARCHITECTURE_A_MULTISTAGE("Architecture A (YOLO + ResNet)")
}
