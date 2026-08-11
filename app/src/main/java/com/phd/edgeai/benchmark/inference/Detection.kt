package com.phd.edgeai.benchmark.inference

import android.graphics.RectF

enum class DetectionKind { ROI, FIBER }

data class Detection(
    val boundingBox: RectF,
    val kind: DetectionKind,
    val label: String,
    val confidence: Float
)
