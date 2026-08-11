package com.phd.edgeai.benchmark.inference

import android.graphics.RectF

data class Detection(
    val boundingBox: RectF,
    val roiConfidence: Float,
    val roiClassId: Int,
    val roiLabel: String,
    val classificationLabel: String? = null,
    val classificationConfidence: Float? = null
)
