package com.example.activitycollector

/**
 * One fused sensor sample. Both sensors are read independently; we merge them
 * by pairing each accelerometer event with the latest gyroscope values seen,
 * which is standard practice for time-series ML datasets.
 */
data class SensorDataRecord(
    val timestamp: Long,   // Unix time in milliseconds
    val ax: Float,
    val ay: Float,
    val az: Float,
    val gx: Float,
    val gy: Float,
    val gz: Float,
    val label: String
) {
    fun toCsvRow(): String =
        "$timestamp,$ax,$ay,$az,$gx,$gy,$gz,$label"
}
