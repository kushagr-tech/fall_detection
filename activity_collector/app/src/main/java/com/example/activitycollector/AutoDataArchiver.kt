package com.example.activitycollector

import android.content.Context
import android.util.Log
import java.io.File
import java.io.FileWriter
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/**
 * AutoDataArchiver
 * ================
 * Automatically archives raw motion sensor windows and fall telemetry in the background.
 * Enables continuous self-learning dataset expansion without user intervention.
 */
object AutoDataArchiver {

    private const val TAG = "AutoDataArchiver"

    fun archiveFallWindow(
        context: Context,
        label: String,
        window2D: Array<FloatArray>,
        impactPeak: Float,
        isConfirmed: Boolean
    ): File? {
        return try {
            val archiveDir = File(context.getExternalFilesDir(null) ?: context.filesDir, "fall_archives")
            if (!archiveDir.exists()) archiveDir.mkdirs()

            val timestamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())
            val statusTag = if (isConfirmed) "FALL_CONFIRMED" else "EVENT"
            val file = File(archiveDir, "${statusTag}_${label}_${timestamp}.csv")

            FileWriter(file).use { writer ->
                writer.append("timestamp_ms,ax,ay,az,gx,gy,gz,label,impact_peak\n")
                val now = System.currentTimeMillis()
                window2D.forEachIndexed { idx, row ->
                    val t = now - (window2D.size - idx) * 20L
                    writer.append("$t,${row[0]},${row[1]},${row[2]},${row[3]},${row[4]},${row[5]},$label,$impactPeak\n")
                }
            }
            Log.i(TAG, "Auto-archived telemetry window: ${file.name} (${file.length()} bytes)")
            file
        } catch (e: Exception) {
            Log.e(TAG, "Failed to auto-archive telemetry: ${e.message}")
            null
        }
    }

    fun getArchivedFiles(context: Context): List<File> {
        val archiveDir = File(context.getExternalFilesDir(null) ?: context.filesDir, "fall_archives")
        if (!archiveDir.exists()) return emptyList()
        return archiveDir.listFiles { f -> f.extension == "csv" }?.toList()?.sortedByDescending { it.lastModified() } ?: emptyList()
    }
}
