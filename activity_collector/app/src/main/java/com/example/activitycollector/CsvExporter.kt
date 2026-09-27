package com.example.activitycollector

import android.content.ContentValues
import android.content.Context
import android.os.Build
import android.os.Environment
import android.provider.MediaStore
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

object CsvExporter {

    private const val CSV_HEADER = "timestamp,ax,ay,az,gx,gy,gz,label"

    /**
     * Writes [records] to a CSV file and returns the absolute path (or URI string)
     * of the saved file, or null on failure.
     *
     * On API 29+ the file lands in the public Downloads folder via MediaStore.
     * On API 26-28 it writes directly to
     * Environment.getExternalStoragePublicDirectory(DIRECTORY_DOWNLOADS).
     */
    fun export(context: Context, records: List<SensorDataRecord>): String? {
        if (records.isEmpty()) return null

        val filename = buildFilename()
        val csvContent = buildString {
            appendLine(CSV_HEADER)
            records.forEach { appendLine(it.toCsvRow()) }
        }

        return if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            exportViaMediaStore(context, filename, csvContent)
        } else {
            exportLegacy(filename, csvContent)
        }
    }

    // ── API 29+ ──────────────────────────────────────────────────────────────

    private fun exportViaMediaStore(
        context: Context,
        filename: String,
        content: String
    ): String? {
        val values = ContentValues().apply {
            put(MediaStore.Downloads.DISPLAY_NAME, filename)
            put(MediaStore.Downloads.MIME_TYPE, "text/csv")
            put(MediaStore.Downloads.RELATIVE_PATH, Environment.DIRECTORY_DOWNLOADS)
        }
        val resolver = context.contentResolver
        val uri = resolver.insert(MediaStore.Downloads.EXTERNAL_CONTENT_URI, values)
            ?: return null

        return try {
            resolver.openOutputStream(uri)?.use { it.write(content.toByteArray()) }
            uri.toString()
        } catch (e: IOException) {
            resolver.delete(uri, null, null)
            null
        }
    }

    // ── API 26-28 ─────────────────────────────────────────────────────────────

    @Suppress("DEPRECATION")
    private fun exportLegacy(filename: String, content: String): String? {
        val dir = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS)
        dir.mkdirs()
        val file = File(dir, filename)
        return try {
            FileOutputStream(file).use { it.write(content.toByteArray()) }
            file.absolutePath
        } catch (e: IOException) {
            null
        }
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

    private fun buildFilename(): String {
        val timestamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())
        return "activity_$timestamp.csv"
    }
}
