package com.example.activitycollector

import android.content.Context
import android.content.SharedPreferences
import android.media.RingtoneManager
import android.net.Uri

/**
 * AppPreferences
 * ==============
 * Centralized settings repository for FallGuard AI:
 * - 24/7 Background tracking switch (Default: true)
 * - Intelligent Low Battery Mode (Default: true)
 * - Custom Alert Ringtone URI
 * - Alert Countdown duration (Default: 30s)
 * - Self-learning auto-archive CSV toggle (Default: true)
 */
class AppPreferences(context: Context) {

    companion object {
        private const val PREFS_NAME = "fallguard_app_prefs"

        private const val KEY_BACKGROUND_TRACKING = "pref_background_tracking"
        private const val KEY_LOW_BATTERY_MODE    = "pref_low_battery_mode"
        private const val KEY_RINGTONE_URI        = "pref_ringtone_uri"
        private const val KEY_COUNTDOWN_SECONDS   = "pref_countdown_seconds"
        private const val KEY_AUTO_ARCHIVE_DATA   = "pref_auto_archive_data"
        private const val KEY_VIBRATION_PATTERN   = "pref_vibration_pattern"
    }

    private val prefs: SharedPreferences =
        context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)

    var isBackgroundTrackingEnabled: Boolean
        get() = prefs.getBoolean(KEY_BACKGROUND_TRACKING, true) // Default: true
        set(value) = prefs.edit().putBoolean(KEY_BACKGROUND_TRACKING, value).apply()

    var isLowBatteryModeEnabled: Boolean
        get() = prefs.getBoolean(KEY_LOW_BATTERY_MODE, true) // Default: true
        set(value) = prefs.edit().putBoolean(KEY_LOW_BATTERY_MODE, value).apply()

    var alertRingtoneUri: String?
        get() = prefs.getString(KEY_RINGTONE_URI, null)
        set(value) = prefs.edit().putString(KEY_RINGTONE_URI, value).apply()

    var countdownSeconds: Int
        get() = prefs.getInt(KEY_COUNTDOWN_SECONDS, 30) // Default: 30s
        set(value) = prefs.edit().putInt(KEY_COUNTDOWN_SECONDS, value).apply()

    var isAutoArchiveEnabled: Boolean
        get() = prefs.getBoolean(KEY_AUTO_ARCHIVE_DATA, true) // Default: true
        set(value) = prefs.edit().putBoolean(KEY_AUTO_ARCHIVE_DATA, value).apply()

    var vibrationPattern: String
        get() = prefs.getString(KEY_VIBRATION_PATTERN, "intense") ?: "intense"
        set(value) = prefs.edit().putString(KEY_VIBRATION_PATTERN, value).apply()

    fun getDefaultAlarmUri(): Uri {
        return RingtoneManager.getDefaultUri(RingtoneManager.TYPE_ALARM)
            ?: RingtoneManager.getDefaultUri(RingtoneManager.TYPE_NOTIFICATION)
    }

    fun getResolvedRingtoneUri(): Uri {
        val saved = alertRingtoneUri
        return if (!saved.isNullOrEmpty()) {
            try {
                Uri.parse(saved)
            } catch (_: Exception) {
                getDefaultAlarmUri()
            }
        } else {
            getDefaultAlarmUri()
        }
    }
}
