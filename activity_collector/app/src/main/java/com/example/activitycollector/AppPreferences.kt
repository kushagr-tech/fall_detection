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
        private const val KEY_THEME_MODE          = "pref_theme_mode"
        private const val KEY_SENSITIVITY_PRESET  = "pref_sensitivity_preset"
        private const val KEY_CUSTOM_IMPACT       = "pref_custom_impact"
        private const val KEY_CUSTOM_FREEFALL     = "pref_custom_freefall"
        private const val KEY_CUSTOM_STILLNESS    = "pref_custom_stillness"
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

    var themeMode: String
        get() = prefs.getString(KEY_THEME_MODE, "system") ?: "system"
        set(value) = prefs.edit().putString(KEY_THEME_MODE, value).apply()

    var sensitivityPreset: String
        get() = prefs.getString(KEY_SENSITIVITY_PRESET, "desk_safe") ?: "desk_safe"
        set(value) = prefs.edit().putString(KEY_SENSITIVITY_PRESET, value).apply()

    var customImpactThreshold: Float
        get() = prefs.getFloat(KEY_CUSTOM_IMPACT, 24.0f)
        set(value) = prefs.edit().putFloat(KEY_CUSTOM_IMPACT, value).apply()

    var customRequireFreefall: Boolean
        get() = prefs.getBoolean(KEY_CUSTOM_FREEFALL, true)
        set(value) = prefs.edit().putBoolean(KEY_CUSTOM_FREEFALL, value).apply()

    var customStillnessSeconds: Float
        get() = prefs.getFloat(KEY_CUSTOM_STILLNESS, 1.5f)
        set(value) = prefs.edit().putFloat(KEY_CUSTOM_STILLNESS, value).apply()

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
