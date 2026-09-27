package com.example.activitycollector

import android.content.Context
import android.content.SharedPreferences
import android.util.Log
import org.json.JSONArray

/**
 * AlertHistoryManager
 * ===================
 * Stores, queries, and manages incident history of fall events.
 */
class AlertHistoryManager(context: Context) {

    companion object {
        private const val TAG = "AlertHistoryManager"
        private const val PREFS_NAME = "alert_history_prefs"
        private const val KEY_ALERT_LIST = "key_alert_events_json"
        private const val MAX_HISTORY_ITEMS = 100
    }

    private val prefs: SharedPreferences =
        context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)

    fun getAlerts(): List<AlertEvent> {
        val jsonStr = prefs.getString(KEY_ALERT_LIST, null) ?: return emptyList()
        val list = mutableListOf<AlertEvent>()
        try {
            val jsonArray = JSONArray(jsonStr)
            for (i in 0 until jsonArray.length()) {
                val obj = jsonArray.getJSONObject(i)
                list.add(AlertEvent.fromJson(obj))
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error parsing alert history: ${e.message}")
        }
        // Most recent first
        return list.sortedByDescending { it.timestamp }
    }

    fun recordAlert(event: AlertEvent) {
        val currentList = getAlerts().toMutableList()
        currentList.add(0, event) // insert at start
        if (currentList.size > MAX_HISTORY_ITEMS) {
            currentList.removeAt(currentList.lastIndex)
        }
        saveAlerts(currentList)
        Log.i(TAG, "Recorded alert event: ${event.id} outcome=${event.outcome}")
    }

    fun clearHistory() {
        prefs.edit().remove(KEY_ALERT_LIST).apply()
    }

    private fun saveAlerts(events: List<AlertEvent>): Boolean {
        return try {
            val jsonArray = JSONArray()
            for (e in events) {
                jsonArray.put(e.toJson())
            }
            prefs.edit().putString(KEY_ALERT_LIST, jsonArray.toString()).apply()
            true
        } catch (e: Exception) {
            Log.e(TAG, "Error saving alert history: ${e.message}")
            false
        }
    }
}
