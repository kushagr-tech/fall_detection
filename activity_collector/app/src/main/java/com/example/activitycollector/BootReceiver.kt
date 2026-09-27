package com.example.activitycollector

import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.util.Log
import androidx.core.content.ContextCompat

/**
 * BootReceiver
 * ============
 * Restarts FallMonitoringService automatically upon device boot or app update
 * if fall monitoring was active prior to shutdown.
 */
class BootReceiver : BroadcastReceiver() {

    companion object {
        private const val TAG = "BootReceiver"
        const val PREFS_SERVICE_STATE = "fall_service_state"
        const val KEY_MONITORING_ACTIVE = "monitoring_should_run"

        fun setMonitoringPersistedState(context: Context, shouldRun: Boolean) {
            context.getSharedPreferences(PREFS_SERVICE_STATE, Context.MODE_PRIVATE)
                .edit()
                .putBoolean(KEY_MONITORING_ACTIVE, shouldRun)
                .apply()
        }

        fun getMonitoringPersistedState(context: Context): Boolean {
            return context.getSharedPreferences(PREFS_SERVICE_STATE, Context.MODE_PRIVATE)
                .getBoolean(KEY_MONITORING_ACTIVE, false)
        }
    }

    override fun onReceive(context: Context, intent: Intent) {
        val action = intent.action
        Log.i(TAG, "Received system broadcast: $action")

        if (action == Intent.ACTION_BOOT_COMPLETED ||
            action == "android.intent.action.QUICKBOOT_POWERON" ||
            action == Intent.ACTION_MY_PACKAGE_REPLACED
        ) {
            val shouldRun = getMonitoringPersistedState(context)
            if (shouldRun) {
                Log.i(TAG, "Restarting FallMonitoringService after boot/update")
                val serviceIntent = Intent(context, FallMonitoringService::class.java).apply {
                    this.action = FallMonitoringService.ACTION_START_MONITORING
                }
                ContextCompat.startForegroundService(context, serviceIntent)
            }
        }
    }
}
