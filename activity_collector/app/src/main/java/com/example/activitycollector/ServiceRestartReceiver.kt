package com.example.activitycollector

import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.util.Log
import androidx.core.content.ContextCompat

/**
 * ServiceRestartReceiver
 * ======================
 * Triggered by AlarmManager or system events when the app is swiped away from Recents
 * (via onTaskRemoved) to guarantee that FallMonitoringService is resurrected immediately,
 * mirroring the persistent background architecture of apps like StepSetGo.
 */
class ServiceRestartReceiver : BroadcastReceiver() {

    companion object {
        private const val TAG = "ServiceRestartReceiver"
        const val ACTION_RESTART_SERVICE = "com.example.activitycollector.ACTION_RESTART_SERVICE"
    }

    override fun onReceive(context: Context, intent: Intent) {
        val action = intent.action
        Log.i(TAG, "ServiceRestartReceiver invoked with action: $action")

        val shouldRun = BootReceiver.getMonitoringPersistedState(context)
        if (shouldRun) {
            Log.i(TAG, "Re-launching FallMonitoringService in foreground")
            val serviceIntent = Intent(context, FallMonitoringService::class.java).apply {
                this.action = FallMonitoringService.ACTION_START_MONITORING
            }
            try {
                ContextCompat.startForegroundService(context, serviceIntent)
            } catch (e: Exception) {
                Log.e(TAG, "Failed to restart FallMonitoringService: ${e.message}")
            }
        } else {
            Log.d(TAG, "Monitoring state is disabled; skipping restart.")
        }
    }
}
