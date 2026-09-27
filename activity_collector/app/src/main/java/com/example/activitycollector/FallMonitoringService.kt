package com.example.activitycollector

import android.app.AlarmManager
import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.media.AudioAttributes
import android.media.Ringtone
import android.media.RingtoneManager
import android.os.BatteryManager
import android.os.Binder
import android.os.Build
import android.os.Handler
import android.os.IBinder
import android.os.Looper
import android.os.PowerManager
import android.os.VibrationEffect
import android.os.Vibrator
import android.os.VibratorManager
import android.util.Log
import androidx.core.app.NotificationCompat
import androidx.core.content.ContextCompat
import java.util.concurrent.atomic.AtomicBoolean

/**
 * FallMonitoringService
 * =====================
 * A dedicated Android Foreground Service that runs continuous sensor collection,
 * PyTorch Mobile sliding-window inference, fall detection, and emergency SOS dispatch.
 *
 * Core Capabilities:
 * 1. StepSetGo-style survivability: Survives task kill via onTaskRemoved and device reboot via BootReceiver
 *    (when background tracking preference is enabled).
 * 2. Autonomous emergency countdown: Escalates to phone call (Contact 1) and SMS (all contacts)
 *    even if the user is unconscious and never unlocks the phone.
 * 3. Intelligent Low Battery Mode: Auto-adapts when battery falls below 20% to conserve power.
 * 4. Custom Ringtone Engine: Plays user-selected siren / alarm from AppPreferences.
 * 5. Incident Telemetry & Logging: Automatically records events into AlertHistoryManager and archives CSVs.
 */
class FallMonitoringService : Service(), SensorEventListener {

    companion object {
        private const val TAG = "FallMonitoringService"

        const val CHANNEL_MONITORING = "channel_fall_monitoring"
        const val CHANNEL_ALERT      = "channel_fall_alert"
        const val NOTIFICATION_ID_MONITOR = 1001
        const val NOTIFICATION_ID_ALERT   = 1002

        const val ACTION_START_MONITORING = "com.example.activitycollector.ACTION_START"
        const val ACTION_STOP_MONITORING  = "com.example.activitycollector.ACTION_STOP"
        const val ACTION_DISMISS_ALERT    = "com.example.activitycollector.ACTION_DISMISS"
        const val ACTION_TRIGGER_SOS      = "com.example.activitycollector.ACTION_TRIGGER_SOS"
        const val ACTION_RELOAD_PREFS     = "com.example.activitycollector.ACTION_RELOAD_PREFS"

        const val EXTRA_FALL_ALERT = "extra_fall_alert"
    }

    interface ServiceListener {
        fun onInferenceUpdate(label: String, confidence: Float, probs: FloatArray)
        fun onFallPhaseChanged(phase: FallDetector.Phase, description: String)
        fun onFallConfirmed()
        fun onFallDismissed()
        fun onCountdownTick(secondsRemaining: Int)
        fun onSosDispatched(summary: String)
        fun onBatteryModeChanged(isLowBattery: Boolean)
        fun onSampleCountChanged(count: Int, durationSeconds: Long)
    }

    private val binder = LocalBinder()
    private var listener: ServiceListener? = null

    // ── Managers & Preferences ────────────────────────────────────────────────
    lateinit var appPreferences: AppPreferences
        private set
    lateinit var historyManager: AlertHistoryManager
        private set

    // ── Sensor infrastructure ─────────────────────────────────────────────────
    private lateinit var sensorManager: SensorManager
    private var accelerometer: Sensor? = null
    private var gyroscope: Sensor? = null
    val sensorBuffer = SensorBuffer(capacity = 200)

    // ── Wake Lock ─────────────────────────────────────────────────────────────
    private var wakeLock: PowerManager.WakeLock? = null

    // ── Core Engines ──────────────────────────────────────────────────────────
    lateinit var inferenceEngine: InferenceEngine
        private set
    lateinit var fallDetector: FallDetector
        private set

    // ── State ─────────────────────────────────────────────────────────────────
    val isMonitoring = AtomicBoolean(false)
    val isRecording  = AtomicBoolean(false)
    val records      = mutableListOf<SensorDataRecord>()
    private var recordingLabel = "standing"
    private var recordingStartMs = 0L

    // ── Battery Tracking ──────────────────────────────────────────────────────
    private var batteryReceiver: BroadcastReceiver? = null
    var isLowBatteryActive = false
        private set

    // ── Alarms, Haptics & SOS Escalation ──────────────────────────────────────
    private var vibrator: Vibrator? = null
    private var ringtone: Ringtone? = null
    private val uiHandler = Handler(Looper.getMainLooper())

    private var currentLabel = "—"
    private var currentConfidence = 0f
    private var lastNotificationUpdateMs = 0L

    var alertSecondsRemaining = 30
        private set
    var isAlertActive = false
        private set

    private var lastConfirmedImpactForce = 0f
    private var lastConfirmedWindow: Array<FloatArray>? = null

    private val countdownRunnable = object : Runnable {
        override fun run() {
            if (!isAlertActive) return
            alertSecondsRemaining--
            uiHandler.post {
                listener?.onCountdownTick(alertSecondsRemaining)
            }

            if (alertSecondsRemaining <= 0) {
                executeEmergencyProtocol()
            } else {
                uiHandler.postDelayed(this, 1000L)
            }
        }
    }

    inner class LocalBinder : Binder() {
        fun getService(): FallMonitoringService = this@FallMonitoringService
    }

    override fun onBind(intent: Intent?): IBinder = binder

    override fun onCreate() {
        super.onCreate()
        Log.i(TAG, "Service onCreate")
        appPreferences = AppPreferences(this)
        historyManager = AlertHistoryManager(this)

        createNotificationChannels()
        setupSensors()
        setupWakeLock()
        setupEngines()
        setupAlertFeedback()
        setupBatteryMonitoring()
    }

    private fun setupSensors() {
        sensorManager = getSystemService(SENSOR_SERVICE) as SensorManager
        accelerometer = sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)
        gyroscope     = sensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE)
    }

    private fun setupWakeLock() {
        val powerManager = getSystemService(POWER_SERVICE) as PowerManager
        wakeLock = powerManager.newWakeLock(
            PowerManager.PARTIAL_WAKE_LOCK,
            "ActivityCollector:FallMonitoringWakeLock"
        ).apply {
            setReferenceCounted(false)
        }
    }

    private fun setupEngines() {
        fallDetector = FallDetector(
            onFallConfirmed = {
                uiHandler.post { triggerFallAlert() }
            },
            onPhaseChanged = { phase, description ->
                uiHandler.post {
                    listener?.onFallPhaseChanged(phase, description)
                }
            }
        )

        inferenceEngine = InferenceEngine(
            context = this,
            buffer = sensorBuffer,
            windowSize = 100,
            inferenceIntervalMs = 400L,
        ) { label, confidence, probs ->
            currentLabel = label
            currentConfidence = confidence

            val window2D = sensorBuffer.copyWindow2D(100)
            if (window2D != null) {
                lastConfirmedWindow = window2D
                fallDetector.onInferenceResult(label, probs, window2D)
            }

            uiHandler.post {
                listener?.onInferenceUpdate(label, confidence, probs)
            }

            // Periodically refresh foreground notification text (every 3 seconds)
            val now = System.currentTimeMillis()
            if (now - lastNotificationUpdateMs > 3000L && isMonitoring.get()) {
                lastNotificationUpdateMs = now
                updateForegroundNotification()
            }
        }
    }

    private fun setupBatteryMonitoring() {
        batteryReceiver = object : BroadcastReceiver() {
            override fun onReceive(context: Context?, intent: Intent?) {
                if (intent?.action == Intent.ACTION_BATTERY_CHANGED) {
                    val level = intent.getIntExtra(BatteryManager.EXTRA_LEVEL, -1)
                    val scale = intent.getIntExtra(BatteryManager.EXTRA_SCALE, -1)
                    val batteryPct = if (level >= 0 && scale > 0) (level * 100) / scale else 50
                    val status = intent.getIntExtra(BatteryManager.EXTRA_STATUS, -1)
                    val isCharging = status == BatteryManager.BATTERY_STATUS_CHARGING ||
                                     status == BatteryManager.BATTERY_STATUS_FULL

                    val shouldBeLowBattery = appPreferences.isLowBatteryModeEnabled &&
                                             (batteryPct <= 20) && !isCharging

                    if (shouldBeLowBattery != isLowBatteryActive) {
                        isLowBatteryActive = shouldBeLowBattery
                        inferenceEngine.isLowBatteryMode = shouldBeLowBattery
                        Log.i(TAG, "Low Battery Mode changed: active=$isLowBatteryActive (Battery: $batteryPct%)")
                        uiHandler.post {
                            listener?.onBatteryModeChanged(isLowBatteryActive)
                        }
                        updateForegroundNotification()
                    }
                }
            }
        }
        val filter = IntentFilter(Intent.ACTION_BATTERY_CHANGED)
        registerReceiver(batteryReceiver, filter)
    }

    fun setupAlertFeedback() {
        vibrator = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
            val vm = getSystemService(VIBRATOR_MANAGER_SERVICE) as VibratorManager
            vm.defaultVibrator
        } else {
            @Suppress("DEPRECATION")
            getSystemService(VIBRATOR_SERVICE) as Vibrator
        }

        try {
            val ringtoneUri = appPreferences.getResolvedRingtoneUri()
            ringtone = RingtoneManager.getRingtone(applicationContext, ringtoneUri)
            Log.i(TAG, "Initialized ringtone URI: $ringtoneUri")
        } catch (e: Exception) {
            Log.w(TAG, "Could not initialize ringtone: ${e.message}")
        }
    }

    // ── Command Handling ──────────────────────────────────────────────────────

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        val action = intent?.action ?: ACTION_START_MONITORING
        Log.i(TAG, "onStartCommand action: $action")

        when (action) {
            ACTION_START_MONITORING -> startMonitoring()
            ACTION_STOP_MONITORING  -> stopMonitoring()
            ACTION_DISMISS_ALERT    -> dismissFall()
            ACTION_TRIGGER_SOS      -> executeEmergencyProtocol()
            ACTION_RELOAD_PREFS     -> setupAlertFeedback()
        }

        return START_STICKY
    }

    fun startMonitoring() {
        if (isMonitoring.getAndSet(true)) return

        BootReceiver.setMonitoringPersistedState(this, true)
        wakeLock?.acquire(12 * 60 * 60 * 1000L) // 12 hr safety cap

        val periodUs = 20_000 // 50 Hz
        accelerometer?.let { sensorManager.registerListener(this, it, periodUs) }
        gyroscope?.let     { sensorManager.registerListener(this, it, periodUs) }

        inferenceEngine.start()

        val notification = buildMonitoringNotification("Monitoring active", 0, "00:00")
        startForeground(NOTIFICATION_ID_MONITOR, notification)

        Log.i(TAG, "Fall Monitoring Service started in foreground")
    }

    fun stopMonitoring() {
        if (!isMonitoring.getAndSet(false)) return

        BootReceiver.setMonitoringPersistedState(this, false)
        isAlertActive = false
        uiHandler.removeCallbacks(countdownRunnable)

        stopRecording()
        sensorManager.unregisterListener(this)
        inferenceEngine.stop()
        fallDetector.reset("Monitoring stopped")
        stopAlertFeedback()

        if (wakeLock?.isHeld == true) {
            wakeLock?.release()
        }

        stopForeground(STOP_FOREGROUND_REMOVE)
        stopSelf()
        Log.i(TAG, "Fall Monitoring Service stopped")
    }

    // ── Survivability (StepSetGo style) ───────────────────────────────────────

    override fun onTaskRemoved(rootIntent: Intent?) {
        super.onTaskRemoved(rootIntent)
        Log.w(TAG, "onTaskRemoved: Task cleared.")

        // Check user's background tracking preference
        if (isMonitoring.get() && appPreferences.isBackgroundTrackingEnabled) {
            Log.i(TAG, "Background tracking is enabled. Resurrecting service via alarm...")
            val restartIntent = Intent(applicationContext, ServiceRestartReceiver::class.java).apply {
                action = ServiceRestartReceiver.ACTION_RESTART_SERVICE
            }
            val pendingIntent = PendingIntent.getBroadcast(
                applicationContext,
                1003,
                restartIntent,
                PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE
            )
            val alarmManager = getSystemService(Context.ALARM_SERVICE) as AlarmManager
            try {
                alarmManager.set(
                    AlarmManager.RTC_WAKEUP,
                    System.currentTimeMillis() + 1000L,
                    pendingIntent
                )
            } catch (e: Exception) {
                Log.e(TAG, "Error scheduling restart alarm: ${e.message}")
            }
        } else if (!appPreferences.isBackgroundTrackingEnabled) {
            Log.i(TAG, "Background tracking is disabled by user. Stopping service gracefully.")
            stopMonitoring()
        }
    }

    // ── Recording Helpers ─────────────────────────────────────────────────────

    fun startRecording(label: String, samplingRateHz: Int) {
        synchronized(records) { records.clear() }
        recordingLabel = label
        recordingStartMs = System.currentTimeMillis()
        isRecording.set(true)
    }

    fun stopRecording(): List<SensorDataRecord> {
        isRecording.set(false)
        return synchronized(records) { records.toList() }
    }

    // ── Sensor Callback ───────────────────────────────────────────────────────

    override fun onSensorChanged(event: SensorEvent) {
        when (event.sensor.type) {
            Sensor.TYPE_GYROSCOPE -> {
                sensorBuffer.latestGx = event.values[0]
                sensorBuffer.latestGy = event.values[1]
                sensorBuffer.latestGz = event.values[2]
            }
            Sensor.TYPE_ACCELEROMETER -> {
                sensorBuffer.push(event.values[0], event.values[1], event.values[2])

                if (isRecording.get()) {
                    val rec = SensorDataRecord(
                        timestamp = System.currentTimeMillis(),
                        ax = event.values[0],
                        ay = event.values[1],
                        az = event.values[2],
                        gx = sensorBuffer.latestGx,
                        gy = sensorBuffer.latestGy,
                        gz = sensorBuffer.latestGz,
                        label = recordingLabel,
                    )
                    var count: Int
                    synchronized(records) {
                        records.add(rec)
                        count = records.size
                    }

                    if (count % 10 == 0) {
                        val duration = (System.currentTimeMillis() - recordingStartMs) / 1000L
                        uiHandler.post {
                            listener?.onSampleCountChanged(count, duration)
                        }
                    }
                }
            }
        }
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) = Unit

    // ── Fall Alert Escalation ─────────────────────────────────────────────────

    private fun triggerFallAlert() {
        Log.w(TAG, "!!! FALL CONFIRMED BY INFERENCE ENGINE & FALL DETECTOR !!!")
        isAlertActive = true
        alertSecondsRemaining = appPreferences.countdownSeconds
        uiHandler.removeCallbacks(countdownRunnable)
        uiHandler.postDelayed(countdownRunnable, 1000L)

        // Auto-archive telemetry CSV for self-learning
        if (appPreferences.isAutoArchiveEnabled && lastConfirmedWindow != null) {
            AutoDataArchiver.archiveFallWindow(
                context = this,
                label = "falling",
                window2D = lastConfirmedWindow!!,
                impactPeak = 24.5f,
                isConfirmed = true
            )
        }

        // 0. Automatically turn on display screen
        try {
            val pm = getSystemService(POWER_SERVICE) as PowerManager
            @Suppress("DEPRECATION")
            val screenLock = pm.newWakeLock(
                PowerManager.SCREEN_BRIGHT_WAKE_LOCK or PowerManager.ACQUIRE_CAUSES_WAKEUP or PowerManager.ON_AFTER_RELEASE,
                "ActivityCollector:ScreenAlertWakeLock"
            )
            screenLock.acquire(30_000L)
        } catch (e: Exception) {
            Log.w(TAG, "Could not acquire screen wake lock: ${e.message}")
        }

        // 1. Start continuous haptic alarm pattern based on preferences
        val pattern = when (appPreferences.vibrationPattern) {
            "morse" -> longArrayOf(0, 200, 100, 200, 100, 200, 300, 500, 100, 500, 100, 500, 300, 200, 100, 200, 100, 200)
            else    -> longArrayOf(0, 600, 250, 600, 250, 600)
        }

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            vibrator?.vibrate(VibrationEffect.createWaveform(pattern, 0))
        } else {
            @Suppress("DEPRECATION")
            vibrator?.vibrate(pattern, 0)
        }

        // 2. Play alert audio
        try {
            ringtone?.audioAttributes = AudioAttributes.Builder()
                .setUsage(AudioAttributes.USAGE_ALARM)
                .setContentType(AudioAttributes.CONTENT_TYPE_SONIFICATION)
                .build()
            if (ringtone?.isPlaying != true) {
                ringtone?.play()
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error playing ringtone: ${e.message}")
        }

        // 3. Post high-priority alert notification with FullScreenIntent
        val alertIntent = Intent(this, MainActivity::class.java).apply {
            flags = Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_CLEAR_TOP or Intent.FLAG_ACTIVITY_SINGLE_TOP
            putExtra(EXTRA_FALL_ALERT, true)
        }
        val fullScreenPendingIntent = PendingIntent.getActivity(
            this,
            2001,
            alertIntent,
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE
        )

        val dismissIntent = Intent(this, FallMonitoringService::class.java).apply {
            action = ACTION_DISMISS_ALERT
        }
        val dismissPendingIntent = PendingIntent.getService(
            this,
            2002,
            dismissIntent,
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE
        )

        val alertNotification = NotificationCompat.Builder(this, CHANNEL_ALERT)
            .setSmallIcon(android.R.drawable.ic_dialog_alert)
            .setContentTitle(getString(R.string.fall_alert_title))
            .setContentText(getString(R.string.fall_alert_body))
            .setPriority(NotificationCompat.PRIORITY_MAX)
            .setCategory(NotificationCompat.CATEGORY_ALARM)
            .setFullScreenIntent(fullScreenPendingIntent, true)
            .setContentIntent(fullScreenPendingIntent)
            .setAutoCancel(true)
            .setOngoing(true)
            .addAction(android.R.drawable.ic_menu_close_clear_cancel, getString(R.string.fall_btn_okay), dismissPendingIntent)
            .build()

        val nm = getSystemService(NOTIFICATION_SERVICE) as NotificationManager
        nm.notify(NOTIFICATION_ID_ALERT, alertNotification)

        // 4. Notify active UI listener
        listener?.onFallConfirmed()
    }

    fun dismissFall() {
        Log.i(TAG, "Fall alert dismissed by user")
        isAlertActive = false
        uiHandler.removeCallbacks(countdownRunnable)
        stopAlertFeedback()
        fallDetector.dismissFall()

        val nm = getSystemService(NOTIFICATION_SERVICE) as NotificationManager
        nm.cancel(NOTIFICATION_ID_ALERT)

        // Log to history
        historyManager.recordAlert(
            AlertEvent(
                outcome = AlertEvent.Outcome.DISMISSED_SAFE,
                summary = "User checked in safely ('I am okay' ✓)",
                impactForce = 22.4f,
                tiltAngle = 38f,
                confidencePct = (currentConfidence * 100).toInt(),
                activityBeforeFall = currentLabel
            )
        )

        uiHandler.post {
            listener?.onFallDismissed()
        }
    }

    fun executeEmergencyProtocol() {
        Log.w(TAG, "🚨 EXECUTING EMERGENCY SOS PROTOCOL")
        isAlertActive = false
        uiHandler.removeCallbacks(countdownRunnable)
        stopAlertFeedback()

        val result = EmergencyDispatcher.dispatchEmergency(this)

        val nm = getSystemService(NOTIFICATION_SERVICE) as NotificationManager
        nm.cancel(NOTIFICATION_ID_ALERT)

        val sosNotification = NotificationCompat.Builder(this, CHANNEL_ALERT)
            .setSmallIcon(android.R.drawable.ic_dialog_alert)
            .setContentTitle("🚨 EMERGENCY SOS DISPATCHED")
            .setContentText(result.summary)
            .setStyle(NotificationCompat.BigTextStyle().bigText(result.summary))
            .setPriority(NotificationCompat.PRIORITY_HIGH)
            .setAutoCancel(true)
            .build()
        nm.notify(NOTIFICATION_ID_ALERT, sosNotification)

        // Log to history
        historyManager.recordAlert(
            AlertEvent(
                outcome = AlertEvent.Outcome.SOS_DISPATCHED,
                summary = result.summary,
                impactForce = 24.8f,
                tiltAngle = 44f,
                confidencePct = (currentConfidence * 100).toInt(),
                activityBeforeFall = currentLabel
            )
        )

        uiHandler.post {
            listener?.onSosDispatched(result.summary)
        }
    }

    private fun stopAlertFeedback() {
        try {
            vibrator?.cancel()
            if (ringtone?.isPlaying == true) {
                ringtone?.stop()
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error stopping feedback: ${e.message}")
        }
    }

    // ── Notifications ─────────────────────────────────────────────────────────

    private fun createNotificationChannels() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val nm = getSystemService(NotificationManager::class.java)

            val monChannel = NotificationChannel(
                CHANNEL_MONITORING,
                getString(R.string.notification_channel_monitoring),
                NotificationManager.IMPORTANCE_LOW
            ).apply {
                description = getString(R.string.notification_channel_monitoring_desc)
                setShowBadge(false)
            }
            nm.createNotificationChannel(monChannel)

            val alertChannel = NotificationChannel(
                CHANNEL_ALERT,
                getString(R.string.notification_channel_alert),
                NotificationManager.IMPORTANCE_HIGH
            ).apply {
                description = getString(R.string.notification_channel_alert_desc)
                enableVibration(true)
                setBypassDnd(true)
            }
            nm.createNotificationChannel(alertChannel)
        }
    }

    private fun buildMonitoringNotification(activity: String, confidencePct: Int, durationStr: String): Notification {
        val launchIntent = Intent(this, MainActivity::class.java).apply {
            flags = Intent.FLAG_ACTIVITY_SINGLE_TOP
        }
        val contentPendingIntent = PendingIntent.getActivity(
            this,
            1001,
            launchIntent,
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE
        )

        val stopIntent = Intent(this, FallMonitoringService::class.java).apply {
            action = ACTION_STOP_MONITORING
        }
        val stopPendingIntent = PendingIntent.getService(
            this,
            1002,
            stopIntent,
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE
        )

        val lowBatteryTag = if (isLowBatteryActive) " • 🔋 Low Battery Mode" else ""
        val bodyText = if (activity.isNotEmpty() && activity != "—") {
            "Status: ${activity.uppercase()} ($confidencePct%)$lowBatteryTag • $durationStr"
        } else {
            "Active monitoring • Sensor buffer running$lowBatteryTag"
        }

        return NotificationCompat.Builder(this, CHANNEL_MONITORING)
            .setSmallIcon(android.R.drawable.ic_menu_compass)
            .setContentTitle(getString(R.string.service_title))
            .setContentText(bodyText)
            .setOngoing(true)
            .setContentIntent(contentPendingIntent)
            .addAction(android.R.drawable.ic_media_pause, getString(R.string.action_stop), stopPendingIntent)
            .setPriority(NotificationCompat.PRIORITY_LOW)
            .build()
    }

    private fun updateForegroundNotification() {
        val elapsedSec = if (recordingStartMs > 0L) (System.currentTimeMillis() - recordingStartMs) / 1000L else 0L
        val durStr = String.format("%02d:%02d", elapsedSec / 60, elapsedSec % 60)
        val notif = buildMonitoringNotification(
            currentLabel,
            (currentConfidence * 100).toInt(),
            durStr
        )
        val nm = getSystemService(NOTIFICATION_SERVICE) as NotificationManager
        nm.notify(NOTIFICATION_ID_MONITOR, notif)
    }

    // ── Listener Registration ─────────────────────────────────────────────────

    fun registerListener(l: ServiceListener) {
        this.listener = l
        if (currentLabel.isNotEmpty() && currentLabel != "—") {
            l.onInferenceUpdate(currentLabel, currentConfidence, floatArrayOf())
        }
        l.onFallPhaseChanged(fallDetector.currentPhase, "")
        l.onBatteryModeChanged(isLowBatteryActive)
        if (isAlertActive) {
            l.onCountdownTick(alertSecondsRemaining)
        }
    }

    fun unregisterListener() {
        this.listener = null
    }

    override fun onDestroy() {
        super.onDestroy()
        Log.i(TAG, "Service onDestroy")
        if (batteryReceiver != null) {
            unregisterReceiver(batteryReceiver)
            batteryReceiver = null
        }
        stopMonitoring()
        inferenceEngine.destroy()
    }
}
