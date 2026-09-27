package com.example.activitycollector

import android.Manifest
import android.app.Activity
import android.content.ComponentName
import android.content.Context
import android.content.Intent
import android.content.ServiceConnection
import android.content.pm.PackageManager
import android.media.Ringtone
import android.media.RingtoneManager
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.os.Handler
import android.os.IBinder
import android.os.Looper
import android.os.PowerManager
import android.provider.Settings
import android.view.LayoutInflater
import android.view.View
import android.view.WindowManager
import android.widget.AdapterView
import android.widget.ArrayAdapter
import android.widget.Button
import android.widget.EditText
import android.widget.ImageButton
import android.widget.TextView
import android.widget.Toast
import android.util.Log
import android.provider.ContactsContract
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.appcompat.app.AppCompatDelegate
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.example.activitycollector.databinding.ActivityMainBinding

/**
 * MainActivity
 * ============
 * Front-end dashboard and control center for FallGuard AI.
 * Built with Google Material Design 3 (Material You) styling.
 *
 * Features:
 * 1. Guard Dashboard: Live activity dial, posture recognition, and instant SOS.
 * 2. Alert History: Chronological log of fall alerts, outcomes, and GPS locations.
 * 3. Settings: 24/7 background tracking, intelligent low battery mode, ringtones,
 *    and automated self-learning data archiving.
 */
class MainActivity : AppCompatActivity(), FallMonitoringService.ServiceListener {

    private lateinit var binding: ActivityMainBinding
    private lateinit var contactManager: EmergencyContactManager
    private lateinit var historyManager: AlertHistoryManager
    private lateinit var appPreferences: AppPreferences

    // ── Service Connection ────────────────────────────────────────────────────
    private var monitoringService: FallMonitoringService? = null
    private var isServiceBound = false

    private val serviceConnection = object : ServiceConnection {
        override fun onServiceConnected(name: ComponentName?, service: IBinder?) {
            val binder = service as? FallMonitoringService.LocalBinder
            monitoringService = binder?.getService()
            isServiceBound = true
            monitoringService?.registerListener(this@MainActivity)

            monitoringService?.let { s ->
                val active = s.isMonitoring.get()
                updateUiMonitoringState(active)
                updateLowBatteryIndicator(s.isLowBatteryActive)
                if (s.isAlertActive || s.fallDetector.currentPhase == FallDetector.Phase.CONFIRMED) {
                    showFallAlert()
                    fallAlertDialog.updateCountdown(s.alertSecondsRemaining)
                }
            }
        }

        override fun onServiceDisconnected(name: ComponentName?) {
            monitoringService?.unregisterListener()
            monitoringService = null
            isServiceBound = false
        }
    }

    // ── Fall Alert Dialog ─────────────────────────────────────────────────────
    private lateinit var fallAlertDialog: FallAlertDialog

    private val permissionRequestCode = 1001
    private var previewRingtone: Ringtone? = null

    // ── Activity Result for Ringtone Picker ───────────────────────────────────
    private val ringtonePickerLauncher = registerForActivityResult(
        ActivityResultContracts.StartActivityForResult()
    ) { result ->
        if (result.resultCode == Activity.RESULT_OK) {
            val uri: Uri? = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
                result.data?.getParcelableExtra(RingtoneManager.EXTRA_RINGTONE_PICKED_URI, Uri::class.java)
            } else {
                @Suppress("DEPRECATION")
                result.data?.getParcelableExtra(RingtoneManager.EXTRA_RINGTONE_PICKED_URI)
            }
            if (uri != null) {
                appPreferences.alertRingtoneUri = uri.toString()
                updateSelectedRingtoneLabel()
                monitoringService?.setupAlertFeedback()
                showToast("Alarm sound updated.")
            }
        }
    }

    // ── Contact Picker Launchers & Callbacks ──────────────────────────────────
    private var pendingContactCallback: ((String, String) -> Unit)? = null

    private val contactPickerLauncher = registerForActivityResult(
        ActivityResultContracts.StartActivityForResult()
    ) { result ->
        if (result.resultCode == Activity.RESULT_OK) {
            val contactUri: Uri? = result.data?.data
            if (contactUri != null) {
                handleContactPicked(contactUri)
            }
        }
    }

    private val requestContactsPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { _ ->
        // Proceed with picker intent either way (system picker works without READ_CONTACTS on Android 7+)
        launchContactPickerIntent()
    }

    private fun pickContactWithPermission(callback: ((String, String) -> Unit)? = null) {
        pendingContactCallback = callback
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.READ_CONTACTS) == PackageManager.PERMISSION_GRANTED) {
            launchContactPickerIntent()
        } else {
            requestContactsPermissionLauncher.launch(Manifest.permission.READ_CONTACTS)
        }
    }

    private fun launchContactPickerIntent() {
        val pickIntent = Intent(Intent.ACTION_PICK, ContactsContract.CommonDataKinds.Phone.CONTENT_URI)
        try {
            contactPickerLauncher.launch(pickIntent)
        } catch (e: Exception) {
            showToast("Could not open contacts: ${e.message}")
        }
    }

    private fun handleContactPicked(contactUri: Uri) {
        val projection = arrayOf(
            ContactsContract.CommonDataKinds.Phone.DISPLAY_NAME,
            ContactsContract.CommonDataKinds.Phone.NUMBER
        )
        var name = ""
        var phone = ""
        try {
            contentResolver.query(contactUri, projection, null, null, null)?.use { cursor ->
                if (cursor.moveToFirst()) {
                    val nameIdx = cursor.getColumnIndex(ContactsContract.CommonDataKinds.Phone.DISPLAY_NAME)
                    val phoneIdx = cursor.getColumnIndex(ContactsContract.CommonDataKinds.Phone.NUMBER)
                    if (nameIdx >= 0) name = cursor.getString(nameIdx) ?: ""
                    if (phoneIdx >= 0) phone = cursor.getString(phoneIdx) ?: ""
                }
            }
        } catch (e: Exception) {
            Log.e("MainActivity", "Error reading contact from URI: ${e.message}")
        }

        if (phone.isNotEmpty()) {
            if (name.isEmpty()) name = "Contact"
            val callback = pendingContactCallback
            if (callback != null) {
                callback.invoke(name, phone)
                pendingContactCallback = null
            } else {
                contactManager.addContact(name, phone)
                renderSettingsContacts()
                updateQuickContactCard()
                checkAndRequestSosPermissions()
                showToast("Added $name ($phone) to emergency contacts.")
            }
        } else {
            showToast("Could not read phone number from selected contact.")
        }
    }

    // ── Live Sensitivity Motion Tracker ───────────────────────────────────────
    private var livePeakAcc = 9.8f

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        appPreferences = AppPreferences(this)
        applyThemeMode(appPreferences.themeMode)

        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        contactManager = EmergencyContactManager(this)
        historyManager = AlertHistoryManager(this)

        configureLockScreenVisibility()
        setupNavigation()
        setupDashboard()
        setupHistoryTab()
        setupSensitivityTab()
        setupSettingsTab()
        setupFallAlertDialog()

        // Bind to background monitoring service
        val serviceIntent = Intent(this, FallMonitoringService::class.java)
        bindService(serviceIntent, serviceConnection, Context.BIND_AUTO_CREATE)

        val initialTab = savedInstanceState?.getInt("KEY_SELECTED_TAB") ?: R.id.nav_dashboard
        if (initialTab != R.id.nav_dashboard) {
            binding.bottomNavigation.selectedItemId = initialTab
        }

        handleAlertIntent(intent)
    }

    override fun onSaveInstanceState(outState: Bundle) {
        super.onSaveInstanceState(outState)
        outState.putInt("KEY_SELECTED_TAB", binding.bottomNavigation.selectedItemId)
    }

    private fun applyThemeMode(mode: String) {
        val targetMode = when (mode) {
            "light" -> AppCompatDelegate.MODE_NIGHT_NO
            "dark"  -> AppCompatDelegate.MODE_NIGHT_YES
            else    -> AppCompatDelegate.MODE_NIGHT_FOLLOW_SYSTEM
        }
        if (AppCompatDelegate.getDefaultNightMode() != targetMode) {
            AppCompatDelegate.setDefaultNightMode(targetMode)
        }
    }

    override fun onResume() {
        super.onResume()
        updateBatteryOptimizationStatus()
        updateQuickContactCard()
        updateSelectedRingtoneLabel()
    }

    override fun onNewIntent(intent: Intent?) {
        super.onNewIntent(intent)
        setIntent(intent)
        handleAlertIntent(intent)
    }

    private fun handleAlertIntent(intent: Intent?) {
        if (intent?.getBooleanExtra(FallMonitoringService.EXTRA_FALL_ALERT, false) == true) {
            showFallAlert()
        }
    }

    private fun configureLockScreenVisibility() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O_MR1) {
            setShowWhenLocked(true)
            setTurnScreenOn(true)
        } else {
            @Suppress("DEPRECATION")
            window.addFlags(
                WindowManager.LayoutParams.FLAG_SHOW_WHEN_LOCKED or
                WindowManager.LayoutParams.FLAG_TURN_SCREEN_ON or
                WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON
            )
        }
    }

    // ── Navigation (Material 3 Tabs) ──────────────────────────────────────────

    private fun setupNavigation() {
        binding.bottomNavigation.setOnItemSelectedListener { item ->
            when (item.itemId) {
                R.id.nav_dashboard -> {
                    binding.viewDashboard.visibility = View.VISIBLE
                    binding.viewHistory.visibility   = View.GONE
                    binding.viewSensitivity.visibility = View.GONE
                    binding.viewSettings.visibility  = View.GONE
                    true
                }
                R.id.nav_history -> {
                    binding.viewDashboard.visibility = View.GONE
                    binding.viewHistory.visibility   = View.VISIBLE
                    binding.viewSensitivity.visibility = View.GONE
                    binding.viewSettings.visibility  = View.GONE
                    loadHistory()
                    true
                }
                R.id.nav_sensitivity -> {
                    binding.viewDashboard.visibility = View.GONE
                    binding.viewHistory.visibility   = View.GONE
                    binding.viewSensitivity.visibility = View.VISIBLE
                    binding.viewSettings.visibility  = View.GONE
                    updateSensitivityTabUi()
                    true
                }
                R.id.nav_settings -> {
                    binding.viewDashboard.visibility = View.GONE
                    binding.viewHistory.visibility   = View.GONE
                    binding.viewSensitivity.visibility = View.GONE
                    binding.viewSettings.visibility  = View.VISIBLE
                    renderSettingsContacts()
                    true
                }
                else -> false
            }
        }
    }

    // ── Tab 1: Guard Dashboard ────────────────────────────────────────────────

    private fun setupDashboard() {
        binding.btnToggleGuard.setOnClickListener {
            val isRunning = monitoringService?.isMonitoring?.get() == true
            if (isRunning) {
                handleStopMonitoring()
            } else {
                handleStartMonitoring()
            }
        }

        binding.cardPrimaryContactQuick.setOnClickListener {
            // Switch to settings tab to manage contacts
            binding.bottomNavigation.selectedItemId = R.id.nav_settings
        }

        binding.btnInstantSos.setOnClickListener {
            AlertDialog.Builder(this)
                .setTitle("Trigger Emergency SOS?")
                .setMessage("This will immediately call your Primary Emergency Contact and broadcast an emergency SMS with your live GPS location.")
                .setPositiveButton("🚨 CALL FOR HELP NOW") { _, _ ->
                    monitoringService?.executeEmergencyProtocol()
                }
                .setNegativeButton("Cancel", null)
                .show()
        }

        updateQuickContactCard()
    }

    private fun updateQuickContactCard() {
        val primary = contactManager.getPrimaryContact()
        val allContacts = contactManager.getContacts()
        if (primary != null) {
            binding.tvQuickContactName.text = primary.name
            binding.tvQuickContactPhone.text = "${primary.phone} • ${allContacts.size} total contacts"
        } else {
            binding.tvQuickContactName.text = "No Contacts Registered"
            binding.tvQuickContactPhone.text = "Tap to setup emergency contacts in Settings"
        }
    }

    private fun updateUiMonitoringState(isActive: Boolean) {
        if (isActive) {
            binding.tvHeroTitle.text = "PROTECTION ACTIVE"
            binding.tvHeroTitle.setTextColor(ContextCompat.getColor(this, R.color.primary_emerald_dark))
            binding.tvHeroSubtitle.text = "Real-time 50 Hz continuous monitoring"
            binding.tvHeroIcon.text = "🛡️"
            binding.btnToggleGuard.text = "Pause Guard"
            binding.btnToggleGuard.backgroundTintList = ContextCompat.getColorStateList(this, R.color.primary_emerald)
            binding.cardHeroGuard.background = ContextCompat.getDrawable(this, R.drawable.card_hero_guard)
            binding.tvTopStatusBadge.text = "● Guarding"
            binding.tvTopStatusBadge.setTextColor(ContextCompat.getColor(this, R.color.primary_emerald))
            binding.tvTopStatusBadge.backgroundTintList = ContextCompat.getColorStateList(this, R.color.primary_emerald_light)
        } else {
            binding.tvHeroTitle.text = "GUARD PAUSED"
            binding.tvHeroTitle.setTextColor(ContextCompat.getColor(this, R.color.text_secondary))
            binding.tvHeroSubtitle.text = "Tap button below to activate fall protection"
            binding.tvHeroIcon.text = "⏸️"
            binding.btnToggleGuard.text = "Resume Guard"
            binding.btnToggleGuard.backgroundTintList = ContextCompat.getColorStateList(this, R.color.btn_export)
            binding.cardHeroGuard.background = ContextCompat.getDrawable(this, R.drawable.card_m3_background)
            binding.tvTopStatusBadge.text = "❚❚ Paused"
            binding.tvTopStatusBadge.setTextColor(ContextCompat.getColor(this, R.color.text_secondary))
            binding.tvTopStatusBadge.backgroundTintList = ContextCompat.getColorStateList(this, R.color.surface_variant)
        }
    }

    private fun updateLowBatteryIndicator(isLowBattery: Boolean) {
        binding.tvLowBatteryBadge.visibility = if (isLowBattery) View.VISIBLE else View.GONE
    }

    // ── Tab 2: Alert History ──────────────────────────────────────────────────

    private fun setupHistoryTab() {
        binding.btnClearHistory.setOnClickListener {
            AlertDialog.Builder(this)
                .setTitle("Clear History")
                .setMessage("Clear all recorded incident logs?")
                .setPositiveButton("Clear") { _, _ ->
                    historyManager.clearHistory()
                    loadHistory()
                    showToast("History cleared.")
                }
                .setNegativeButton("Cancel", null)
                .show()
        }
    }

    private fun loadHistory() {
        val alerts = historyManager.getAlerts()
        binding.containerHistoryItems.removeAllViews()

        if (alerts.isEmpty()) {
            binding.layoutEmptyHistory.visibility = View.VISIBLE
            binding.scrollHistoryList.visibility = View.GONE
            binding.tvHistoryCount.text = "0 recorded incidents"
            return
        }

        binding.layoutEmptyHistory.visibility = View.GONE
        binding.scrollHistoryList.visibility = View.VISIBLE
        binding.tvHistoryCount.text = "${alerts.size} recorded incident(s)"

        val inflater = LayoutInflater.from(this)
        alerts.forEach { event ->
            val itemView = inflater.inflate(R.layout.item_alert_history, binding.containerHistoryItems, false)

            val tvIcon = itemView.findViewById<TextView>(R.id.tvHistoryIcon)
            val tvTitle = itemView.findViewById<TextView>(R.id.tvHistoryTitle)
            val tvTime = itemView.findViewById<TextView>(R.id.tvHistoryTime)
            val tvBadge = itemView.findViewById<TextView>(R.id.tvHistoryBadge)
            val tvSummary = itemView.findViewById<TextView>(R.id.tvHistorySummary)
            val tvMetrics = itemView.findViewById<TextView>(R.id.tvHistoryMetrics)
            val btnMap = itemView.findViewById<Button>(R.id.btnHistoryMap)

            tvTime.text = "${event.relativeTimeSpan} • ${event.formattedDateTime}"
            tvSummary.text = event.summary.ifEmpty { "Incident handled" }
            tvMetrics.text = "Impact: ${"%.1f".format(event.impactForce)} m/s² • Tilt: ${"%.0f".format(event.tiltAngle)}° • Conf: ${event.confidencePct}%"

            when (event.outcome) {
                AlertEvent.Outcome.DISMISSED_SAFE -> {
                    tvIcon.text = "🛡️"
                    tvTitle.text = "Safety Check-In"
                    tvBadge.text = "Safe Check-in"
                    tvBadge.setTextColor(ContextCompat.getColor(this, R.color.primary_emerald))
                    tvBadge.backgroundTintList = ContextCompat.getColorStateList(this, R.color.primary_emerald_light)
                }
                AlertEvent.Outcome.SOS_DISPATCHED -> {
                    tvIcon.text = "🚨"
                    tvTitle.text = "Emergency SOS Dispatched"
                    tvBadge.text = "SOS Triggered"
                    tvBadge.setTextColor(ContextCompat.getColor(this, R.color.emergency_crimson))
                    tvBadge.backgroundTintList = ContextCompat.getColorStateList(this, R.color.emergency_crimson_light)
                }
                AlertEvent.Outcome.CANCELLED_EARLY -> {
                    tvIcon.text = "⚠️"
                    tvTitle.text = "False Motion Filtered"
                    tvBadge.text = "Cancelled"
                    tvBadge.setTextColor(ContextCompat.getColor(this, R.color.text_secondary))
                    tvBadge.backgroundTintList = ContextCompat.getColorStateList(this, R.color.surface_variant)
                }
            }

            if (event.mapsUrl != null) {
                btnMap.visibility = View.VISIBLE
                btnMap.setOnClickListener {
                    try {
                        val mapIntent = Intent(Intent.ACTION_VIEW, Uri.parse(event.mapsUrl))
                        startActivity(mapIntent)
                    } catch (e: Exception) {
                        showToast("Could not open maps application.")
                    }
                }
            } else {
                btnMap.visibility = View.GONE
            }

            binding.containerHistoryItems.addView(itemView)
        }
    }

    // ── Tab 3: Detection Sensitivity & Desk Surface Calibration ──────────────

    private fun setupSensitivityTab() {
        binding.cardPresetDeskSafe.setOnClickListener { selectSensitivityPreset("desk_safe") }
        binding.rbPresetDeskSafe.setOnClickListener   { selectSensitivityPreset("desk_safe") }

        binding.cardPresetBalanced.setOnClickListener { selectSensitivityPreset("balanced") }
        binding.rbPresetBalanced.setOnClickListener   { selectSensitivityPreset("balanced") }

        binding.cardPresetHigh.setOnClickListener     { selectSensitivityPreset("high") }
        binding.rbPresetHigh.setOnClickListener       { selectSensitivityPreset("high") }

        binding.cardPresetCustom.setOnClickListener   { selectSensitivityPreset("custom") }
        binding.rbPresetCustom.setOnClickListener     { selectSensitivityPreset("custom") }

        // Custom Slider: Impact Force Spike
        val initialImpact = appPreferences.customImpactThreshold.coerceIn(15.0f, 35.0f)
        val snappedImpact = Math.round(initialImpact).toFloat()
        binding.sliderCustomImpact.value = snappedImpact
        binding.tvLabelCustomImpact.text = "Impact Spike Threshold: ${"%.1f".format(snappedImpact)} m/s²"
        binding.sliderCustomImpact.addOnChangeListener { _, value, fromUser ->
            if (fromUser) {
                val rounded = Math.round(value).toFloat()
                appPreferences.customImpactThreshold = rounded
                binding.tvLabelCustomImpact.text = "Impact Spike Threshold: ${"%.1f".format(rounded)} m/s²"
                binding.tvLiveThreshold.text = "${"%.1f".format(rounded)} m/s²"
                monitoringService?.applySensitivityPreferences()
            }
        }

        // Custom Switch: Require Free-Fall Weightlessness Dip
        binding.switchCustomFreefall.isChecked = appPreferences.customRequireFreefall
        binding.switchCustomFreefall.setOnCheckedChangeListener { _, isChecked ->
            appPreferences.customRequireFreefall = isChecked
            monitoringService?.applySensitivityPreferences()
        }

        // Custom Slider: Post-Fall Stillness Delay
        val initialStillness = appPreferences.customStillnessSeconds.coerceIn(0.5f, 3.0f)
        val snappedStillness = ((Math.round((initialStillness - 0.5f) / 0.1f) * 0.1f) + 0.5f).coerceIn(0.5f, 3.0f)
        val validStillness = String.format(java.util.Locale.US, "%.1f", snappedStillness).toFloat()
        binding.sliderCustomStillness.value = validStillness
        binding.tvLabelCustomStillness.text = "Post-Fall Stillness Delay: ${"%.1f".format(validStillness)} seconds"
        binding.sliderCustomStillness.addOnChangeListener { _, value, fromUser ->
            if (fromUser) {
                val rounded = String.format(java.util.Locale.US, "%.1f", value).toFloat()
                appPreferences.customStillnessSeconds = rounded
                binding.tvLabelCustomStillness.text = "Post-Fall Stillness Delay: ${"%.1f".format(rounded)} seconds"
                monitoringService?.applySensitivityPreferences()
            }
        }

        // Live Peak Reset Button
        binding.btnResetPeakAcc.setOnClickListener {
            livePeakAcc = 9.8f
            binding.tvLivePeakAcc.text = "9.8 m/s²"
            binding.progressLiveAcc.progress = 10
            binding.tvLiveCalibratorResult.text = "✅ SAFE • Desk placement will not trigger alert"
            binding.tvLiveCalibratorResult.setTextColor(ContextCompat.getColor(this, R.color.primary_emerald))
            binding.tvLiveCalibratorResult.backgroundTintList = ContextCompat.getColorStateList(this, R.color.primary_emerald_light)
        }

        updateSensitivityTabUi()
    }

    private fun selectSensitivityPreset(presetKey: String) {
        val changed = appPreferences.sensitivityPreset != presetKey
        appPreferences.sensitivityPreset = presetKey
        monitoringService?.applySensitivityPreferences()
        updateSensitivityTabUi()
        if (changed) {
            showToast("Sensitivity preset updated: ${getPresetTitle(presetKey)}")
        }
    }

    private fun getPresetTitle(preset: String): String = when (preset) {
        "desk_safe" -> "Desk-Safe Calibration"
        "balanced"  -> "Balanced Everyday Guard"
        "high"      -> "High Vigilance (Elderly / Risk)"
        "custom"    -> "Custom Fine-Tuning"
        else        -> "Desk-Safe Calibration"
    }

    private fun getPresetDescription(preset: String): String = when (preset) {
        "desk_safe" -> "Ignores desk placement & bed toss. Requires >26 m/s² impact + weightless free-fall dip."
        "balanced"  -> "Balanced multi-phase sensitivity (>22 m/s² impact) for active everyday routines."
        "high"      -> "Maximum protection (>18 m/s² impact) for frail individuals or soft, slow slips."
        "custom"    -> "User-defined custom thresholds for impact, weightlessness dip, and stillness."
        else        -> "Ignores desk placement & bed toss."
    }

    private fun updateSensitivityTabUi() {
        val currentPreset = appPreferences.sensitivityPreset
        binding.rbPresetDeskSafe.isChecked = (currentPreset == "desk_safe")
        binding.rbPresetBalanced.isChecked = (currentPreset == "balanced")
        binding.rbPresetHigh.isChecked     = (currentPreset == "high")
        binding.rbPresetCustom.isChecked   = (currentPreset == "custom")

        binding.containerCustomSensitivity.visibility = if (currentPreset == "custom") View.VISIBLE else View.GONE

        binding.tvSensitivityActiveTitle.text = getPresetTitle(currentPreset)
        binding.tvSensitivityActiveDesc.text  = getPresetDescription(currentPreset)

        val threshold = when (currentPreset) {
            "desk_safe" -> 26.0f
            "balanced"  -> 22.0f
            "high"      -> 18.0f
            "custom"    -> appPreferences.customImpactThreshold
            else        -> 26.0f
        }
        binding.tvLiveThreshold.text = "${"%.1f".format(threshold)} m/s²"
    }

    // ── Tab 4: Settings & Preferences ─────────────────────────────────────────

    private fun setupSettingsTab() {
        // 0. Appearance & Theme Mode Toggle
        when (appPreferences.themeMode) {
            "light" -> binding.toggleThemeGroup.check(R.id.btnThemeLight)
            "dark"  -> binding.toggleThemeGroup.check(R.id.btnThemeDark)
            else    -> binding.toggleThemeGroup.check(R.id.btnThemeSystem)
        }
        binding.toggleThemeGroup.addOnButtonCheckedListener { _, checkedId, isChecked ->
            if (!isChecked) return@addOnButtonCheckedListener
            val newMode = when (checkedId) {
                R.id.btnThemeLight -> "light"
                R.id.btnThemeDark  -> "dark"
                else               -> "system"
            }
            if (appPreferences.themeMode != newMode) {
                appPreferences.themeMode = newMode
                applyThemeMode(newMode)
            }
        }

        // 1. Background Tracking Switch
        binding.switchBackgroundTracking.isChecked = appPreferences.isBackgroundTrackingEnabled
        binding.switchBackgroundTracking.setOnCheckedChangeListener { _, isChecked ->
            appPreferences.isBackgroundTrackingEnabled = isChecked
            showToast(if (isChecked) "24/7 background tracking enabled." else "Background tracking disabled (foreground only).")
        }

        // 2. Low Battery Mode Switch
        binding.switchLowBatteryMode.isChecked = appPreferences.isLowBatteryModeEnabled
        binding.switchLowBatteryMode.setOnCheckedChangeListener { _, isChecked ->
            appPreferences.isLowBatteryModeEnabled = isChecked
            monitoringService?.inferenceEngine?.isLowBatteryMode = isChecked
            showToast(if (isChecked) "Intelligent Low Battery Mode active." else "Low Battery Mode disabled.")
        }

        // 3. Battery Whitelist Card
        binding.cardSettingsBatteryOptimization.setOnClickListener {
            val pm = getSystemService(POWER_SERVICE) as PowerManager
            if (!pm.isIgnoringBatteryOptimizations(packageName)) {
                try {
                    val intent = Intent(Settings.ACTION_REQUEST_IGNORE_BATTERY_OPTIMIZATIONS).apply {
                        data = Uri.parse("package:$packageName")
                    }
                    startActivity(intent)
                } catch (_: Exception) {
                    val intent = Intent(Settings.ACTION_IGNORE_BATTERY_OPTIMIZATION_SETTINGS)
                    startActivity(intent)
                }
            } else {
                showToast("Unrestricted execution is already active.")
            }
        }
        updateBatteryOptimizationStatus()

        // 4. Ringtone Picker
        binding.layoutRingtonePicker.setOnClickListener {
            val intent = Intent(RingtoneManager.ACTION_RINGTONE_PICKER).apply {
                putExtra(RingtoneManager.EXTRA_RINGTONE_TYPE, RingtoneManager.TYPE_ALARM)
                putExtra(RingtoneManager.EXTRA_RINGTONE_TITLE, "Select Fall Siren Sound")
                putExtra(RingtoneManager.EXTRA_RINGTONE_EXISTING_URI, appPreferences.getResolvedRingtoneUri())
                putExtra(RingtoneManager.EXTRA_RINGTONE_SHOW_DEFAULT, true)
                putExtra(RingtoneManager.EXTRA_RINGTONE_SHOW_SILENT, false)
            }
            ringtonePickerLauncher.launch(intent)
        }
        updateSelectedRingtoneLabel()

        // 5. Ringtone Preview Button
        binding.btnPreviewRingtone.setOnClickListener {
            playRingtonePreview()
        }

        // 6. Countdown Duration Spinner
        val durations = listOf("15 seconds", "30 seconds (Default)", "45 seconds")
        val spinnerAdapter = ArrayAdapter(this, R.layout.item_spinner_selected, durations)
        spinnerAdapter.setDropDownViewResource(R.layout.item_spinner_dropdown)
        binding.spinnerCountdown.adapter = spinnerAdapter
        when (appPreferences.countdownSeconds) {
            15 -> binding.spinnerCountdown.setSelection(0)
            45 -> binding.spinnerCountdown.setSelection(2)
            else -> binding.spinnerCountdown.setSelection(1)
        }
        binding.spinnerCountdown.onItemSelectedListener = object : AdapterView.OnItemSelectedListener {
            override fun onItemSelected(parent: AdapterView<*>?, view: View?, position: Int, id: Long) {
                appPreferences.countdownSeconds = when (position) {
                    0 -> 15
                    2 -> 45
                    else -> 30
                }
            }
            override fun onNothingSelected(parent: AdapterView<*>?) = Unit
        }

        // 7. Auto-Archive Telemetry Switch
        binding.switchAutoArchive.isChecked = appPreferences.isAutoArchiveEnabled
        binding.switchAutoArchive.setOnCheckedChangeListener { _, isChecked ->
            appPreferences.isAutoArchiveEnabled = isChecked
        }

        // 8. Export Archives Button
        binding.btnExportArchives.setOnClickListener {
            val files = AutoDataArchiver.getArchivedFiles(this)
            if (files.isEmpty()) {
                showToast("No telemetry archives found yet. Incident telemetry will appear here.")
            } else {
                showToast("${files.size} archived telemetry session(s) saved in app storage.")
            }
        }

        // 9. Emergency Contacts in Settings
        binding.btnSettingsPickContact.setOnClickListener {
            pickContactWithPermission(null)
        }
        binding.btnSettingsAddContact.setOnClickListener {
            showAddContactDialog()
        }
        renderSettingsContacts()
    }

    private fun updateSelectedRingtoneLabel() {
        val uri = appPreferences.getResolvedRingtoneUri()
        val title = RingtoneManager.getRingtone(this, uri)?.getTitle(this) ?: "Emergency Alarm Siren"
        binding.tvSelectedRingtone.text = title
    }

    private fun playRingtonePreview() {
        try {
            if (previewRingtone?.isPlaying == true) {
                previewRingtone?.stop()
                binding.btnPreviewRingtone.text = "▶ Test"
                return
            }
            val uri = appPreferences.getResolvedRingtoneUri()
            previewRingtone = RingtoneManager.getRingtone(this, uri)
            previewRingtone?.play()
            binding.btnPreviewRingtone.text = "■ Stop"

            // Auto stop after 4 seconds
            Handler(Looper.getMainLooper()).postDelayed({
                if (previewRingtone?.isPlaying == true) {
                    previewRingtone?.stop()
                    binding.btnPreviewRingtone.text = "▶ Test"
                }
            }, 4000L)
        } catch (e: Exception) {
            showToast("Could not preview sound.")
        }
    }

    private fun updateBatteryOptimizationStatus() {
        val pm = getSystemService(POWER_SERVICE) as PowerManager
        val isIgnored = pm.isIgnoringBatteryOptimizations(packageName)
        if (isIgnored) {
            binding.tvSettingsBatteryAction.text = "✓ Active"
            binding.tvSettingsBatteryAction.setTextColor(ContextCompat.getColor(this, R.color.primary_emerald))
        } else {
            binding.tvSettingsBatteryAction.text = "Fix →"
            binding.tvSettingsBatteryAction.setTextColor(ContextCompat.getColor(this, R.color.btn_export))
        }
    }

    private fun renderSettingsContacts() {
        val contacts = contactManager.getContacts()
        binding.containerSettingsContacts.removeAllViews()
        binding.tvSettingsContactCount.text = "${contacts.size} contact(s)"

        if (contacts.isEmpty()) {
            val emptyTv = TextView(this).apply {
                text = "⚠️ No emergency contacts registered yet. Please add at least one contact."
                textSize = 13f
                setTextColor(ContextCompat.getColor(context, R.color.warning_amber))
                setPadding(0, 8, 0, 8)
            }
            binding.containerSettingsContacts.addView(emptyTv)
            return
        }

        val inflater = LayoutInflater.from(this)
        contacts.forEachIndexed { index, contact ->
            val itemView = inflater.inflate(R.layout.item_emergency_contact, binding.containerSettingsContacts, false)
            val tvName = itemView.findViewById<TextView>(R.id.tvContactName)
            val tvPhone = itemView.findViewById<TextView>(R.id.tvContactPhone)
            val tvBadge = itemView.findViewById<TextView>(R.id.tvContactBadge)
            val btnDelete = itemView.findViewById<ImageButton>(R.id.btnDeleteContact)

            tvName.text = contact.name
            tvPhone.text = contact.phone

            if (index == 0) {
                tvBadge.text = "★ Primary (Call + SMS)"
                tvBadge.setBackgroundColor(ContextCompat.getColor(this, R.color.primary_emerald))
            } else {
                tvBadge.text = "SMS"
                tvBadge.setBackgroundColor(ContextCompat.getColor(this, R.color.btn_export))
            }

            btnDelete.setOnClickListener {
                AlertDialog.Builder(this)
                    .setTitle("Remove Contact")
                    .setMessage("Remove ${contact.name} from emergency contacts?")
                    .setPositiveButton("Remove") { _, _ ->
                        contactManager.removeContact(contact.id)
                        renderSettingsContacts()
                        updateQuickContactCard()
                        showToast("${contact.name} removed.")
                    }
                    .setNegativeButton("Cancel", null)
                    .show()
            }

            binding.containerSettingsContacts.addView(itemView)
        }
    }

    private fun showAddContactDialog() {
        val dialogView = LayoutInflater.from(this).inflate(R.layout.dialog_add_contact, null)
        val etName = dialogView.findViewById<EditText>(R.id.etContactName)
        val etPhone = dialogView.findViewById<EditText>(R.id.etContactPhone)
        val btnPick = dialogView.findViewById<Button>(R.id.btnPickFromContacts)

        btnPick?.setOnClickListener {
            pickContactWithPermission { name, phone ->
                etName.setText(name)
                etPhone.setText(phone)
            }
        }

        AlertDialog.Builder(this)
            .setTitle("Add Emergency Contact")
            .setView(dialogView)
            .setPositiveButton("Save") { _, _ ->
                val name = etName.text.toString().trim()
                val phone = etPhone.text.toString().trim()
                if (name.isNotEmpty() && phone.isNotEmpty()) {
                    contactManager.addContact(name, phone)
                    renderSettingsContacts()
                    updateQuickContactCard()
                    checkAndRequestSosPermissions()
                    showToast("Emergency contact saved: $name")
                } else {
                    showToast("Please provide both name and phone number.")
                }
            }
            .setNegativeButton("Cancel", null)
            .show()
    }

    private fun setupFallAlertDialog() {
        fallAlertDialog = FallAlertDialog(
            context = this,
            onCancelled = {
                monitoringService?.dismissFall()
                clearFallBanner()
            },
            onSosNow = {
                monitoringService?.executeEmergencyProtocol()
            }
        )
    }

    // ── Monitoring Lifecycle ──────────────────────────────────────────────────

    private fun handleStartMonitoring() {
        val permissionsNeeded = mutableListOf<String>()

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            if (ContextCompat.checkSelfPermission(this, Manifest.permission.POST_NOTIFICATIONS) != PackageManager.PERMISSION_GRANTED) {
                permissionsNeeded.add(Manifest.permission.POST_NOTIFICATIONS)
            }
        }
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.SEND_SMS) != PackageManager.PERMISSION_GRANTED) {
            permissionsNeeded.add(Manifest.permission.SEND_SMS)
        }
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CALL_PHONE) != PackageManager.PERMISSION_GRANTED) {
            permissionsNeeded.add(Manifest.permission.CALL_PHONE)
        }
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.ACCESS_FINE_LOCATION) != PackageManager.PERMISSION_GRANTED) {
            permissionsNeeded.add(Manifest.permission.ACCESS_FINE_LOCATION)
        }

        if (permissionsNeeded.isNotEmpty()) {
            ActivityCompat.requestPermissions(this, permissionsNeeded.toTypedArray(), permissionRequestCode)
            return
        }

        startMonitoringSession()
    }

    private fun startMonitoringSession() {
        val startIntent = Intent(this, FallMonitoringService::class.java).apply {
            action = FallMonitoringService.ACTION_START_MONITORING
        }
        ContextCompat.startForegroundService(this, startIntent)
        monitoringService?.startMonitoring()
        monitoringService?.startRecording("standing", 50)
        updateUiMonitoringState(true)
        showToast("FallGaurd AI protection active.")
    }

    private fun handleStopMonitoring() {
        monitoringService?.stopMonitoring()
        updateUiMonitoringState(false)
        showToast("Fall protection paused.")
    }

    private fun checkAndRequestSosPermissions() {
        val sosPerms = mutableListOf<String>()
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.SEND_SMS) != PackageManager.PERMISSION_GRANTED) {
            sosPerms.add(Manifest.permission.SEND_SMS)
        }
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CALL_PHONE) != PackageManager.PERMISSION_GRANTED) {
            sosPerms.add(Manifest.permission.CALL_PHONE)
        }
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.ACCESS_FINE_LOCATION) != PackageManager.PERMISSION_GRANTED) {
            sosPerms.add(Manifest.permission.ACCESS_FINE_LOCATION)
        }
        if (sosPerms.isNotEmpty()) {
            ActivityCompat.requestPermissions(this, sosPerms.toTypedArray(), permissionRequestCode)
        }
    }

    // ── Service Listener Callbacks ────────────────────────────────────────────

    override fun onInferenceUpdate(label: String, confidence: Float, probs: FloatArray) {
        val pct = (confidence * 100).toInt()
        val icon = when (label.lowercase()) {
            "walking"  -> "🚶"
            "running"  -> "🏃"
            "sitting"  -> "🪑"
            "lying"    -> "🛌"
            "standing" -> "🧍"
            "falling"  -> "⚠️"
            else       -> "⚡"
        }
        binding.tvActivityIcon.text = icon
        binding.tvActivityLabel.text = label.uppercase()
        binding.tvConfidencePill.text = "$pct% Confident"
        binding.progressBarConfidence.progress = pct
    }

    override fun onFallPhaseChanged(phase: FallDetector.Phase, description: String) {
        val text = when (phase) {
            FallDetector.Phase.IDLE                   -> "Normal motion • Posture stable"
            FallDetector.Phase.PHASE1_SUDDEN_MOVEMENT -> "⚡ Sudden movement detected"
            FallDetector.Phase.PHASE2_ROTATION        -> "🔄 Rotation / posture change"
            FallDetector.Phase.PHASE3_IMPACT          -> "💥 Impact confirmed by model"
            FallDetector.Phase.CONFIRMED              -> "⚠️ Fall Confirmed!"
        }
        binding.tvFallStatus.text = text
    }

    override fun onFallConfirmed() {
        showFallAlert()
    }

    override fun onFallDismissed() {
        clearFallBanner()
    }

    override fun onCountdownTick(secondsRemaining: Int) {
        fallAlertDialog.updateCountdown(secondsRemaining)
    }

    override fun onSosDispatched(summary: String) {
        fallAlertDialog.showEscalatedState(summary)
        showToast(summary)
        loadHistory()
    }

    override fun onBatteryModeChanged(isLowBattery: Boolean) {
        updateLowBatteryIndicator(isLowBattery)
    }

    override fun onSampleCountChanged(count: Int, durationSeconds: Long) {
        val durStr = String.format("%02d:%02d", durationSeconds / 60, durationSeconds % 60)
        binding.tvSessionStats.text = "Session: $durStr • $count telemetry samples"
    }

    override fun onSensorMagnitudeUpdate(currentAcc: Float) {
        if (binding.viewSensitivity.visibility != View.VISIBLE) return

        binding.tvLiveCurrentAcc.text = "${"%.1f".format(currentAcc)} m/s²"
        if (currentAcc > livePeakAcc) {
            livePeakAcc = currentAcc
            binding.tvLivePeakAcc.text = "${"%.1f".format(livePeakAcc)} m/s²"
        }

        val currentInt = currentAcc.toInt().coerceIn(0, 40)
        binding.progressLiveAcc.progress = currentInt

        val thresh = when (appPreferences.sensitivityPreset) {
            "desk_safe" -> 26.0f
            "balanced"  -> 22.0f
            "high"      -> 18.0f
            "custom"    -> appPreferences.customImpactThreshold
            else        -> 26.0f
        }

        if (livePeakAcc >= thresh) {
            binding.tvLiveCalibratorResult.text = "⚠️ HIGH IMPACT • Exceeds threshold (${"%.1f".format(livePeakAcc)} m/s²)"
            binding.tvLiveCalibratorResult.setTextColor(ContextCompat.getColor(this, R.color.warning_amber))
            binding.tvLiveCalibratorResult.backgroundTintList = ContextCompat.getColorStateList(this, R.color.warning_amber_light)
        } else {
            binding.tvLiveCalibratorResult.text = "✅ SAFE • Desk placement (${"%.1f".format(livePeakAcc)} m/s²) will not trigger"
            binding.tvLiveCalibratorResult.setTextColor(ContextCompat.getColor(this, R.color.primary_emerald))
            binding.tvLiveCalibratorResult.backgroundTintList = ContextCompat.getColorStateList(this, R.color.primary_emerald_light)
        }
    }

    // ── Alert UI ──────────────────────────────────────────────────────────────

    private fun showFallAlert() {
        binding.tvFallBanner.visibility = View.VISIBLE
        if (!fallAlertDialog.isShowing) {
            val sec = monitoringService?.alertSecondsRemaining ?: appPreferences.countdownSeconds
            fallAlertDialog.show(sec)
        }
    }

    private fun clearFallBanner() {
        binding.tvFallBanner.visibility = View.GONE
        binding.tvFallStatus.text = "Normal motion • Posture stable"
        fallAlertDialog.dismiss()
    }

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<out String>,
        grantResults: IntArray,
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == permissionRequestCode) {
            startMonitoringSession()
        }
    }

    override fun onStart() {
        super.onStart()
        monitoringService?.registerListener(this)
    }

    override fun onStop() {
        super.onStop()
        previewRingtone?.stop()
        monitoringService?.unregisterListener()
    }

    override fun onDestroy() {
        super.onDestroy()
        if (isServiceBound) {
            unbindService(serviceConnection)
            isServiceBound = false
        }
        fallAlertDialog.dismiss()
    }

    private fun showToast(msg: String) =
        Toast.makeText(this, msg, Toast.LENGTH_LONG).show()
}
