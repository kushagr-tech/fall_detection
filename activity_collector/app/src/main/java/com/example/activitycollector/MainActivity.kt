package com.example.activitycollector

import android.Manifest
import android.content.ComponentName
import android.content.Context
import android.content.Intent
import android.content.ServiceConnection
import android.content.pm.PackageManager
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
import android.widget.ArrayAdapter
import android.widget.EditText
import android.widget.ImageButton
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.example.activitycollector.databinding.ActivityMainBinding

/**
 * MainActivity
 * ============
 * Front-end dashboard for the Activity Collector and Fall Detection System.
 *
 * Connected to FallMonitoringService for continuous 24/7 background monitoring,
 * real-time inference feedback, sensor recording, emergency contact management,
 * and automated SOS dispatch.
 */
class MainActivity : AppCompatActivity(), FallMonitoringService.ServiceListener {

    private lateinit var binding: ActivityMainBinding
    private lateinit var contactManager: EmergencyContactManager

    // ── Service Connection ────────────────────────────────────────────────────
    private var monitoringService: FallMonitoringService? = null
    private var isServiceBound = false

    private val serviceConnection = object : ServiceConnection {
        override fun onServiceConnected(name: ComponentName?, service: IBinder?) {
            val binder = service as? FallMonitoringService.LocalBinder
            monitoringService = binder?.getService()
            isServiceBound = true
            monitoringService?.registerListener(this@MainActivity)

            // Sync UI to existing service state
            monitoringService?.let { s ->
                val active = s.isMonitoring.get()
                updateUiRecordingState(active)
                if (active && !s.isRecording.get()) {
                    val rateHz = binding.etSamplingRate.text.toString().toIntOrNull()?.coerceIn(1, 200) ?: 50
                    val label = binding.spinnerLabel.selectedItem?.toString() ?: "standing"
                    s.startRecording(label, rateHz)
                }
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

    // ── Local records cache for export after service stop ─────────────────────
    private val localRecords = mutableListOf<SensorDataRecord>()

    private val permissionRequestCode = 1001

    private val activityLabels = listOf(
        "standing", "walking", "running", "sitting", "lying", "falling"
    )

    private val labelColors = mapOf(
        "falling"  to R.color.label_falling,
        "walking"  to R.color.label_walking,
        "running"  to R.color.label_running,
        "standing" to R.color.label_standing,
        "sitting"  to R.color.label_sitting,
        "lying"    to R.color.label_lying,
    )

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        contactManager = EmergencyContactManager(this)

        configureLockScreenVisibility()
        setupLabelSpinner()
        setupSamplingRateInput()
        setupButtons()
        setupEmergencyContacts()
        setupBatteryOptimizationCard()
        setupFallAlertDialog()

        // Bind to background monitoring service if already running
        val serviceIntent = Intent(this, FallMonitoringService::class.java)
        bindService(serviceIntent, serviceConnection, Context.BIND_AUTO_CREATE)

        handleAlertIntent(intent)
    }

    override fun onResume() {
        super.onResume()
        updateBatteryOptimizationStatus()
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

    private fun setupLabelSpinner() {
        val adapter = ArrayAdapter(this, android.R.layout.simple_spinner_item, activityLabels)
        adapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item)
        binding.spinnerLabel.adapter = adapter
    }

    private fun setupSamplingRateInput() {
        binding.etSamplingRate.setText("50")
    }

    private fun setupButtons() {
        binding.btnStart.setOnClickListener  { handleStart() }
        binding.btnStop.setOnClickListener   { handleStop() }
        binding.btnExport.setOnClickListener { handleExport() }
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

    // ── Emergency Contacts Section ────────────────────────────────────────────

    private fun setupEmergencyContacts() {
        binding.btnAddContact.setOnClickListener {
            showAddContactDialog()
        }
        renderEmergencyContacts()
    }

    private fun renderEmergencyContacts() {
        val contacts = contactManager.getContacts()
        binding.contactsContainer.removeAllViews()

        if (contacts.isEmpty()) {
            binding.tvEmptyContacts.visibility = View.VISIBLE
            binding.tvContactCountBadge.text = "0 contacts"
            return
        }

        binding.tvEmptyContacts.visibility = View.GONE
        binding.tvContactCountBadge.text = "${contacts.size} contact(s)"

        val inflater = LayoutInflater.from(this)
        contacts.forEachIndexed { index, contact ->
            val itemView = inflater.inflate(R.layout.item_emergency_contact, binding.contactsContainer, false)
            val tvName = itemView.findViewById<TextView>(R.id.tvContactName)
            val tvPhone = itemView.findViewById<TextView>(R.id.tvContactPhone)
            val tvBadge = itemView.findViewById<TextView>(R.id.tvContactBadge)
            val btnDelete = itemView.findViewById<ImageButton>(R.id.btnDeleteContact)

            tvName.text = contact.name
            tvPhone.text = contact.phone

            if (index == 0) {
                tvBadge.text = "★ Primary (Call + SMS)"
                tvBadge.setBackgroundColor(ContextCompat.getColor(this, R.color.btn_start))
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
                        renderEmergencyContacts()
                        showToast("${contact.name} removed.")
                    }
                    .setNegativeButton("Cancel", null)
                    .show()
            }

            binding.contactsContainer.addView(itemView)
        }
    }

    private fun showAddContactDialog() {
        val dialogView = LayoutInflater.from(this).inflate(R.layout.dialog_add_contact, null)
        val etName = dialogView.findViewById<EditText>(R.id.etContactName)
        val etPhone = dialogView.findViewById<EditText>(R.id.etContactPhone)

        AlertDialog.Builder(this)
            .setTitle("Add Emergency Contact")
            .setView(dialogView)
            .setPositiveButton("Save") { _, _ ->
                val name = etName.text.toString().trim()
                val phone = etPhone.text.toString().trim()
                if (name.isNotEmpty() && phone.isNotEmpty()) {
                    contactManager.addContact(name, phone)
                    renderEmergencyContacts()
                    checkAndRequestSosPermissions()
                    showToast("Emergency contact saved: $name")
                } else {
                    showToast("Please provide both name and phone number.")
                }
            }
            .setNegativeButton("Cancel", null)
            .show()
    }

    // ── Battery Optimization Card (StepSetGo style) ───────────────────────────

    private fun setupBatteryOptimizationCard() {
        binding.cardBatteryOptimization.setOnClickListener {
            val pm = getSystemService(POWER_SERVICE) as PowerManager
            if (!pm.isIgnoringBatteryOptimizations(packageName)) {
                try {
                    val intent = Intent(Settings.ACTION_REQUEST_IGNORE_BATTERY_OPTIMIZATIONS).apply {
                        data = Uri.parse("package:$packageName")
                    }
                    startActivity(intent)
                } catch (e: Exception) {
                    val intent = Intent(Settings.ACTION_IGNORE_BATTERY_OPTIMIZATION_SETTINGS)
                    startActivity(intent)
                }
            } else {
                showToast("Background battery optimization is already unrestricted. 24/7 protection is active!")
            }
        }
        updateBatteryOptimizationStatus()
    }

    private fun updateBatteryOptimizationStatus() {
        val pm = getSystemService(POWER_SERVICE) as PowerManager
        val isIgnored = pm.isIgnoringBatteryOptimizations(packageName)
        if (isIgnored) {
            binding.tvBatteryIcon.text = "🛡️"
            binding.tvBatteryTitle.text = "24/7 Background Protection Active"
            binding.tvBatterySubtitle.text = "Unrestricted execution enabled (survives app kill & sleep)"
            binding.tvBatteryAction.text = "✓ Active"
            binding.tvBatteryAction.setTextColor(ContextCompat.getColor(this, R.color.btn_start))
        } else {
            binding.tvBatteryIcon.text = "⚡"
            binding.tvBatteryTitle.text = "24/7 Background Reliability"
            binding.tvBatterySubtitle.text = "Tap to enable unrestricted execution (like StepSetGo)"
            binding.tvBatteryAction.text = "Fix →"
            binding.tvBatteryAction.setTextColor(ContextCompat.getColor(this, R.color.btn_export))
        }
    }

    // ── User Actions ──────────────────────────────────────────────────────────

    private fun handleStart() {
        val permissionsNeeded = mutableListOf<String>()

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            if (ContextCompat.checkSelfPermission(this, Manifest.permission.POST_NOTIFICATIONS)
                != PackageManager.PERMISSION_GRANTED) {
                permissionsNeeded.add(Manifest.permission.POST_NOTIFICATIONS)
            }
        }
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.SEND_SMS)
            != PackageManager.PERMISSION_GRANTED) {
            permissionsNeeded.add(Manifest.permission.SEND_SMS)
        }
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CALL_PHONE)
            != PackageManager.PERMISSION_GRANTED) {
            permissionsNeeded.add(Manifest.permission.CALL_PHONE)
        }
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.ACCESS_FINE_LOCATION)
            != PackageManager.PERMISSION_GRANTED) {
            permissionsNeeded.add(Manifest.permission.ACCESS_FINE_LOCATION)
        }
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.Q) {
            if (ContextCompat.checkSelfPermission(this, Manifest.permission.WRITE_EXTERNAL_STORAGE)
                != PackageManager.PERMISSION_GRANTED) {
                permissionsNeeded.add(Manifest.permission.WRITE_EXTERNAL_STORAGE)
            }
        }

        if (permissionsNeeded.isNotEmpty()) {
            ActivityCompat.requestPermissions(
                this,
                permissionsNeeded.toTypedArray(),
                permissionRequestCode
            )
            return
        }

        startMonitoringSession()
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

    private fun startMonitoringSession() {
        val startIntent = Intent(this, FallMonitoringService::class.java).apply {
            action = FallMonitoringService.ACTION_START_MONITORING
        }
        ContextCompat.startForegroundService(this, startIntent)

        val rateHz = binding.etSamplingRate.text.toString().toIntOrNull()?.coerceIn(1, 200) ?: 50
        val label = binding.spinnerLabel.selectedItem.toString()

        monitoringService?.startMonitoring()
        monitoringService?.startRecording(label, rateHz)

        updateUiRecordingState(true)
        showToast("Fall monitoring started in background.")
    }

    private fun handleStop() {
        monitoringService?.let { s ->
            val collected = s.stopRecording()
            synchronized(localRecords) {
                localRecords.clear()
                localRecords.addAll(collected)
            }
            s.stopMonitoring()
        }

        updateUiRecordingState(false)
        val count = localRecords.size
        showToast("Monitoring stopped. $count samples collected.")
    }

    private fun handleExport() {
        val activeRecords = if (localRecords.isNotEmpty()) {
            localRecords.toList()
        } else {
            monitoringService?.records?.toList() ?: emptyList()
        }
        if (activeRecords.isEmpty()) {
            showToast("No data to export.")
            return
        }
        val path = CsvExporter.export(this, activeRecords)
        if (path != null) {
            showToast("Saved: ${path.substringAfterLast('/')}")
        } else {
            showToast("Export failed — check storage permissions.")
        }
    }

    // ── Service Listener Callbacks ────────────────────────────────────────────

    override fun onInferenceUpdate(label: String, confidence: Float, probs: FloatArray) {
        val pct = (confidence * 100).toInt()
        val color = labelColors[label]?.let { ContextCompat.getColor(this, it) }
            ?: ContextCompat.getColor(this, R.color.text_primary)

        binding.tvActivityLabel.text = label.uppercase()
        binding.tvActivityLabel.setTextColor(color)
        binding.tvConfidence.text = getString(R.string.confidence_format, pct)
        binding.tvConfidenceBar.progress = pct
    }

    override fun onFallPhaseChanged(phase: FallDetector.Phase, description: String) {
        val (text, colorRes) = when (phase) {
            FallDetector.Phase.IDLE                   -> "" to R.color.recording_idle
            FallDetector.Phase.PHASE1_SUDDEN_MOVEMENT -> "⚡ Sudden movement" to R.color.phase_yellow
            FallDetector.Phase.PHASE2_ROTATION        -> "🔄 Rotation / posture change" to R.color.phase_orange
            FallDetector.Phase.PHASE3_IMPACT          -> "💥 Impact confirmed by model" to R.color.phase_red
            FallDetector.Phase.CONFIRMED              -> "⚠️ Fall Confirmed!" to R.color.fall_alert_bg
        }
        binding.tvFallStatus.text = text
        binding.tvFallStatus.setTextColor(ContextCompat.getColor(this, colorRes))
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
    }

    override fun onSampleCountChanged(count: Int, durationSeconds: Long) {
        binding.tvSampleCount.text = getString(R.string.sample_count_format, count)
        binding.tvDuration.text = getString(R.string.duration_format, durationSeconds / 60, durationSeconds % 60)
    }

    // ── Alert UI ──────────────────────────────────────────────────────────────

    private fun showFallAlert() {
        binding.tvFallBanner.visibility = View.VISIBLE
        if (!fallAlertDialog.isShowing) {
            val s = monitoringService
            val sec = s?.alertSecondsRemaining ?: 30
            fallAlertDialog.show(sec)
        }
    }

    private fun clearFallBanner() {
        binding.tvFallBanner.visibility = View.GONE
        binding.tvFallStatus.text = ""
        fallAlertDialog.dismiss()
    }

    private fun updateUiRecordingState(isActive: Boolean) {
        binding.btnStart.isEnabled       = !isActive
        binding.btnStop.isEnabled        = isActive
        binding.btnExport.isEnabled      = !isActive
        binding.spinnerLabel.isEnabled   = !isActive
        binding.etSamplingRate.isEnabled = !isActive
        binding.statusIndicator.setBackgroundColor(
            ContextCompat.getColor(this,
                if (isActive) R.color.recording_active else R.color.recording_idle)
        )
        if (!isActive) {
            binding.tvDuration.text    = getString(R.string.duration_default)
            binding.tvSampleCount.text = getString(R.string.sample_count_format, localRecords.size)
            binding.tvActivityLabel.text = getString(R.string.activity_idle)
            binding.tvConfidence.text  = ""
            binding.tvConfidenceBar.progress = 0
            binding.tvFallStatus.text  = ""
        }
    }

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<out String>,
        grantResults: IntArray,
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == permissionRequestCode) {
            val allGranted = grantResults.isNotEmpty() && grantResults.all { it == PackageManager.PERMISSION_GRANTED }
            if (allGranted) {
                startMonitoringSession()
            } else {
                showToast("Some permissions not granted. Emergency SOS features may be limited.")
                startMonitoringSession()
            }
        }
    }

    override fun onStart() {
        super.onStart()
        monitoringService?.registerListener(this)
    }

    override fun onStop() {
        super.onStop()
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
