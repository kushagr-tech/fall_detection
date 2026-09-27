package com.example.activitycollector

import android.app.Dialog
import android.content.Context
import android.graphics.Color
import android.graphics.drawable.ColorDrawable
import android.view.View
import android.view.ViewGroup
import android.view.WindowManager
import android.widget.Button
import android.widget.TextView
import androidx.core.content.ContextCompat

/**
 * FallAlertDialog
 * ===============
 * A full-screen emergency modal displayed when FallDetector confirms a fall.
 *
 * Features:
 * - Real-time countdown driven by FallMonitoringService
 * - Displays registered primary contact to be called and SMS recipients count
 * - "I am okay ✓" and "Cancel alert" dismissal buttons
 * - "🚨 Call SOS Now" immediate help button (bypasses countdown)
 * - Post-escalation status banner showing dispatch results
 */
class FallAlertDialog(
    private val context: Context,
    private val onCancelled: () -> Unit,
    private val onSosNow:    () -> Unit = {},
) {

    private val dialog = Dialog(context, android.R.style.Theme_Black_NoTitleBar_Fullscreen)

    private var tvTitle: TextView? = null
    private var tvBody: TextView? = null
    private var tvEmergencyDetails: TextView? = null
    private var tvCountdown: TextView? = null
    private var btnSosNow: Button? = null

    fun show(secondsRemaining: Int = 30) {
        if (dialog.isShowing) return

        val view = buildView(secondsRemaining)
        dialog.setContentView(view)
        dialog.window?.apply {
            setLayout(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.MATCH_PARENT)
            setBackgroundDrawable(ColorDrawable(Color.TRANSPARENT))
            addFlags(
                WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON or
                WindowManager.LayoutParams.FLAG_SHOW_WHEN_LOCKED or
                WindowManager.LayoutParams.FLAG_TURN_SCREEN_ON
            )
        }
        dialog.setCancelable(false)
        dialog.show()
    }

    fun dismiss() {
        if (dialog.isShowing) dialog.dismiss()
    }

    val isShowing: Boolean get() = dialog.isShowing

    fun updateCountdown(secondsLeft: Int) {
        if (!dialog.isShowing) return
        tvCountdown?.text = context.getString(R.string.fall_alert_countdown, secondsLeft)
    }

    fun showEscalatedState(summary: String) {
        if (!dialog.isShowing) return
        tvTitle?.text = "🚨 EMERGENCY SOS DISPATCHED"
        tvBody?.text = summary
        tvCountdown?.text = "Call placed & SMS sent with GPS coordinates."
        tvCountdown?.setTextColor(0xFFFFDD44.toInt())
        btnSosNow?.visibility = View.GONE
    }

    private fun buildView(initialSeconds: Int): View {
        val root = android.widget.LinearLayout(context).apply {
            orientation = android.widget.LinearLayout.VERTICAL
            gravity     = android.view.Gravity.CENTER
            setBackgroundColor(ContextCompat.getColor(context, R.color.fall_alert_bg))
            setPadding(48, 60, 48, 60)
        }

        // ── Warning Icon ──────────────────────────────────────────────────────
        TextView(context).apply {
            text     = "⚠️"
            textSize = 68f
            gravity  = android.view.Gravity.CENTER
            root.addView(this)
        }

        // ── Title ─────────────────────────────────────────────────────────────
        tvTitle = TextView(context).apply {
            text      = context.getString(R.string.fall_alert_title)
            textSize  = 26f
            setTextColor(Color.WHITE)
            typeface  = android.graphics.Typeface.DEFAULT_BOLD
            gravity   = android.view.Gravity.CENTER
            setPadding(0, 16, 0, 12)
            root.addView(this)
        }

        // ── Body ──────────────────────────────────────────────────────────────
        tvBody = TextView(context).apply {
            text      = context.getString(R.string.fall_alert_body)
            textSize  = 16f
            setTextColor(0xEEFFFFFF.toInt())
            gravity   = android.view.Gravity.CENTER
            setPadding(0, 0, 0, 16)
            root.addView(this)
        }

        // ── Contact Info Card ─────────────────────────────────────────────────
        val contactManager = EmergencyContactManager(context)
        val contacts = contactManager.getContacts()
        val primary = contacts.firstOrNull()

        tvEmergencyDetails = TextView(context).apply {
            val detailsText = if (primary != null) {
                "📞 Direct Call: ${primary.name} (${primary.phone})\n💬 Emergency SMS: ${contacts.size} contact(s) with GPS link"
            } else {
                "⚠️ No emergency contacts configured in app!\nPlease add contacts in settings."
            }
            text = detailsText
            textSize = 14f
            setTextColor(0xDDFFFFFF.toInt())
            gravity = android.view.Gravity.CENTER
            setPadding(24, 16, 24, 16)
            setBackgroundColor(0x33000000)
            root.addView(this)
        }

        // ── Countdown ─────────────────────────────────────────────────────────
        tvCountdown = TextView(context).apply {
            text      = context.getString(R.string.fall_alert_countdown, initialSeconds)
            textSize  = 18f
            typeface  = android.graphics.Typeface.DEFAULT_BOLD
            setTextColor(0xFFFFFFFF.toInt())
            gravity   = android.view.Gravity.CENTER
            setPadding(0, 24, 0, 24)
            root.addView(this)
        }

        // ── Immediate SOS Button ──────────────────────────────────────────────
        btnSosNow = Button(context).apply {
            text = "🚨 CALL HELP NOW (SOS)"
            setTextColor(Color.WHITE)
            setBackgroundColor(0xFF991B1B.toInt()) // darker red
            textSize = 15f
            typeface = android.graphics.Typeface.DEFAULT_BOLD
            val lp = android.widget.LinearLayout.LayoutParams(
                android.widget.LinearLayout.LayoutParams.MATCH_PARENT,
                android.widget.LinearLayout.LayoutParams.WRAP_CONTENT
            ).apply {
                bottomMargin = 16
            }
            layoutParams = lp
            setOnClickListener {
                onSosNow()
            }
            root.addView(this)
        }

        // ── Dismiss Buttons Row ───────────────────────────────────────────────
        val btnRow = android.widget.LinearLayout(context).apply {
            orientation = android.widget.LinearLayout.HORIZONTAL
            gravity     = android.view.Gravity.CENTER
            root.addView(this)
        }

        Button(context).apply {
            text    = context.getString(R.string.fall_btn_cancel)
            setTextColor(ContextCompat.getColor(context, R.color.fall_alert_bg))
            setBackgroundColor(0xFFFFFFFF.toInt())
            textSize = 15f
            val lp = android.widget.LinearLayout.LayoutParams(0,
                android.widget.LinearLayout.LayoutParams.WRAP_CONTENT, 1f).apply {
                marginEnd = 12
            }
            layoutParams = lp
            setOnClickListener {
                dismiss()
                onCancelled()
            }
            btnRow.addView(this)
        }

        Button(context).apply {
            text    = context.getString(R.string.fall_btn_okay)
            setTextColor(Color.WHITE)
            setBackgroundColor(0xFF16A34A.toInt()) // vibrant green
            textSize = 15f
            typeface = android.graphics.Typeface.DEFAULT_BOLD
            val lp = android.widget.LinearLayout.LayoutParams(0,
                android.widget.LinearLayout.LayoutParams.WRAP_CONTENT, 1f).apply {
                marginStart = 12
            }
            layoutParams = lp
            setOnClickListener {
                dismiss()
                onCancelled()
            }
            btnRow.addView(this)
        }

        return root
    }
}
