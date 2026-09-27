package com.example.activitycollector

import android.app.Dialog
import android.content.Context
import android.graphics.Color
import android.graphics.drawable.ColorDrawable
import android.os.Handler
import android.os.Looper
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
 * - 30-second countdown with auto-escalation
 * - "I am okay" and "Cancel alert" dismissal buttons
 * - Post-timeout simulated emergency escalation display
 * - Full-screen layout with wake-up and show-when-locked flags
 */
class FallAlertDialog(
    private val context: Context,
    private val onCancelled: () -> Unit,
    private val onTimeout:   () -> Unit = {},
) {

    companion object {
        private const val AUTO_DISMISS_SECONDS = 30
    }

    private val dialog = Dialog(context, android.R.style.Theme_Black_NoTitleBar_Fullscreen)
    private val handler = Handler(Looper.getMainLooper())
    private var secondsLeft = AUTO_DISMISS_SECONDS

    private var tvTitle: TextView? = null
    private var tvBody: TextView? = null
    private var tvCountdown: TextView? = null
    private var isTimedOut = false

    private val countdownRunnable = object : Runnable {
        override fun run() {
            secondsLeft--
            tvCountdown?.text = context.getString(R.string.fall_alert_countdown, secondsLeft)

            if (secondsLeft <= 0) {
                isTimedOut = true
                showEscalatedState()
                onTimeout()
            } else {
                handler.postDelayed(this, 1_000)
            }
        }
    }

    fun show() {
        if (dialog.isShowing) return

        isTimedOut = false
        secondsLeft = AUTO_DISMISS_SECONDS

        val view = buildView()
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

        handler.postDelayed(countdownRunnable, 1_000)
    }

    fun dismiss() {
        handler.removeCallbacks(countdownRunnable)
        if (dialog.isShowing) dialog.dismiss()
    }

    val isShowing: Boolean get() = dialog.isShowing

    private fun showEscalatedState() {
        tvTitle?.text = "EMERGENCY PROTOCOL ACTIVATED (SIMULATED)"
        tvBody?.text = "30-second timer elapsed without response.\n\nIn live production, an automated SMS/call with GPS location would be dispatched to registered caregivers and emergency contacts.\n\n(No actual emergency services contacted)."
        tvCountdown?.text = "🚨 SIMULATED RESCUE SIGNAL SENT"
        tvCountdown?.setTextColor(0xFFFFDD44.toInt())
    }

    private fun buildView(): View {
        val root = android.widget.LinearLayout(context).apply {
            orientation = android.widget.LinearLayout.VERTICAL
            gravity     = android.view.Gravity.CENTER
            setBackgroundColor(ContextCompat.getColor(context, R.color.fall_alert_bg))
            setPadding(48, 80, 48, 80)
        }

        // ── Warning Icon ──────────────────────────────────────────────────────
        TextView(context).apply {
            text     = "⚠️"
            textSize = 72f
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
            setPadding(0, 24, 0, 16)
            root.addView(this)
        }

        // ── Body ──────────────────────────────────────────────────────────────
        tvBody = TextView(context).apply {
            text      = context.getString(R.string.fall_alert_body)
            textSize  = 16f
            setTextColor(0xEEFFFFFF.toInt())
            gravity   = android.view.Gravity.CENTER
            setPadding(0, 0, 0, 32)
            root.addView(this)
        }

        // ── Countdown ─────────────────────────────────────────────────────────
        tvCountdown = TextView(context).apply {
            text      = context.getString(R.string.fall_alert_countdown, AUTO_DISMISS_SECONDS)
            textSize  = 16f
            typeface  = android.graphics.Typeface.DEFAULT_BOLD
            setTextColor(0xFFFFFFFF.toInt())
            gravity   = android.view.Gravity.CENTER
            setPadding(0, 0, 0, 48)
            root.addView(this)
        }

        // ── Buttons ───────────────────────────────────────────────────────────
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
                marginEnd = 16
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
            setTextColor(ContextCompat.getColor(context, R.color.fall_alert_bg))
            setBackgroundColor(0xFF22C55E.toInt())
            textSize = 15f
            val lp = android.widget.LinearLayout.LayoutParams(0,
                android.widget.LinearLayout.LayoutParams.WRAP_CONTENT, 1f).apply {
                marginStart = 16
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
