package com.example.activitycollector

import android.util.Log
import kotlin.math.acos
import kotlin.math.sqrt

/**
 * FallDetector
 * ============
 * A temporal state machine that confirms a genuine human fall using multi-window
 * temporal fusion between:
 *   1. Real-time physical sensor metrics (free-fall dip, impact spike, angular velocity)
 *   2. On-device PyTorch Mobile HAR classification (prob(falling) >= threshold)
 *   3. Post-impact posture assessment (orientation tilt change and/or "lying" label)
 *   4. Post-fall stillness verification (low SVM variance across consecutive windows)
 *
 * Phone-Drop & False-Trigger Defense
 * ----------------------------------
 * • Does NOT trigger on mere acceleration spikes: the ML model MUST corroborate
 *   the physiological dynamics of a fall (not a mechanical impulse).
 * • If the user resumes active motion ("walking", "running") in the post-impact
 *   window, the event is immediately dismissed as a recovery or stumble.
 * • Requires multi-window temporal confirmation (at least 2 consecutive windows)
 *   before escalating to an alert.
 */
class FallDetector(
    private val onFallConfirmed: () -> Unit,
    private val onPhaseChanged:  (phase: Phase, description: String) -> Unit = { _, _ -> },
) {

    companion object {
        private const val TAG = "FallDetector"

        // Physical thresholds (defaults)
        const val SVM_SPIKE_THRESHOLD     = 24.0f   // m/s² (impact peak acceleration)
        const val SVM_FREEFALL_THRESHOLD  = 6.5f    // m/s² (free-fall weightlessness dip)
        const val GYR_SPIKE_THRESHOLD     = 1.8f    // rad/s (rotation during body topple)
        const val STILLNESS_STD_THRESHOLD = 1.2f    // m/s² (max std-dev of SVM during post-fall rest)
        const val MIN_TILT_CHANGE_DEG     = 28.0f   // degrees (posture change between pre-fall & rest)
        const val FALL_MODEL_CONF_THRESH  = 0.35f   // minimum model probability for 'falling'

        const val PHASE_TIMEOUT_MS        = 2_500L  // max gap between phases 1-3
        const val POST_FALL_TIMEOUT_MS    = 4_500L  // duration to observe post-fall state
        const val REQUIRED_STILL_WINDOWS  = 2       // consecutive windows required to confirm
    }

    enum class Phase {
        IDLE,
        PHASE1_SUDDEN_MOVEMENT,
        PHASE2_ROTATION,
        PHASE3_IMPACT,
        CONFIRMED
    }

    @Volatile var currentPhase: Phase = Phase.IDLE
        private set

    // Configurable thresholds for sensitivity tuning & desk placement defense
    var svmSpikeThreshold: Float = 26.0f
    var svmFreefallThreshold: Float = 5.5f
    var requireFreefallDip: Boolean = true // When true, filters out placing phone on desk
    var gyrSpikeThreshold: Float = GYR_SPIKE_THRESHOLD
    var stillnessStdThreshold: Float = STILLNESS_STD_THRESHOLD
    var minTiltChangeDeg: Float = MIN_TILT_CHANGE_DEG
    var fallModelConfThresh: Float = FALL_MODEL_CONF_THRESH
    var requiredStillWindows: Int = 3

    private var phaseEnteredAt = 0L
    private var confirmedWindows = 0
    private var preImpactGravity = floatArrayOf(0f, 9.8f, 0f)  // rolling baseline
    private var lastAccSvmPeak = 0f

    fun applySensitivity(
        preset: String,
        customSpike: Float = 24.0f,
        customFreefall: Boolean = true,
        customStillSec: Float = 1.5f
    ) {
        when (preset) {
            "desk_safe" -> {
                // Calibrated to IGNORE desk placement / bed toss
                // Placing phone down on desk produces ~12-18 m/s² spike with ZERO freefall dip
                svmSpikeThreshold = 26.0f
                svmFreefallThreshold = 5.5f
                requireFreefallDip = true
                fallModelConfThresh = 0.45f
                minTiltChangeDeg = 32.0f
                requiredStillWindows = 3 // 1.2s stillness
            }
            "high" -> {
                // High vigilance for elderly
                svmSpikeThreshold = 18.0f
                svmFreefallThreshold = 7.5f
                requireFreefallDip = false
                fallModelConfThresh = 0.30f
                minTiltChangeDeg = 20.0f
                requiredStillWindows = 2 // 800ms
            }
            "custom" -> {
                svmSpikeThreshold = customSpike
                requireFreefallDip = customFreefall
                requiredStillWindows = (customStillSec / 0.4f).toInt().coerceAtLeast(1)
                fallModelConfThresh = 0.40f
                minTiltChangeDeg = 28.0f
            }
            else -> { // "balanced"
                svmSpikeThreshold = 22.0f
                svmFreefallThreshold = 6.5f
                requireFreefallDip = false
                fallModelConfThresh = 0.35f
                minTiltChangeDeg = 28.0f
                requiredStillWindows = 2
            }
        }
        Log.i(TAG, "Applied sensitivity '$preset': spike=$svmSpikeThreshold, reqFreefall=$requireFreefallDip, stillWindows=$requiredStillWindows")
    }

    /**
     * Process one inference result and corresponding sensor window.
     */
    @Synchronized
    fun onInferenceResult(
        label: String,
        probs: FloatArray,
        window2D: Array<FloatArray>,
    ) {
        val now = System.currentTimeMillis()
        val accPeak = computeAccSvmPeak(window2D)
        val accMin  = computeAccSvmMin(window2D)
        val gyrPeak = computeGyrSvmPeak(window2D)
        val accStd  = computeAccSvmStd(window2D)
        val currentOrientation = computeOrientationVector(window2D)

        lastAccSvmPeak = accPeak

        // Class index 0 is 'falling', class index 1 is 'lying'
        val fallProb = if (probs.isNotEmpty()) probs[0] else 0f
        val isFallPredicted = label.equals("falling", ignoreCase = true) || fallProb >= fallModelConfThresh
        val isLyingPredicted = label.equals("lying", ignoreCase = true) || (probs.size > 1 && probs[1] >= 0.40f)
        val isActiveMotion = label.equals("walking", ignoreCase = true) || label.equals("running", ignoreCase = true)

        val freefallDetected = accMin < svmFreefallThreshold
        val rotationDetected = gyrPeak > gyrSpikeThreshold

        when (currentPhase) {
            Phase.IDLE -> {
                // Update pre-fall baseline orientation during stationary/normal activity
                if (!isFallPredicted && accStd < 2.0f) {
                    preImpactGravity = currentOrientation
                }

                // Initial fall initiation check:
                // When requireFreefallDip is enabled (Desk-Safe mode), require weightlessness dip before impact.
                // Placing a phone on a desk produces deceleration only, NEVER free-fall dip.
                val isInitiated = if (requireFreefallDip) {
                    (freefallDetected && accPeak > 18.0f) || (accPeak > svmSpikeThreshold && freefallDetected)
                } else {
                    accPeak > svmSpikeThreshold || (freefallDetected && gyrPeak > 1.2f)
                }

                if (isInitiated) {
                    enterPhase(
                        Phase.PHASE1_SUDDEN_MOVEMENT,
                        now,
                        "Acc spike = ${"%.1f".format(accPeak)} m/s² (dip: ${"%.1f".format(accMin)}, reqDip=$requireFreefallDip)"
                    )
                }
            }

            Phase.PHASE1_SUDDEN_MOVEMENT -> {
                if (timedOut(now, PHASE_TIMEOUT_MS)) {
                    reset("Phase 1 timed out (no rotation)")
                    return
                }
                if (rotationDetected) {
                    enterPhase(
                        Phase.PHASE2_ROTATION,
                        now,
                        "Gyro rotation = ${"%.2f".format(gyrPeak)} rad/s"
                    )
                } else if (accPeak > svmSpikeThreshold) {
                    phaseEnteredAt = now
                }
            }

            Phase.PHASE2_ROTATION -> {
                if (timedOut(now, PHASE_TIMEOUT_MS)) {
                    reset("Phase 2 timed out (no impact/model confirmation)")
                    return
                }

                // Must have ML model confirmation of 'falling' AND physical impact spike
                if (isFallPredicted && accPeak > (svmSpikeThreshold * 0.75f)) {
                    enterPhase(
                        Phase.PHASE3_IMPACT,
                        now,
                        "Fall impact confirmed by model (P(fall)=${"%.0f".format(fallProb * 100)}%, peak=${"%.1f".format(accPeak)})"
                    )
                }
            }

            Phase.PHASE3_IMPACT -> {
                if (timedOut(now, POST_FALL_TIMEOUT_MS)) {
                    reset("Phase 3 timed out without sustained lying/stillness")
                    return
                }

                // If user immediately gets up and continues walking/running, reject false alarm
                if (isActiveMotion && accStd > 2.0f) {
                    reset("User active ($label) — recovered from stumble/drop")
                    return
                }

                // Calculate tilt angle change from pre-impact baseline
                val tiltAngle = computeTiltAngleDeg(preImpactGravity, currentOrientation)

                // Condition for post-fall state:
                // 1. Model predicts 'lying' OR
                // 2. Post-fall stillness (low acc variance) with a significant orientation tilt change
                val isPostFallRest = isLyingPredicted || (accStd < stillnessStdThreshold && tiltAngle >= minTiltChangeDeg)

                if (isPostFallRest) {
                    confirmedWindows++
                    Log.d(TAG, "Post-fall window $confirmedWindows / $requiredStillWindows (tilt: ${"%.1f".format(tiltAngle)}°, std: ${"%.2f".format(accStd)}, label: $label)")

                    if (confirmedWindows >= requiredStillWindows) {
                        enterPhase(Phase.CONFIRMED, now, "Fall confirmed: model verified + sustained post-fall rest")
                        onFallConfirmed()
                    }
                } else {
                    if (accStd > 2.0f) {
                        confirmedWindows = 0
                    }
                }
            }

            Phase.CONFIRMED -> {
                // Awaiting user response via dismissFall()
            }
        }
    }

    /** Reset state when user cancels alert or taps "I am okay". */
    @Synchronized
    fun dismissFall() {
        reset("User dismissed fall alert")
    }

    /** Reset detector state. */
    @Synchronized
    fun reset(reason: String = "manual reset") {
        Log.d(TAG, "Reset: $reason (was $currentPhase)")
        currentPhase     = Phase.IDLE
        phaseEnteredAt   = 0L
        confirmedWindows = 0
    }

    private fun enterPhase(phase: Phase, now: Long, reason: String) {
        Log.i(TAG, "→ $phase [$reason]")
        currentPhase     = phase
        phaseEnteredAt   = now
        confirmedWindows = 0
        onPhaseChanged(phase, reason)
    }

    private fun timedOut(now: Long, limitMs: Long): Boolean =
        phaseEnteredAt > 0L && (now - phaseEnteredAt) > limitMs

    // ── Signal feature helpers ────────────────────────────────────────────────

    private fun computeAccSvmPeak(w: Array<FloatArray>): Float =
        w.maxOf { row -> sqrt(row[0]*row[0] + row[1]*row[1] + row[2]*row[2]) }

    private fun computeAccSvmMin(w: Array<FloatArray>): Float =
        w.minOf { row -> sqrt(row[0]*row[0] + row[1]*row[1] + row[2]*row[2]) }

    private fun computeGyrSvmPeak(w: Array<FloatArray>): Float =
        w.maxOf { row -> sqrt(row[3]*row[3] + row[4]*row[4] + row[5]*row[5]) }

    private fun computeAccSvmStd(w: Array<FloatArray>): Float {
        val svms = w.map { row -> sqrt(row[0]*row[0] + row[1]*row[1] + row[2]*row[2]) }
        val mean = svms.average().toFloat()
        var variance = 0.0
        for (v in svms) {
            val diff = v - mean
            variance += diff * diff
        }
        return sqrt((variance / svms.size).toFloat())
    }

    private fun computeOrientationVector(w: Array<FloatArray>): FloatArray {
        var sx = 0f; var sy = 0f; var sz = 0f
        for (row in w) {
            sx += row[0]; sy += row[1]; sz += row[2]
        }
        val n = w.size.toFloat()
        val mx = sx / n; val my = sy / n; val mz = sz / n
        val mag = sqrt(mx*mx + my*my + mz*mz)
        return if (mag > 0.001f) {
            floatArrayOf(mx / mag, my / mag, mz / mag)
        } else {
            floatArrayOf(0f, 1f, 0f)
        }
    }

    private fun computeTiltAngleDeg(u: FloatArray, v: FloatArray): Float {
        val dot = (u[0]*v[0] + u[1]*v[1] + u[2]*v[2]).coerceIn(-1.0f, 1.0f)
        return (acos(dot.toDouble()) * (180.0 / Math.PI)).toFloat()
    }
}
