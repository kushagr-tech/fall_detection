package com.example.activitycollector

import java.util.concurrent.atomic.AtomicInteger

/**
 * SensorBuffer
 * ============
 * A fixed-capacity circular buffer that holds raw sensor samples as a flat
 * float array.  Each slot stores 6 floats: [ax, ay, az, gx, gy, gz].
 *
 * Thread-safety
 * -------------
 * Writes happen on the sensor-callback thread (SensorEventListener).
 * Reads happen on the inference thread.  Because the write pointer advances
 * atomically and the buffer is pre-allocated, no lock is needed for typical
 * producer-single-consumer usage.  The [copyWindow] snapshot call is
 * synchronised only when a full window snapshot is needed.
 *
 * Layout in [data]
 * ----------------
 * Slot i occupies indices [i*6 .. i*6+5].
 */
class SensorBuffer(val capacity: Int) {

    companion object {
        const val CHANNELS = 6  // ax, ay, az, gx, gy, gz
    }

    private val data    = FloatArray(capacity * CHANNELS)
    private val writeAt = AtomicInteger(0)    // next write slot (mod capacity)
    private val total   = AtomicInteger(0)    // total samples written ever

    // ── Latest gyroscope carry-forward ────────────────────────────────────────
    // Gyroscope events arrive slightly less frequently than accelerometer.
    // We carry the last known gyro values and attach them to every acc sample.
    @Volatile var latestGx = 0f
    @Volatile var latestGy = 0f
    @Volatile var latestGz = 0f

    // ── Write ─────────────────────────────────────────────────────────────────

    /**
     * Push one fused sample into the ring buffer.
     * Called from the accelerometer callback, which already holds the latest
     * gyroscope values in [latestGx/y/z].
     */
    fun push(ax: Float, ay: Float, az: Float) {
        val slot  = writeAt.getAndUpdate { (it + 1) % capacity }
        val base  = slot * CHANNELS
        data[base    ] = ax
        data[base + 1] = ay
        data[base + 2] = az
        data[base + 3] = latestGx
        data[base + 4] = latestGy
        data[base + 5] = latestGz
        total.incrementAndGet()
    }

    // ── Read ──────────────────────────────────────────────────────────────────

    /** True once the buffer has been filled at least once end-to-end. */
    val isFull: Boolean get() = total.get() >= capacity

    /** Number of samples written so far (saturates at Int.MAX_VALUE). */
    val size: Int get() = minOf(total.get(), capacity)

    /**
     * Copy the most recent [windowSize] samples into a flat [FloatArray] of
     * length (windowSize × 6), ordered oldest → newest.
     *
     * Returns null if fewer than [windowSize] samples have been collected yet.
     */
    fun copyWindow(windowSize: Int): FloatArray? {
        if (total.get() < windowSize) return null

        val out  = FloatArray(windowSize * CHANNELS)
        val head = writeAt.get()   // next-write slot  = oldest slot in ring

        // The oldest of the last [windowSize] samples starts at:
        //   (head - windowSize + capacity) % capacity
        // We copy slot-by-slot in chronological order.
        val startSlot = ((head - windowSize) % capacity + capacity) % capacity

        for (i in 0 until windowSize) {
            val slot = (startSlot + i) % capacity
            val src  = slot * CHANNELS
            val dst  = i    * CHANNELS
            data.copyInto(out, dst, src, src + CHANNELS)
        }
        return out
    }

    /**
     * Return a snapshot as a 2-D array [windowSize][CHANNELS] for callers
     * that prefer row-major indexing (e.g. the FallDetector).
     */
    fun copyWindow2D(windowSize: Int): Array<FloatArray>? {
        val flat = copyWindow(windowSize) ?: return null
        return Array(windowSize) { row ->
            FloatArray(CHANNELS) { col -> flat[row * CHANNELS + col] }
        }
    }

    /** Reset — call before starting a new recording session. */
    fun reset() {
        writeAt.set(0)
        total.set(0)
        latestGx = 0f; latestGy = 0f; latestGz = 0f
    }
}
