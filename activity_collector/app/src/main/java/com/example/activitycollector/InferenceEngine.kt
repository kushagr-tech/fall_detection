package com.example.activitycollector

import android.content.Context
import android.util.Log
import org.pytorch.IValue
import org.pytorch.LiteModuleLoader
import org.pytorch.Module
import org.pytorch.Tensor
import java.io.File
import java.io.FileOutputStream
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean

/**
 * InferenceEngine
 * ===============
 * Loads the TorchScript (.ptl) model from the app's assets folder and runs
 * sliding-window classification on sensor data supplied by [SensorBuffer].
 *
 * Design
 * ------
 * • Inference runs on a dedicated single-thread executor so it never blocks
 *   the sensor callback thread or the UI thread.
 * • The engine schedules itself at [inferenceIntervalMs] (default 400 ms).
 *   At 50 Hz / 100-sample windows this gives a fresh classification every
 *   half-window stride — the same 50 % overlap used during training.
 * • The model input shape is (1, windowSize, 6) — batch=1, time, channels.
 * • Softmax is applied in-place; the class with the highest probability is
 *   returned together with its confidence value.
 *
 * Model file
 * ----------
 * Place "har_model.ptl" in app/src/main/assets/.
 * The first call to [start] copies it to the app's internal file cache
 * (PyTorch Mobile requires a File path, not an InputStream).
 *
 * Output labels
 * -------------
 * Must match the order used when the model was exported.  The label order is
 * stored in "har_labels.txt" in assets (one label per line, alphabetical).
 * Default fallback order: falling, lying, running, sitting, standing, walking.
 */
class InferenceEngine(
    private val context: Context,
    private val buffer: SensorBuffer,
    private val windowSize: Int = 100,
    private val inferenceIntervalMs: Long = 400L,
    private val onResult: (label: String, confidence: Float, probs: FloatArray) -> Unit,
) {

    companion object {
        private const val TAG          = "InferenceEngine"
        private const val MODEL_ASSET  = "har_model.ptl"
        private const val LABELS_ASSET = "har_labels.txt"

        private val DEFAULT_LABELS = listOf(
            "falling", "lying", "running", "sitting", "standing", "walking"
        )
    }

    private var module: Module? = null
    private val labels: List<String> by lazy { loadLabels() }
    private val executor = Executors.newSingleThreadExecutor { r ->
        Thread(r, "inference-thread").also { it.isDaemon = true }
    }
    private val running = AtomicBoolean(false)

    private fun loopInference() {
        if (!running.get()) return
        try {
            runInference()
        } catch (e: Exception) {
            Log.e(TAG, "Inference error: ${e.message}", e)
        }
        scheduleNextInference()
    }

    private fun scheduleNextInference() {
        if (!running.get()) return
        executor.execute {
            try {
                Thread.sleep(inferenceIntervalMs)
            } catch (_: InterruptedException) {
                return@execute
            }
            if (running.get()) {
                executor.execute(::loopInference)
            }
        }
    }

    // ── Lifecycle ─────────────────────────────────────────────────────────────

    /**
     * Load the model (once) and start the periodic inference loop.
     * Safe to call multiple times — subsequent calls are no-ops if running.
     */
    fun start() {
        if (running.getAndSet(true)) return
        executor.execute {
            if (module == null) {
                module = loadModel()
            }
            if (module != null && running.get()) {
                executor.execute(::loopInference)
            }
        }
    }

    /** Stop inference.  The executor remains alive for a quick restart. */
    fun stop() {
        running.set(false)
    }

    /** Release all resources.  Do not call [start] after this. */
    fun destroy() {
        stop()
        executor.shutdownNow()
        module?.destroy()
        module = null
    }

    // ── Core inference ────────────────────────────────────────────────────────

    private fun runInference() {
        val currentModule = module ?: return
        val flat = buffer.copyWindow(windowSize) ?: return   // not enough data yet

        // Input tensor shape: (1, windowSize, 6)  — float32
        val inputTensor = Tensor.fromBlob(
            flat,
            longArrayOf(1L, windowSize.toLong(), SensorBuffer.CHANNELS.toLong()),
        )

        val outputTensor: Tensor = currentModule
            .forward(IValue.from(inputTensor))
            .toTensor()

        val logits = outputTensor.dataAsFloatArray   // shape: (n_classes,)
        val probs  = softmax(logits)
        val best   = probs.indices.maxByOrNull { probs[it] } ?: 0
        val label  = labels.getOrElse(best) { "unknown" }

        onResult(label, probs[best], probs)
    }

    // ── Model loading ─────────────────────────────────────────────────────────

    /**
     * PyTorch Mobile's [LiteModuleLoader.load] requires a filesystem path.
     * We copy the asset to the cache dir on first use.
     */
    private fun loadModel(): Module? {
        return try {
            val modelFile = File(context.cacheDir, MODEL_ASSET)
            context.assets.open(MODEL_ASSET).use { input ->
                FileOutputStream(modelFile).use { output -> input.copyTo(output) }
            }
            Log.i(TAG, "Model synced to cache: ${modelFile.absolutePath} (${modelFile.length()} bytes)")
            LiteModuleLoader.load(modelFile.absolutePath).also {
                Log.i(TAG, "TorchScript model loaded. Labels: $labels")
            }
        } catch (e: Exception) {
            Log.e(TAG, "Failed to load TorchScript model: ${e.message}", e)
            null
        }
    }

    private fun loadLabels(): List<String> {
        return try {
            context.assets.open(LABELS_ASSET)
                .bufferedReader()
                .readLines()
                .map { it.trim() }
                .filter { it.isNotEmpty() }
                .also { Log.i(TAG, "Loaded ${it.size} labels: $it") }
        } catch (e: Exception) {
            Log.w(TAG, "har_labels.txt not found in assets, using default order.")
            DEFAULT_LABELS
        }
    }

    // ── Math ──────────────────────────────────────────────────────────────────

    private fun softmax(logits: FloatArray): FloatArray {
        val max = logits.maxOrNull() ?: 0f
        val exp = FloatArray(logits.size) { Math.exp((logits[it] - max).toDouble()).toFloat() }
        val sum = exp.sum()
        return if (sum > 0f) {
            FloatArray(exp.size) { exp[it] / sum }
        } else {
            FloatArray(exp.size) { 1f / exp.size }
        }
    }
}
