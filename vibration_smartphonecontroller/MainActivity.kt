package com.yourname.advancedvibrator

import android.os.Build
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.os.VibrationEffect
import android.os.Vibrator
import android.util.Log
import android.webkit.JavascriptInterface
import android.webkit.WebView
import androidx.appcompat.app.AppCompatActivity
import kotlin.math.PI
import kotlin.math.sin

class MainActivity : AppCompatActivity() {

    private lateinit var vibrator: Vibrator
    private lateinit var webView: WebView

    private val handler = Handler(Looper.getMainLooper())
    private var vibrationTask: Runnable? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        vibrator = getSystemService(VIBRATOR_SERVICE) as Vibrator
        webView = findViewById(R.id.myWebView)

        webView.settings.javaScriptEnabled = true
        webView.addJavascriptInterface(WebAppInterface(this), "AndroidVibrator")
        WebView.setWebContentsDebuggingEnabled(true)
        webView.loadUrl("file:///android_asset/your_vibration_page.html")
    }

    override fun onDestroy() {
        super.onDestroy()
        stopVibrationSequenceInternal(false) // Activity销毁时清理
    }

    // --- [MOVED] stopVibrationSequenceInternal 函数现在是 MainActivity 的一部分 ---
    private fun stopVibrationSequenceInternal(notifyUI: Boolean) {
        Log.d("VibrationDebug", "Internal stop request. Notify UI: $notifyUI")
        vibrator.cancel()
        if (vibrationTask != null) {
            handler.removeCallbacks(vibrationTask!!)
        }
        if (notifyUI) {
            // 我们需要通过 handler 来确保在主线程更新 WebView
            handler.post {
                webView.evaluateJavascript("updateStatus('振动已手动停止。')", null)
            }
        }
    }

    // WebAppInterface 内部类
    inner class WebAppInterface(private val activity: MainActivity) {

        @JavascriptInterface
        fun startVibrationSequence(
            startFreq: Float,
            endFreq: Float,
            freqStep: Float,
            stageDurationMs: Int
        ) {
            activity.handler.post {
                activity.stopVibrationSequenceInternal(false) // 通过 activity 实例调用

                var currentFreq = startFreq

                activity.vibrationTask = object : Runnable {
                    override fun run() {
                        if (currentFreq > endFreq) {
                            Log.d("VibrationDebug", "Vibration sequence finished.")
                            updateWebViewStatus("振动序列已完成。")
                            return
                        }

                        generateAndPlaySingleStage(currentFreq, stageDurationMs)

                        currentFreq += freqStep

                        activity.handler.postDelayed(this, stageDurationMs.toLong())
                    }
                }

                Log.d("VibrationDebug", "Scheduling first vibration task.")
                updateWebViewStatus("正在准备振动序列...")
                activity.handler.post(activity.vibrationTask!!)
            }
        }

        private fun generateAndPlaySingleStage(freq: Float, durationMs: Int) {
            if (Build.VERSION.SDK_INT < Build.VERSION_CODES.O) return

            Log.d("VibrationDebug", "Generating stage for frequency: $freq Hz on MAIN thread.")
            updateWebViewStatus("正在播放: ${"%.1f".format(freq)} Hz")

            val timings = mutableListOf<Long>()
            val amplitudes = mutableListOf<Int>()

            val timeSliceMs = 20L
            val numSlices = durationMs / timeSliceMs
            val periodMs = if (freq > 0) 1000f / freq else 0f
            if (periodMs <= 0) return

            for (i in 0 until numSlices) {
                val currentTimeMs = i * timeSliceMs
                val progressInPeriod = (currentTimeMs % periodMs) / periodMs
                val amplitude = ((sin(progressInPeriod * 2 * PI) + 1) / 2 * 255).toInt().coerceIn(0, 255)
                timings.add(timeSliceMs)
                amplitudes.add(amplitude)
            }

            if (timings.isEmpty()) {
                Log.w("VibrationDebug", "Waveform for $freq Hz is empty.")
                return
            }

            try {
                val effect = VibrationEffect.createWaveform(timings.toLongArray(), amplitudes.toIntArray(), -1)
                activity.vibrator.vibrate(effect)
                Log.d("VibrationDebug", "Vibration command sent for $freq Hz.")
            } catch (e: Exception) {
                Log.e("VibrationDebug", "Error playing vibration for $freq Hz", e)
            }
        }

        private fun updateWebViewStatus(message: String) {
            activity.handler.post {
                activity.webView.evaluateJavascript("updateStatus('${message}')", null)
            }
        }

        @JavascriptInterface
        fun stopVibration() {
            activity.handler.post {
                // 通过 activity 实例调用
                activity.stopVibrationSequenceInternal(true)
            }
        }
    }
}
