package dev.callguard

import android.annotation.SuppressLint
import android.content.Context
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.os.Build
import android.util.Log
import dev.callguard.dsp.MelConfig
import dev.callguard.ml.DetectorFactory
import dev.callguard.model.GuardBus
import kotlin.io.path.joinToPath
import kotlin.io.path.createTempFile
import java.io.File
import java.io.FileOutputStream

/**
 * Foreground service that records audio during phone calls.
 * Starts recording when a call goes off-hook, stops when the call ends.
 * The recorded audio is then available for AI analysis.
 */
class CallRecordingService : BaseRecordingService() {

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        val stopRecording = intent?.getBooleanExtra("stop_recording", false) ?: false

        if (stopRecording) {
            stopRecording()
            return START_STICKY
        }

        super.onStartCommand(intent, flags, startId)
        startRecordingIfCallActive()
        return START_STICKY
    }

    private fun startRecordingIfCallActive() {
        // Check if there's an active call
        val telephonyManager = getSystemService(Context.TELEPHONY_SERVICE) as android.telephony.TelephonyManager
        val currentCallState = telephonyManager.currentCallState

        if (currentCallState == android.telephony.TelephonyManager.CALL_STATE_OFFHOOK) {
            startRecording()
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        stopRecording()
    }
}

/**
 * Base class providing the audio recording logic for call recording.
 * Uses AudioRecord to capture microphone audio during calls.
 */
abstract class BaseRecordingService : android.app.Service() {

    private var record: AudioRecord? = null
    private var recordingThread: Thread? = null
    @Volatile
    private var running = false
    val isRunning: Boolean get() = running

    /** Directory where recorded WAV files will be saved */
    protected var recordingsDir: String = ""

    override fun onCreate() {
        super.onCreate()
        // Ensure we have the RECORD_AUDIO permission
        // (already declared in manifest)
    }

    /**
     * Start recording audio from the microphone.
     * Called when a call is detected as active.
     */
    fun startRecording() {
        if (running) return

        running = true

        // Get min buffer size for 16kHz mono
        val sampleRate = MelConfig.SAMPLE_RATE
        val channelConfig = AudioFormat.CHANNEL_IN_MONO
        val audioFormat = AudioFormat.ENCODING_PCM_16BIT
        val minBufferSize = AudioRecord.getMinBufferSize(
            sampleRate,
            channelConfig,
            audioFormat
        )

        if (minBufferSize <= 0) {
            Log.e(TAG, "unsupported audio config, minBufferSize=$minBufferSize")
            running = false
            return
        }

        val bufferSize = minBufferSize * 2

        record = AudioRecord(
            MediaRecorder.AudioSource.MIC,
            sampleRate,
            channelConfig,
            audioFormat,
            bufferSize
        )

        if (record!!.state != AudioRecord.STATE_INITIALIZED) {
            Log.e(TAG, "AudioRecord not initialized")
            record = null
            running = false
            return
        }

        record!!.startRecording()

        recordingThread = Thread(java.lang.Thread.MIN_PRIORITY, "CallGuard/Recording") {
            val file = createRecordingFile()
            val os = FileOutputStream(file)
            val shorts = ShortArray(minBufferSize)
            var totalSamples = 0

            while (running) {
                val read = try {
                    record!!.read(shorts, 0, shorts.size)
                } catch (e: Exception) {
                    Log.e(TAG, "AudioRecord read failed", e)
                    0
                }

                if (read > 0) {
                    os.write(shorts, 0, read)
                    totalSamples += read

                    // Limit recording to max duration
                    if (totalSamples >= MelConfig.SAMPLE_RATE * 5) { // 5 minutes
                        running = false
                    }
                }
            }

            os.close()
            runOnUiThread {
                GuardBus.callRecordingSaved(file.absolutePath)
            }
        }

        recordingThread?.start()
    }

    /** Stop recording and release resources. */
    fun stopRecording() {
        running = false
        recordingThread?.let { it.join(3000) }
        recordingThread = null

        record?.let { recorder ->
            runCatching { recorder.stop() }
            runCatching { recorder.release() }
        }
        record = null
    }

    /** Create a temp WAV file for the recording */
    private fun createRecordingFile(): File {
        val dir = java.io.File(recordingsDir).apply { mkdirs() }
        return createTempFile("call_guard_", ".wav", dir)
    }

    @SuppressLint("UnsafeOptInUsageError")
    private fun runOnUiThread(action: () -> Unit) {
        // Run on the main thread to update GuardBus
        android.os.Handler(android.os.Looper.getMainLooper()).post(action)
    }

    @SuppressLint("SuspiciousTrampoline")
    private fun runCatching(block: () -> Unit) {
        try { block() } catch (e: Exception) { Log.e(TAG, "Error in running task", e) }
    }

    private companion object {
        private const val TAG = "CallGuard/RecordingService"
    }
}