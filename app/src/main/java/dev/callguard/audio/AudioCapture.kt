package dev.callguard.audio

import android.annotation.SuppressLint
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.util.Log
import dev.callguard.dsp.MelConfig
import kotlin.concurrent.thread
import kotlin.math.max

/**
 * Streams 16 kHz mono PCM from the microphone.
 *
 * [MediaRecorder.AudioSource.VOICE_COMMUNICATION] asks the platform for the
 * echo-cancelled / noise-suppressed voice path, which is the best a normal app can
 * get while a call is on speaker. It is still *not* the remote party's clean stream:
 * both speakers are mixed and the room is in there too.
 */
class AudioCapture(
    private val onBlock: (FloatArray) -> Unit,
) {

    private var record: AudioRecord? = null
    private var worker: Thread? = null

    @Volatile
    private var running = false

    val isRunning: Boolean get() = running

    @SuppressLint("MissingPermission")
    fun start(): Boolean {
        if (running) return true

        val minBuffer = AudioRecord.getMinBufferSize(
            MelConfig.SAMPLE_RATE,
            AudioFormat.CHANNEL_IN_MONO,
            AudioFormat.ENCODING_PCM_16BIT,
        )
        if (minBuffer <= 0) {
            Log.e(TAG, "unsupported capture config, minBuffer=$minBuffer")
            return false
        }

        val recorder = try {
            AudioRecord(
                MediaRecorder.AudioSource.VOICE_COMMUNICATION,
                MelConfig.SAMPLE_RATE,
                AudioFormat.CHANNEL_IN_MONO,
                AudioFormat.ENCODING_PCM_16BIT,
                max(minBuffer * 2, 8192),
            )
        } catch (t: Throwable) {
            Log.e(TAG, "AudioRecord creation failed", t)
            return false
        }

        if (recorder.state != AudioRecord.STATE_INITIALIZED) {
            Log.e(TAG, "AudioRecord not initialised")
            recorder.release()
            return false
        }

        record = recorder
        running = true

        try {
            recorder.startRecording()
        } catch (t: Throwable) {
            Log.e(TAG, "startRecording failed", t)
            running = false
            recorder.release()
            record = null
            return false
        }

        worker = thread(name = "callguard-capture", priority = Thread.MAX_PRIORITY) {
            val shorts = ShortArray(1600) // 100 ms
            while (running) {
                val read = try {
                    recorder.read(shorts, 0, shorts.size)
                } catch (t: Throwable) {
                    Log.e(TAG, "read failed", t)
                    -1
                }
                when {
                    read > 0 -> {
                        val block = FloatArray(read) { shorts[it] / 32768f }
                        onBlock(block)
                    }
                    read < 0 -> {
                        Log.w(TAG, "AudioRecord.read returned $read")
                        break
                    }
                }
            }
        }
        return true
    }

    fun stop() {
        running = false
        worker?.let { runCatching { it.join(500) } }
        worker = null
        record?.let { recorder ->
            runCatching { recorder.stop() }
            runCatching { recorder.release() }
        }
        record = null
    }

    private companion object {
        const val TAG = "CallGuard/Audio"
    }
}
