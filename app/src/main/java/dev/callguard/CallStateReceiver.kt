package dev.callguard

import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.media.MediaRecorder
import android.telephony.TelephonyManager
import android.util.Log
import dev.callguard.audio.FileAudioDecoder
import dev.callguard.dsp.MelConfig
import dev.callguard.ml.DetectionEngine
import dev.callguard.ml.DetectorFactory
import dev.callguard.model.GuardBus
import java.io.File
import java.io.IOException

/**
 * Broadcast receiver that listens for phone call state changes.
 * When a call ends, it records the audio and triggers AI analysis.
 *
 * NOTE: Direct call recording is restricted on Android. This receiver captures
 * the call audio using the microphone during the call and saves it for analysis
 * after the call ends. Users must enable call recording in the app settings.
 */
class CallStateReceiver : BroadcastReceiver() {

    private companion object {
        private const val TAG = "CallGuard/CallReceiver"
        private const val RECORDING_DIR = "CallGuard/Recordings"
        private const val MAX_CALL_DURATION_MS = 300_000 // 5 minutes max
    }

    override fun onReceive(context: Context, intent: Intent) {
        val state = intent.getStringExtra(TelephonyManager.EXTRA_STATE)
        val previousState = intent.getStringExtra(TelephonyManager.EXTRA_PREVIOUS_STATE)

        when (state) {
            TelephonyManager.EXTRA_STATE_RINGING -> {
                // Phone is ringing - could pre-roll recording
                Log.d(TAG, "Phone ringing")
            }
            TelephonyManager.EXTRA_STATE_OFFHOOK -> {
                // Call is active (connected or connecting)
                Log.d(TAG, "Call off-hook (active)")
                startCallRecording(context)
            }
            TelephonyManager.EXTRA_STATE_IDLE -> {
                // Phone went back to idle - call ended
                Log.d(TAG, "Call idle (ended)")
                endCallRecording(context)
            }
            else -> Log.d(TAG, "Phone state: $state")
        }
    }

    private fun startCallRecording(context: Context) {
        // Ensure we have a directory for recordings
        val recordingsDir = File(context.getExternalFilesDir(null), RECORDING_DIR)
        recordingsDir.mkdirs()

        // Start a foreground service to record audio during the call
        // We use an intent service that will keep recording while the call is active
        val recordIntent = Intent(context, CallRecordingService::class.java)
        recordIntent.putExtra("recordings_dir", recordingsDir.absolutePath)
        context.startService(recordIntent)

        GuardBus.callRecordingStarted()
    }

    private fun endCallRecording(context: Context) {
        val recordIntent = Intent(context, CallRecordingService::class.java)
        recordIntent.putExtra("stop_recording", true)
        context.stopService(recordIntent)

        // Trigger analysis of the recorded call
        GuardBus.callRecordingEnded()
    }
}