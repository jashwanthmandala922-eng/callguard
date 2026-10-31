package dev.callguard.audio

import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Context
import android.content.Intent
import android.content.pm.ServiceInfo
import android.os.Build
import android.os.IBinder
import android.os.SystemClock
import android.util.Log
import androidx.core.app.NotificationCompat
import androidx.core.app.ServiceCompat
import androidx.core.content.ContextCompat
import dev.callguard.MainActivity
import dev.callguard.R
import dev.callguard.dsp.MelConfig
import dev.callguard.ml.DetectionEngine
import dev.callguard.ml.DetectorFactory
import dev.callguard.model.GuardBus

/**
 * Foreground service that owns microphone capture and the inference engine while
 * the live guard is running.
 *
 * A foreground service is mandatory: Android will not let a background app hold the
 * mic, and the user needs a persistent "this is recording" affordance.
 */
class LiveGuardService : Service() {

    private var engine: DetectionEngine? = null
    private var capture: AudioCapture? = null
    private var secondsAnalyzed = 0f

    override fun onCreate() {
        super.onCreate()
        createChannel()

        val detector = DetectorFactory.create(this)
        GuardBus.configureModel(detector.name, detector.isDemo)
        engine = DetectionEngine(detector)
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        if (intent?.action == ACTION_STOP) {
            stopSelf()
            return START_NOT_STICKY
        }

        val type = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
            ServiceInfo.FOREGROUND_SERVICE_TYPE_MICROPHONE
        } else {
            0
        }
        ServiceCompat.startForeground(this, NOTIFICATION_ID, buildNotification(), type)

        if (capture == null) startCapture()
        return START_NOT_STICKY
    }

    private fun startCapture() {
        secondsAnalyzed = 0f
        GuardBus.liveStarted()

        val audioCapture = AudioCapture { block ->
            val update = engine?.push(block) ?: return@AudioCapture
            secondsAnalyzed += MelConfig.DECISION_HOP_SAMPLES.toFloat() / MelConfig.SAMPLE_RATE

            GuardBus.liveUpdate(
                risk = update.risk,
                rawRisk = update.rawRisk,
                cues = update.cues,
                spectrogram = update.mel.toList(),
                level = update.level,
                seconds = secondsAnalyzed,
            )
        }

        if (!audioCapture.start()) {
            Log.e(TAG, "capture failed to start")
            GuardBus.fail("Could not open the microphone. Another app may be using it.")
            stopSelf()
            return
        }
        capture = audioCapture
    }

    override fun onDestroy() {
        capture?.stop()
        capture = null
        engine?.close()
        engine = null
        GuardBus.liveStopped()
        ServiceCompat.stopForeground(this, ServiceCompat.STOP_FOREGROUND_REMOVE)
        super.onDestroy()
    }

    override fun onBind(intent: Intent?): IBinder? = null

    private fun buildNotification() = NotificationCompat.Builder(this, CHANNEL_ID)
        .setSmallIcon(R.drawable.ic_stat_guard)
        .setContentTitle(getString(R.string.notification_title))
        .setContentText(getString(R.string.notification_text))
        .setOngoing(true)
        .setSilent(true)
        .setPriority(NotificationCompat.PRIORITY_LOW)
        .setContentIntent(
            PendingIntent.getActivity(
                this,
                0,
                Intent(this, MainActivity::class.java),
                PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE,
            ),
        )
        .addAction(
            0,
            getString(R.string.notification_stop),
            PendingIntent.getService(
                this,
                1,
                Intent(this, LiveGuardService::class.java).setAction(ACTION_STOP),
                PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE,
            ),
        )
        .build()

    private fun createChannel() {
        val manager = getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
        val channel = NotificationChannel(
            CHANNEL_ID,
            getString(R.string.notification_channel_name),
            NotificationManager.IMPORTANCE_LOW,
        ).apply {
            description = getString(R.string.notification_text)
            setShowBadge(false)
        }
        manager.createNotificationChannel(channel)
    }

    companion object {
        private const val TAG = "CallGuard/Service"
        private const val CHANNEL_ID = "live_guard"
        private const val NOTIFICATION_ID = 42
        const val ACTION_STOP = "dev.callguard.action.STOP"

        fun start(context: Context) {
            val intent = Intent(context, LiveGuardService::class.java)
            ContextCompat.startForegroundService(context, intent)
        }

        fun stop(context: Context) {
            context.stopService(Intent(context, LiveGuardService::class.java))
        }
    }
}
