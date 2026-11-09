package dev.callguard

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.content.Context
import android.os.Build
import android.util.Log
import dev.callguard.model.GuardUiState
import dev.callguard.model.Mode
import dev.callguard.model.Verdict
import kotlin.math.abs

const val CALL_ANALYSIS_CHANNEL_ID = "call_analysis"
const val CALL_ANALYSIS_NOTIFICATION_ID = 1001

object NotificationHelper {

    fun createAnalysisChannel(context: Context) {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val channel = NotificationChannel(
                CALL_ANALYSIS_CHANNEL_ID,
                "Call Analysis Results",
                NotificationManager.IMPORTANCE_LOW
            ).apply {
                description = "Results from AI analysis of call recordings"
                importance = NotificationManager.IMPORTANCE_LOW
            }
            val manager = context.getSystemService(NotificationManager::class.java)
            manager.createNotificationChannel(channel)
        }
    }

    fun showAnalysisResult(
        context: Context,
        risk: Float,
        rawRisk: Float,
        verdict: Verdict,
        caveat: String?,
        modelName: String,
        secondsAnalyzed: Float
    ) {
        createAnalysisChannel(context)

        val verdictLabel = when (verdict) {
            Verdict.LIKELY_HUMAN -> "Sounds human"
            Verdict.UNCERTAIN -> "Not sure"
            Verdict.LIKELY_SYNTHETIC -> "Likely synthetic"
        }

        val explanation = when (verdict) {
            Verdict.LIKELY_HUMAN -> "The model found no strong synthetic cues in this audio."
            Verdict.UNCERTAIN -> "Cues are mixed. The audio could be real or synthetic."
            Verdict.LIKELY_SYNTHETIC -> "Vocoder-like artefacts were detected. Verify through another channel."
        }

        val scoreText = "${(risk * 100).roundToInt()}% synthetic"
        val confidence = ((abs(risk - 0.5f) * 2f).coerceIn(0f, 1f) * 100).roundToInt()

        val notification = Notification.Builder(context, CALL_ANALYSIS_CHANNEL_ID)
            .setSmallIcon(R.drawable.ic_guard)
            .setContentTitle("CallGuard: Call Analysis Complete")
            .setContentText("This call was analyzed: $verdictLabel\n$scoreText (${confidence}% confidence)")
            .setStyle(Notification.BigTextStyle()
                .bigText("""
                    Result: $verdictLabel
                    
                    Synthetic likelihood: ${scoreText}
                    Confidence: ${confidence}%
                    
                    $explanation
                    
                    Audio analyzed: ${secondsAnalyzed.toInt()} seconds
                    
                    ${caveat ?: ""}
                """.trimToNull())
            )
            .setPriority(Notification.PRIORITY_LOW)
            .setOngoing(false)
            .setAutoCancel(true)
            .build()

        val manager = context.getSystemService(NotificationManager::class.java)
        manager.notify(CALL_ANALYSIS_NOTIFICATION_ID, notification)

        Log.i("CallGuard/Notification", "Showing analysis result: $verdict, risk=$risk")
    }
}