package dev.callguard.ui

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.os.Build
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ColumnScope
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.navigationBarsPadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.statusBarsPadding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.core.content.ContextCompat
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import dev.callguard.GuardViewModel
import dev.callguard.audio.LiveGuardService
import dev.callguard.dsp.Cue
import dev.callguard.model.GuardUiState
import dev.callguard.model.Mode
import kotlin.math.roundToInt

@Composable
fun GuardScreen(viewModel: GuardViewModel) {
    val state by viewModel.state.collectAsStateWithLifecycle()
    val context = LocalContext.current

    var micGranted by remember {
        mutableStateOf(
            ContextCompat.checkSelfPermission(context, Manifest.permission.RECORD_AUDIO) ==
                PackageManager.PERMISSION_GRANTED,
        )
    }

    val notificationLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.RequestPermission(),
    ) { }

    val micLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.RequestPermission(),
    ) { granted ->
        micGranted = granted
        if (granted) {
            requestNotificationsIfNeeded(context) { notificationLauncher.launch(it) }
            LiveGuardService.start(context)
        } else {
            viewModel.reportError("Microphone permission is required for live mode.")
        }
    }

    val fileLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.OpenDocument(),
    ) { uri ->
        uri?.let(viewModel::analyzeFile)
    }

    val onToggleLive: () -> Unit = {
        if (state.running) {
            LiveGuardService.stop(context)
        } else if (micGranted) {
            requestNotificationsIfNeeded(context) { notificationLauncher.launch(it) }
            LiveGuardService.start(context)
        } else {
            micLauncher.launch(Manifest.permission.RECORD_AUDIO)
        }
    }

    val onPickFile: () -> Unit = {
        fileLauncher.launch(arrayOf("audio/*"))
    }

    Surface(Modifier.fillMaxSize(), color = MaterialTheme.colorScheme.background) {
        Column(
            Modifier
                .fillMaxSize()
                .verticalScroll(rememberScrollState())
                .statusBarsPadding()
                .navigationBarsPadding()
                .padding(horizontal = 20.dp, vertical = 14.dp),
            verticalArrangement = Arrangement.spacedBy(14.dp),
        ) {
            Header(state)
            VerdictCard(state)

            state.error?.let { message ->
                ErrorCard(message) { viewModel.clearError() }
            }

            state.progress?.let { progress ->
                ProgressCard(progress, state.mode)
            }

            if (state.cues.isNotEmpty()) {
                CuesCard(state.cues)
            }

            SpectrogramCard(state)

            Controls(
                running = state.running,
                onToggleLive = onToggleLive,
                onPickFile = onPickFile,
            )

            Footer(state)
        }
    }
}

@Composable
private fun Header(state: GuardUiState) {
    Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
        Column(Modifier.weight(1f)) {
            Text(
                "CallGuard",
                style = MaterialTheme.typography.headlineSmall,
                fontWeight = FontWeight.Bold,
                color = GuardPalette.TextPrimary,
            )
            Text(
                "On-device synthetic speech check",
                style = MaterialTheme.typography.bodySmall,
                color = GuardPalette.TextSecondary,
            )
        }
        StatusChip(state.mode, state.running)
    }
}

@Composable
private fun VerdictCard(state: GuardUiState) {
    SectionCard {
        Box(Modifier.fillMaxWidth(), contentAlignment = Alignment.Center) {
            Box(Modifier.size(220.dp), contentAlignment = Alignment.Center) {
                RiskGauge(state.risk, Modifier.fillMaxSize())
                Column(horizontalAlignment = Alignment.CenterHorizontally) {
                    Text(
                        text = state.risk?.let { "${(it * 100).roundToInt()}%" } ?: "—",
                        fontSize = 46.sp,
                        fontWeight = FontWeight.Bold,
                        color = verdictColor(state.risk),
                    )
                    Text(
                        text = if (state.risk != null) "synthetic likelihood" else "no score yet",
                        style = MaterialTheme.typography.labelSmall,
                        color = GuardPalette.TextSecondary,
                    )
                }
            }
        }

        Spacer(Modifier.height(10.dp))
        Text(
            text = verdictLabel(state.verdict),
            style = MaterialTheme.typography.titleLarge,
            fontWeight = FontWeight.SemiBold,
            color = verdictColor(state.risk),
        )
        Spacer(Modifier.height(4.dp))
        Text(
            text = verdictExplanation(state.verdict),
            style = MaterialTheme.typography.bodySmall,
            color = GuardPalette.TextSecondary,
        )

        if (state.risk != null) {
            Spacer(Modifier.height(12.dp))
            Text(
                // Deliberately no "confidence" figure: with a ~30% cross-dataset
                // EER, a distance-from-0.5 number would overstate certainty.
                text = buildString {
                    append(if (state.mode == Mode.LIVE) "Streaming" else "Analysed")
                    append(" · ")
                    append("%.1f".format(state.secondsAnalyzed))
                    append(" s")
                },
                style = MaterialTheme.typography.labelSmall,
                color = GuardPalette.TextSecondary,
            )
        }
    }
}

@Composable
private fun CuesCard(cues: List<Cue>) {
    SectionCard {
        Text(
            "What the audio looks like",
            style = MaterialTheme.typography.titleSmall,
            fontWeight = FontWeight.SemiBold,
            color = GuardPalette.TextPrimary,
        )
        Spacer(Modifier.height(12.dp))
        cues.forEachIndexed { index, cue ->
            if (index > 0) Spacer(Modifier.height(14.dp))
            CueBar(cue)
        }
    }
}

@Composable
private fun SpectrogramCard(state: GuardUiState) {
    SectionCard {
        Row(
            Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Text(
                "Log-mel spectrogram",
                style = MaterialTheme.typography.titleSmall,
                fontWeight = FontWeight.SemiBold,
                color = GuardPalette.TextPrimary,
            )
            Text(
                if (state.spectrogram.isEmpty()) "idle" else "${state.spectrogram.size} frames",
                style = MaterialTheme.typography.labelSmall,
                color = GuardPalette.TextSecondary,
            )
        }
        Spacer(Modifier.height(10.dp))
        SpectrogramStrip(
            mel = state.spectrogram,
            modifier = Modifier
                .fillMaxWidth()
                .height(96.dp),
        )
        Spacer(Modifier.height(10.dp))
        LevelMeter(state.level)
    }
}

@Composable
private fun ProgressCard(progress: Float, mode: Mode) {
    SectionCard {
        Text(
            if (mode == Mode.FILE) "Decoding and scoring the file…" else "Working…",
            style = MaterialTheme.typography.bodySmall,
            color = GuardPalette.TextSecondary,
        )
        Spacer(Modifier.height(8.dp))
        LinearProgressIndicator(
            progress = { progress },
            modifier = Modifier.fillMaxWidth(),
            color = GuardPalette.Accent,
            trackColor = GuardPalette.SurfaceRaised,
        )
    }
}

@Composable
private fun ErrorCard(message: String, onDismiss: () -> Unit) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        shape = RoundedCornerShape(16.dp),
        colors = CardDefaults.cardColors(
            containerColor = GuardPalette.Synthetic.copy(alpha = 0.10f),
        ),
    ) {
        Row(
            Modifier
                .fillMaxWidth()
                .padding(start = 14.dp, end = 6.dp, top = 6.dp, bottom = 6.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Text(
                message,
                Modifier.weight(1f),
                style = MaterialTheme.typography.bodySmall,
                color = GuardPalette.Synthetic,
            )
            TextButton(onClick = onDismiss) {
                Text("Dismiss", color = GuardPalette.Synthetic)
            }
        }
    }
}

@Composable
private fun Controls(running: Boolean, onToggleLive: () -> Unit, onPickFile: () -> Unit) {
    Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
        Button(
            onClick = onToggleLive,
            modifier = Modifier
                .fillMaxWidth()
                .height(54.dp),
            shape = RoundedCornerShape(16.dp),
            colors = ButtonDefaults.buttonColors(
                containerColor = if (running) {
                    GuardPalette.Synthetic.copy(alpha = 0.16f)
                } else {
                    GuardPalette.Accent
                },
                contentColor = if (running) GuardPalette.Synthetic else Color(0xFF00121F),
            ),
        ) {
            Text(
                if (running) "Stop live guard" else "Start live guard",
                fontWeight = FontWeight.SemiBold,
                fontSize = 16.sp,
            )
        }

        OutlinedButton(
            onClick = onPickFile,
            enabled = !running,
            modifier = Modifier
                .fillMaxWidth()
                .height(50.dp),
            shape = RoundedCornerShape(16.dp),
            border = BorderStroke(1.dp, GuardPalette.Outline),
            colors = ButtonDefaults.outlinedButtonColors(contentColor = GuardPalette.TextPrimary),
        ) {
            Text("Analyse an audio file…", fontSize = 15.sp)
        }
    }
}

@Composable
private fun Footer(state: GuardUiState) {
    Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
        if (state.caveat != null) {
            Text(
                "ABOUT THIS MODEL",
                style = MaterialTheme.typography.labelSmall,
                fontWeight = FontWeight.Bold,
                letterSpacing = 1.sp,
                color = GuardPalette.Uncertain,
            )
            Text(
                state.caveat,
                style = MaterialTheme.typography.bodySmall,
                color = GuardPalette.TextSecondary,
            )
        }
        Text(
            "Live mode hears the room through the microphone, so put the call on speakerphone. " +
                "Android does not expose the carrier call stream to third-party apps.",
            style = MaterialTheme.typography.bodySmall,
            color = GuardPalette.TextSecondary,
        )
        Text(
            "Everything runs on this device. No audio leaves the phone.",
            style = MaterialTheme.typography.bodySmall,
            color = GuardPalette.TextSecondary,
        )
        Text(
            "Model: ${state.modelName}",
            style = MaterialTheme.typography.labelSmall,
            color = GuardPalette.TextSecondary,
        )
        Spacer(Modifier.height(4.dp))

        // Call recording status
        if (state.mode == Mode.IDLE && !state.running) {
            Row(
                modifier = Modifier.padding(top = 4.dp),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                if (!state.callRecordingEnabled) {
                    Button(
                        onClick = { viewModel.setCallRecordingEnabled(true) },
                        modifier = Modifier.fillMaxWidth(),
                        colors = ButtonDefaults.buttonColors(
                            containerColor = GuardPalette.Accent,
                            contentColor = Color(0xFF00121F),
                        ),
                        shape = RoundedCornerShape(8.dp),
                    ) {
                        Text("Enable call recording for analysis")
                    }
                } else {
                    Text("Call recording enabled - AI will analyse ended calls", style = MaterialTheme.typography.labelSmall, color = GuardPalette.Human)
                }
            }
        }
    }
}

@Composable
private fun SectionCard(content: @Composable ColumnScope.() -> Unit) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        shape = RoundedCornerShape(22.dp),
        colors = CardDefaults.cardColors(containerColor = GuardPalette.Surface),
    ) {
        Column(
            Modifier
                .fillMaxWidth()
                .padding(18.dp),
            content = content,
        )
    }
}

private fun requestNotificationsIfNeeded(context: Context, launch: (String) -> Unit) {
    if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
        val granted = ContextCompat.checkSelfPermission(
            context,
            Manifest.permission.POST_NOTIFICATIONS,
        ) == PackageManager.PERMISSION_GRANTED
        if (!granted) launch(Manifest.permission.POST_NOTIFICATIONS)
    }
}
