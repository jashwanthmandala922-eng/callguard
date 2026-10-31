package dev.callguard.ui

import androidx.compose.animation.core.RepeatMode
import androidx.compose.animation.core.animateFloat
import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.animation.core.infiniteRepeatable
import androidx.compose.animation.core.rememberInfiniteTransition
import androidx.compose.animation.core.tween
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.lerp
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import dev.callguard.dsp.Cue
import dev.callguard.model.Mode
import dev.callguard.model.Verdict
import kotlin.math.roundToInt
import kotlin.math.sqrt

fun verdictColor(risk: Float?): Color = when {
    risk == null -> GuardPalette.TextSecondary
    risk >= 0.65f -> GuardPalette.Synthetic
    risk <= 0.35f -> GuardPalette.Human
    else -> GuardPalette.Uncertain
}

fun verdictLabel(verdict: Verdict?): String = when (verdict) {
    Verdict.LIKELY_HUMAN -> "Sounds human"
    Verdict.UNCERTAIN -> "Not sure"
    Verdict.LIKELY_SYNTHETIC -> "Likely synthetic"
    null -> "Waiting for audio"
}

fun verdictExplanation(verdict: Verdict?): String = when (verdict) {
    Verdict.LIKELY_HUMAN -> "The model found no strong synthetic cues in this window."
    Verdict.UNCERTAIN -> "Cues are mixed. Ask a question only the real person could answer."
    Verdict.LIKELY_SYNTHETIC -> "Vocoder-like artefacts were detected. Verify through another channel."
    null -> "Start the live guard or pick a recording to analyse."
}

@Composable
fun StatusChip(mode: Mode, running: Boolean) {
    val transition = rememberInfiniteTransition(label = "status")
    val pulse by transition.animateFloat(
        initialValue = 0.35f,
        targetValue = 1f,
        animationSpec = infiniteRepeatable(tween(900), RepeatMode.Reverse),
        label = "statusPulse",
    )

    val (label, color) = when {
        running -> "LIVE" to GuardPalette.Synthetic
        mode == Mode.FILE -> "FILE" to GuardPalette.Accent
        else -> "IDLE" to GuardPalette.TextSecondary
    }
    val dotAlpha = if (running) pulse else 1f

    Row(
        modifier = Modifier
            .background(color.copy(alpha = 0.14f), RoundedCornerShape(50))
            .padding(horizontal = 12.dp, vertical = 7.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Box(
            Modifier
                .size(7.dp)
                .background(color.copy(alpha = dotAlpha), CircleShape),
        )
        Spacer(Modifier.width(6.dp))
        Text(
            text = label,
            style = MaterialTheme.typography.labelSmall,
            color = color,
            fontWeight = FontWeight.Bold,
            letterSpacing = 1.sp,
        )
    }
}

@Composable
fun RiskGauge(risk: Float?, modifier: Modifier = Modifier) {
    val animated by animateFloatAsState(
        targetValue = risk ?: 0f,
        animationSpec = tween(450),
        label = "risk",
    )
    val color = verdictColor(risk)

    Canvas(modifier) {
        val stroke = 16.dp.toPx()
        val inset = stroke / 2f + 6.dp.toPx()
        val arcSize = Size(size.width - inset * 2f, size.height - inset * 2f)
        val topLeft = Offset(inset, inset)
        val startAngle = 150f
        val maxSweep = 240f

        drawArc(
            color = GuardPalette.SurfaceRaised,
            startAngle = startAngle,
            sweepAngle = maxSweep,
            useCenter = false,
            topLeft = topLeft,
            size = arcSize,
            style = Stroke(width = stroke, cap = StrokeCap.Round),
        )
        if (risk != null) {
            drawArc(
                color = color,
                startAngle = startAngle,
                sweepAngle = maxSweep * animated.coerceIn(0f, 1f),
                useCenter = false,
                topLeft = topLeft,
                size = arcSize,
                style = Stroke(width = stroke, cap = StrokeCap.Round),
            )
        }
    }
}

@Composable
fun CueBar(cue: Cue) {
    Column(Modifier.fillMaxWidth()) {
        Row(
            Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Text(
                cue.label,
                style = MaterialTheme.typography.bodyMedium,
                color = GuardPalette.TextPrimary,
            )
            Text(
                "${(cue.synthetic * 100).roundToInt()}%",
                style = MaterialTheme.typography.labelMedium,
                color = GuardPalette.TextSecondary,
            )
        }
        Spacer(Modifier.height(6.dp))
        Box(
            Modifier
                .fillMaxWidth()
                .height(6.dp)
                .background(GuardPalette.SurfaceRaised, RoundedCornerShape(3.dp)),
        ) {
            Box(
                Modifier
                    .fillMaxWidth(cue.synthetic.coerceIn(0f, 1f))
                    .height(6.dp)
                    .background(
                        lerp(GuardPalette.Human, GuardPalette.Synthetic, cue.synthetic),
                        RoundedCornerShape(3.dp),
                    ),
            )
        }
        Spacer(Modifier.height(4.dp))
        Text(
            cue.detail,
            style = MaterialTheme.typography.labelSmall,
            color = GuardPalette.TextSecondary,
        )
    }
}

@Composable
fun SpectrogramStrip(mel: List<FloatArray>, modifier: Modifier = Modifier) {
    if (mel.isEmpty()) {
        Box(
            modifier
                .fillMaxWidth()
                .background(GuardPalette.SurfaceRaised, RoundedCornerShape(10.dp)),
            contentAlignment = Alignment.Center,
        ) {
            Text(
                "Waiting for audio…",
                style = MaterialTheme.typography.labelMedium,
                color = GuardPalette.TextSecondary,
            )
        }
        return
    }

    val rows = mel.first().size
    val columns = 100

    Canvas(modifier) {
        // robust contrast: mean +/- 2 sigma, so a quiet window still shows structure
        var sum = 0.0
        var sumSq = 0.0
        var count = 0
        for (frame in mel) {
            for (v in frame) {
                sum += v
                sumSq += v.toDouble() * v
                count++
            }
        }
        val mean = sum / count
        val sigma = sqrt((sumSq / count - mean * mean).coerceAtLeast(0.0))
        val lo = mean - 2.0 * sigma
        val span = (4.0 * sigma).coerceAtLeast(1e-6)

        val cellW = size.width / columns
        val cellH = size.height / rows
        val step = mel.size.toFloat() / columns

        for (c in 0 until columns) {
            val frame = mel[(c * step).toInt().coerceIn(0, mel.size - 1)]
            for (r in 0 until rows) {
                val norm = ((frame[r] - lo) / span).coerceIn(0.0, 1.0).toFloat()
                drawRect(
                    color = heat(norm),
                    topLeft = Offset(c * cellW, (rows - 1 - r) * cellH),
                    size = Size(cellW + 0.6f, cellH + 0.6f),
                )
            }
        }
    }
}

@Composable
fun LevelMeter(level: Float, modifier: Modifier = Modifier) {
    val fraction = (level * 6f).coerceIn(0f, 1f)
    Row(modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
        Text(
            "Input",
            style = MaterialTheme.typography.labelSmall,
            color = GuardPalette.TextSecondary,
            modifier = Modifier.width(38.dp),
        )
        Box(
            Modifier
                .weight(1f)
                .height(4.dp)
                .background(GuardPalette.SurfaceRaised, RoundedCornerShape(2.dp)),
        ) {
            Box(
                Modifier
                    .fillMaxWidth(fraction)
                    .height(4.dp)
                    .background(GuardPalette.Accent, RoundedCornerShape(2.dp)),
            )
        }
    }
}

private fun heat(v: Float): Color = if (v < 0.5f) {
    lerp(Color(0xFF061019), Color(0xFF2C6C9E), v * 2f)
} else {
    lerp(Color(0xFF2C6C9E), Color(0xFFFFD98A), (v - 0.5f) * 2f)
}

