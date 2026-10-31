package dev.callguard.ui

import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.darkColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.graphics.Color

object GuardPalette {
    val Background = Color(0xFF080B10)
    val Surface = Color(0xFF111823)
    val SurfaceRaised = Color(0xFF1A2432)
    val Outline = Color(0xFF273447)
    val TextPrimary = Color(0xFFE8EFF9)
    val TextSecondary = Color(0xFF93A6BF)
    val Accent = Color(0xFF4DA3FF)
    val Human = Color(0xFF35D07F)
    val Uncertain = Color(0xFFF2B23E)
    val Synthetic = Color(0xFFFF5C5C)
}

private val GuardColors = darkColorScheme(
    primary = GuardPalette.Accent,
    onPrimary = Color(0xFF00121F),
    secondary = Color(0xFF7BD1C4),
    onSecondary = Color(0xFF00201B),
    background = GuardPalette.Background,
    onBackground = GuardPalette.TextPrimary,
    surface = GuardPalette.Surface,
    onSurface = GuardPalette.TextPrimary,
    surfaceVariant = GuardPalette.SurfaceRaised,
    onSurfaceVariant = GuardPalette.TextSecondary,
    outline = GuardPalette.Outline,
    error = GuardPalette.Synthetic,
)

@Composable
fun CallGuardTheme(content: @Composable () -> Unit) {
    @Suppress("UNUSED_EXPRESSION")
    isSystemInDarkTheme() // the app is dark-only by design
    MaterialTheme(colorScheme = GuardColors, content = content)
}
