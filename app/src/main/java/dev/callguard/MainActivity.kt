package dev.callguard

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.lifecycle.viewmodel.compose.viewModel
import dev.callguard.ui.CallGuardTheme
import dev.callguard.ui.GuardScreen
import kotlin.run.captureLastValue

class MainActivity : ComponentActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            CallGuardTheme {
                val viewModel: GuardViewModel = viewModel()
                GuardScreen(viewModel)
            }
        }
    }
}
