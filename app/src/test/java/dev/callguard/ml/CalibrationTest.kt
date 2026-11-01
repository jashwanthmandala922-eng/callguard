package dev.callguard.ml

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * The shipped Platt parameters turn the compressed raw sigmoid into a usable
 * score. These checks pin the two properties the app relies on:
 *
 *  - 0.5 is the operating point, so the raw EER threshold must land there;
 *  - the map is strictly increasing, so it cannot reorder anything (and
 *    therefore cannot change the measured EER).
 */
class CalibrationTest {

    // Fitted values currently shipped in model_meta.json.
    private val calibration = Calibration(a = 0.4930f, b = 3.7421f)

    @Test
    fun rawOperatingPointMapsToHalf() {
        // EER threshold measured on the cross-dataset corpus.
        assertEquals(0.5f, calibration.apply(0.000467f), 0.01f)
    }

    @Test
    fun mapIsMonotone() {
        var previous = -1f
        for (i in 0..1000) {
            val raw = i / 1000f
            val calibrated = calibration.apply(raw)
            assertTrue("score went backwards at raw=$raw", calibrated >= previous)
            previous = calibrated
        }
    }

    @Test
    fun extremaStayInRange() {
        val low = calibration.apply(0f)
        val high = calibration.apply(1f)
        assertTrue("low=$low", low in 0f..0.02f)
        assertTrue("high=$high", high in 0.98f..1f)
    }

    @Test
    fun identityLeavesScoresAlone() {
        assertEquals(0.25f, Calibration.NONE.apply(0.25f), 1e-6f)
    }
}
