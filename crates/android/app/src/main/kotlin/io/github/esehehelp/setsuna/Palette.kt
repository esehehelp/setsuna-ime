package io.github.esehehelp.setsuna

import android.content.Context
import android.content.res.Configuration

/**
 * The keyboard's colours, by the system's light or dark theme. The layout
 * is Gboard's, the look is Setsuna's own: slightly warm greys, and the
 * icon's gradient (amber, coral, violet) on the Enter key and the stripe
 * along the top, coral where something is on.
 */
class Palette(context: Context) {
    val dark = (context.resources.configuration.uiMode and Configuration.UI_MODE_NIGHT_MASK) ==
        Configuration.UI_MODE_NIGHT_YES

    val background = if (dark) 0xFF17161A.toInt() else 0xFFF3F1F4.toInt()

    /** Letters, digits, space, the period. */
    val key = if (dark) 0xFF2A292E.toInt() else 0xFFFFFFFF.toInt()

    /** Shift, delete, the arrows, the mode keys. */
    val special = if (dark) 0xFF38363D.toInt() else 0xFFE2DEE6.toInt()
    val pressed = if (dark) 0xFF4D4A53.toInt() else 0xFFCBC6D1.toInt()
    val text = if (dark) 0xFFECEAEF.toInt() else 0xFF1D1B20.toInt()
    val dim = if (dark) 0xFF8F8B96.toInt() else 0xFF6E6A75.toInt()

    /** The bubble over a pressed key. */
    val preview = if (dark) 0xFF3E3B44.toInt() else 0xFFFFFFFF.toInt()

    /** The icon's coral: the mode in use, Shift, the open tool, the picked candidate. */
    val accent = 0xFFFF7066.toInt()
    val accentSoft = if (dark) 0x55FF7066 else 0x33FF7066

    /** The icon's gradient, start to end. */
    val brand = intArrayOf(0xFFFFB03B.toInt(), 0xFFFF7066.toInt(), 0xFFAB78FF.toInt())

    /** Text and icons on the gradient. */
    val onBrand = 0xFFFFFFFF.toInt()
}
