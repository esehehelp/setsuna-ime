package io.github.esehehelp.setsuna

import android.content.Context
import org.json.JSONObject

/**
 * The keyboard's own settings, the settings page's キーボード: kept in the
 * app's preferences, apart from config.toml (what setsuna-core reads). The
 * service takes them up when the keyboard comes up for a field.
 */
data class KeyboardPrefs(
    /** A vibration on a key press. */
    val vibrate: Boolean = true,
    /** The bubble over a pressed key. */
    val preview: Boolean = true,
    /** The keys' height, in percent of the usual. */
    val height: Int = 100,
) {
    fun json(): JSONObject = JSONObject()
        .put("vibrate", vibrate)
        .put("preview", preview)
        .put("height", height)

    companion object {
        private const val FILE = "keyboard"
        const val MIN_HEIGHT = 80
        const val MAX_HEIGHT = 130

        fun load(context: Context): KeyboardPrefs {
            val p = context.getSharedPreferences(FILE, Context.MODE_PRIVATE)
            return KeyboardPrefs(
                p.getBoolean("vibrate", true),
                p.getBoolean("preview", true),
                p.getInt("height", 100).coerceIn(MIN_HEIGHT, MAX_HEIGHT),
            )
        }

        /** The page's `{vibrate, preview, height}`. */
        fun save(context: Context, j: JSONObject) {
            context.getSharedPreferences(FILE, Context.MODE_PRIVATE).edit()
                .putBoolean("vibrate", j.optBoolean("vibrate", true))
                .putBoolean("preview", j.optBoolean("preview", true))
                .putInt("height", j.optInt("height", 100).coerceIn(MIN_HEIGHT, MAX_HEIGHT))
                .apply()
        }
    }
}
