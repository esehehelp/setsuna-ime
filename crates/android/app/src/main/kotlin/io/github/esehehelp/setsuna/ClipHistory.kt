package io.github.esehehelp.setsuna

import android.content.ClipboardManager
import android.content.Context

/**
 * What was copied while the keyboard ran, newest first, for the clipboard
 * panel. Kept in memory only and for an hour at most; a clip the app marked
 * sensitive (a password) is not kept.
 */
class ClipHistory(private val context: Context) {
    class Clip(val text: String, val time: Long)

    private val items = ArrayDeque<Clip>()
    private val manager = context.getSystemService(ClipboardManager::class.java)
    private val listener = ClipboardManager.OnPrimaryClipChangedListener { take() }

    /** Called when something changes: the panel and the toolbar redraw. */
    var onChange: () -> Unit = {}

    fun start() {
        manager?.addPrimaryClipChangedListener(listener)
        take()
    }

    fun stop() {
        manager?.removePrimaryClipChangedListener(listener)
    }

    /** The clipboard's text, unless it is sensitive or already the newest. */
    private fun take() {
        // only the default input method and the focused app may read it
        val clip = try {
            manager?.primaryClip
        } catch (e: SecurityException) {
            null
        } ?: return
        if (clip.description?.extras?.getBoolean(EXTRA_IS_SENSITIVE) == true) return
        if (clip.itemCount == 0) return
        val text = clip.getItemAt(0).coerceToText(context)?.toString() ?: return
        if (text.isBlank()) return
        if (items.firstOrNull()?.text == text) return
        items.removeAll { it.text == text }
        items.addFirst(Clip(text, System.currentTimeMillis()))
        while (items.size > MAX) items.removeLast()
        onChange()
    }

    /** The clips of the last hour, newest first. */
    fun list(): List<Clip> {
        val cutoff = System.currentTimeMillis() - KEEP_MS
        items.removeAll { it.time < cutoff }
        return items.toList()
    }

    /** The newest clip if it was copied within `ms`. */
    fun recent(ms: Long): Clip? =
        items.firstOrNull()?.takeIf { System.currentTimeMillis() - it.time < ms }

    fun remove(clip: Clip) {
        items.remove(clip)
        onChange()
    }

    fun clear() {
        items.clear()
        onChange()
    }

    companion object {
        /** ClipDescription.EXTRA_IS_SENSITIVE (API 33), read by its key on older ones too. */
        private const val EXTRA_IS_SENSITIVE = "android.content.extra.IS_SENSITIVE"
        private const val MAX = 30
        private const val KEEP_MS = 60 * 60 * 1000L
    }
}
