package io.github.esehehelp.setsuna

import android.view.KeyEvent

/**
 * X keysyms, which setsuna-core takes keys as (crates/ime-core/src/keysym.rs):
 * a printable ASCII key is its character, the others their X codes.
 */
object Keysym {
    const val BACKSPACE = 0xff08
    const val TAB = 0xff09
    const val RETURN = 0xff0d
    const val ESCAPE = 0xff1b
    const val HOME = 0xff50
    const val LEFT = 0xff51
    const val UP = 0xff52
    const val RIGHT = 0xff53
    const val DOWN = 0xff54
    const val PAGE_UP = 0xff55
    const val PAGE_DOWN = 0xff56
    const val END = 0xff57
    const val F6 = 0xffc3
    const val F7 = 0xffc4
    const val F8 = 0xffc5
    const val F9 = 0xffc6
    const val F10 = 0xffc7
    const val DELETE = 0xffff
    const val SPACE = 0x20
    const val YEN = 0xa5
    const val ZENKAKU_HANKAKU = 0xff2a

    /** A hardware key's keysym, 0 for one the IME has no use for. */
    fun of(event: KeyEvent): Int {
        when (event.keyCode) {
            KeyEvent.KEYCODE_DEL -> return BACKSPACE
            KeyEvent.KEYCODE_FORWARD_DEL -> return DELETE
            KeyEvent.KEYCODE_ENTER, KeyEvent.KEYCODE_NUMPAD_ENTER -> return RETURN
            KeyEvent.KEYCODE_TAB -> return TAB
            KeyEvent.KEYCODE_ESCAPE -> return ESCAPE
            KeyEvent.KEYCODE_DPAD_LEFT -> return LEFT
            KeyEvent.KEYCODE_DPAD_RIGHT -> return RIGHT
            KeyEvent.KEYCODE_DPAD_UP -> return UP
            KeyEvent.KEYCODE_DPAD_DOWN -> return DOWN
            KeyEvent.KEYCODE_MOVE_HOME -> return HOME
            KeyEvent.KEYCODE_MOVE_END -> return END
            KeyEvent.KEYCODE_PAGE_UP -> return PAGE_UP
            KeyEvent.KEYCODE_PAGE_DOWN -> return PAGE_DOWN
            KeyEvent.KEYCODE_F6 -> return F6
            KeyEvent.KEYCODE_F7 -> return F7
            KeyEvent.KEYCODE_F8 -> return F8
            KeyEvent.KEYCODE_F9 -> return F9
            KeyEvent.KEYCODE_F10 -> return F10
            KeyEvent.KEYCODE_SPACE -> return SPACE
            KeyEvent.KEYCODE_ZENKAKU_HANKAKU -> return ZENKAKU_HANKAKU
        }
        // the character without Ctrl / Alt / Meta: Ctrl+U is u with MOD_CTRL
        val meta = event.metaState and
            (KeyEvent.META_CTRL_MASK or KeyEvent.META_ALT_MASK or KeyEvent.META_META_MASK).inv()
        val c = event.getUnicodeChar(meta)
        return when {
            c in 0x21..0x7e -> c
            c == YEN -> YEN
            else -> 0
        }
    }

    /** A hardware key's modifiers as `Native.MOD_*` bits. */
    fun mods(event: KeyEvent): Int {
        var m = 0
        if (event.isShiftPressed) m = m or Native.MOD_SHIFT
        if (event.isCtrlPressed) m = m or Native.MOD_CTRL
        if (event.isAltPressed) m = m or Native.MOD_ALT
        if (event.isMetaPressed) m = m or Native.MOD_SUPER
        return m
    }
}
