package io.github.esehehelp.setsuna

import android.annotation.SuppressLint
import android.content.Context
import android.graphics.Canvas
import android.graphics.LinearGradient
import android.graphics.Paint
import android.graphics.RectF
import android.graphics.Shader
import android.os.Handler
import android.os.Looper
import android.util.SparseArray
import android.util.TypedValue
import android.view.HapticFeedbackConstants
import android.view.MotionEvent
import android.view.View
import kotlin.math.abs

/**
 * The keyboard, laid out as Gboard's Japanese QWERTY with its number row
 * (the look is Setsuna's: `Palette`, the gradient Enter key, "Setsuna" on
 * the space key):
 * the あa1 key cycles あ (romaji into setsuna-core), a (letters as they
 * are) and 1 (Gboard's number pad, whose keys flick to symbols); in a, ?123
 * opens the calculator pad, and from there !?# the two pages of symbols.
 * Shift puts a JIS keyboard's symbols on the number row; the globe key is ?
 * while something is composed.
 * The space key's drag moves the caret (the segment while converting).
 */
@SuppressLint("ViewConstructor")
class KeyboardView(context: Context, private val listener: Listener) : View(context) {

    interface Listener {
        /** A key for setsuna-core, as the keysym it types. */
        fun onKey(sym: Int, mods: Int)

        /** Text to put in as it is: a symbol no keysym types, the number pad's. */
        fun onText(text: String)

        /** A mode was chosen: the IME is on in あ, off in a and 1. */
        fun onMode(mode: Mode)

        /** The globe key: the next input method. */
        fun onSwitchInputMethod()

        /** The globe key held: the input method picker. */
        fun onPickInputMethod()

        /** The number pad's undo key. */
        fun onUndo()

        /** The bubble over a pressed key: its label and rect (this view's coordinates); null hides it. */
        fun onPreview(label: String?, rect: RectF?)
    }

    enum class Mode { JA, EN, NUM }

    enum class Page { KEYS, CALC, SYM1, SYM2 }

    enum class Type { CHAR, TEXT, FLICK, COLUMN, SHIFT, BACKSPACE, ENTER, SPACE, MODE, GLOBE, LEFT, RIGHT, CALC, SYM1, SYM2, JA, UNDO, SPACER }

    class Key(
        val label: String,
        val type: Type = Type.CHAR,
        val sym: Int = 0,
        val w: Float = 1f,
        /** TEXT's text; FLICK's without a flick. */
        val text: String = label,
        /** FLICK: what a flick left, up, right, down types ("" for none); COLUMN: its symbols, top first. */
        val more: List<String> = emptyList(),
        /** Rows it spans downwards. */
        val rows: Int = 1,
        val special: Boolean = false,
        val pill: Boolean = false,
        val icon: Int = 0,
        /** CHAR: what it types with Shift (the number row's symbols). */
        val shifted: String = "",
    ) {
        val rect = RectF()
    }

    private class Layout(val rows: List<List<Key>>, val weights: List<Float>)

    private val palette = Palette(context)
    private val density = resources.displayMetrics.density
    private fun sp(v: Float) = TypedValue.applyDimension(TypedValue.COMPLEX_UNIT_SP, v, resources.displayMetrics)

    // ---- the layouts ----

    private fun chars(s: String) = s.map { Key(it.toString(), Type.CHAR, it.code) }

    /**
     * The number row, whose Shift types a JIS keyboard's symbols (the user's
     * PC has one); 0, which has none there, the ~ the number pad's 0 flicks to.
     */
    private fun numberRow() = "1234567890".zip("!\"#\$%&'()~").map { (d, s) ->
        Key(d.toString(), Type.CHAR, d.code, shifted = s.toString())
    }

    /** While something is composed, the globe key types ? (as Gboard's does). */
    private val question = Key("?", Type.CHAR, '?'.code)

    /** A symbol key: by its keysym when it has one (` is the IME's on / off key), else as text. */
    private fun sym(s: String, w: Float = 1f): Key = when {
        s.length == 1 && s[0].code in 0x21..0x7e && s[0] != '`' -> Key(s, Type.CHAR, s[0].code, w)
        s == "¥" -> Key(s, Type.CHAR, Keysym.YEN, w)
        else -> Key(s, Type.TEXT, w = w)
    }

    private fun syms(vararg s: String) = s.map { sym(it) }
    private fun digit(d: String, w: Float) = Key(d, Type.CHAR, d[0].code, w)
    private fun shiftKey() = Key("", Type.SHIFT, w = 1.5f, special = true)
    private fun backspaceKey(w: Float) = Key("", Type.BACKSPACE, Keysym.BACKSPACE, w, special = true, icon = R.drawable.ic_backspace)
    private fun modeKey(w: Float = 1.5f) = Key("あa1", Type.MODE, w = w, special = true, pill = true)
    private fun jaKey() = Key("あいう", Type.JA, w = 1.5f, special = true, pill = true)
    private fun spaceKey(w: Float, label: String = "", special: Boolean = false) =
        Key(label, Type.SPACE, Keysym.SPACE, w, special = special)
    private fun leftKey(w: Float = 1f) = Key("", Type.LEFT, Keysym.LEFT, w, special = true, icon = R.drawable.ic_left)
    private fun rightKey(w: Float = 1f) = Key("", Type.RIGHT, Keysym.RIGHT, w, special = true, icon = R.drawable.ic_right)
    private fun enterKey(w: Float = 1.5f) = Key("", Type.ENTER, Keysym.RETURN, w, special = true, pill = true)
    private fun spacer(w: Float) = Key("", Type.SPACER, w = w)
    private fun flick(label: String, center: String, l: String, u: String, r: String) =
        Key(label, Type.FLICK, w = 2f, text = center, more = listOf(l, u, r, ""))

    /** Gboard's number row is a little lower than the others. */
    private val withNumberRow = listOf(0.93f, 1f, 1f, 1f, 1f)
    private val fourRows = listOf(1f, 1f, 1f, 1f)

    private val ja = Layout(
        listOf(
            numberRow(),
            chars("qwertyuiop"),
            chars("asdfghjkl") + Key("ー", Type.CHAR, '-'.code),
            listOf(shiftKey()) + chars("zxcvbnm") + backspaceKey(1.5f),
            listOf(
                modeKey(),
                Key("、", Type.CHAR, ','.code, special = true),
                Key("", Type.GLOBE, icon = R.drawable.ic_globe),
                spaceKey(2f),
                Key("。", Type.CHAR, '.'.code),
                leftKey(),
                rightKey(),
                enterKey(),
            ),
        ),
        withNumberRow,
    )

    private val en = Layout(
        listOf(
            numberRow(),
            chars("qwertyuiop"),
            chars("asdfghjkl"),
            listOf(shiftKey()) + chars("zxcvbnm") + backspaceKey(1.5f),
            listOf(
                modeKey(),
                Key("?123", Type.CALC, special = true),
                Key(",", Type.CHAR, ','.code),
                spaceKey(2f),
                Key(".", Type.CHAR, '.'.code),
                leftKey(),
                rightKey(),
                enterKey(),
            ),
        ),
        withNumberRow,
    )

    private val num = Layout(
        listOf(
            listOf(
                Key("", Type.UNDO, w = 2f, special = true, icon = R.drawable.ic_undo),
                flick("1", "1", "☆", "♪", "→"),
                flick("2", "2", "¥", "$", "€"),
                flick("3", "3", "%", "°", "#"),
                backspaceKey(2f),
            ),
            listOf(
                leftKey(2f),
                flick("4", "4", "○", "*", "・"),
                flick("5", "5", "+", "×", "÷"),
                flick("6", "6", "<", "=", ">"),
                rightKey(2f),
            ),
            listOf(
                Key("!?#", Type.SYM1, w = 2f, special = true),
                flick("7", "7", "「", "」", ":"),
                flick("8", "8", "〒", "々", "〆"),
                flick("9", "9", "^", "|", "\\"),
                spaceKey(2f, "⎵", special = true),
            ),
            listOf(
                modeKey(2f),
                flick("()[]", "(", ")", "[", "]"),
                flick("0", "0", "~", "…", "@"),
                flick(".,-/", ".", ",", "-", "/"),
                enterKey(2f),
            ),
        ),
        fourRows,
    )

    private val calc = Layout(
        listOf(
            listOf(
                Key("", Type.COLUMN, w = 1.5f, more = listOf("+", "-", "*", "/"), rows = 3, special = true),
                digit("1", 2.333f), digit("2", 2.333f), digit("3", 2.333f),
                sym("%", 1.5f),
            ),
            listOf(spacer(1.5f), digit("4", 2.333f), digit("5", 2.333f), digit("6", 2.333f), spaceKey(1.5f, "⎵", special = true)),
            listOf(spacer(1.5f), digit("7", 2.333f), digit("8", 2.333f), digit("9", 2.333f), backspaceKey(1.5f)),
            listOf(
                jaKey(),
                Key(",", Type.CHAR, ','.code),
                Key("!?#", Type.SYM1, w = 1.25f, special = true),
                digit("0", 2.5f),
                Key("=", Type.CHAR, '='.code, 1.25f),
                Key(".", Type.CHAR, '.'.code),
                enterKey(),
            ),
        ),
        fourRows,
    )

    private val sym1 = Layout(
        listOf(
            chars("1234567890"),
            syms("@", "#", "¥", "%", "&", "-", "+", "(", ")", "/"),
            listOf(Key("=\\<", Type.SYM2, w = 1.5f, special = true)) + syms("*", "\"", "'", ":", ";", "!", "?") + backspaceKey(1.5f),
            listOf(jaKey(), Key(",", Type.CHAR, ','.code), Key("1234", Type.CALC, special = true), spaceKey(4f), Key(".", Type.CHAR, '.'.code), enterKey()),
        ),
        fourRows,
    )

    private val sym2 = Layout(
        listOf(
            syms("~", "`", "|", "•", "√", "π", "÷", "×", "§", "Δ"),
            syms("£", "€", "$", "¢", "^", "°", "=", "{", "}", "\\"),
            listOf(Key("?123", Type.SYM1, w = 1.5f, special = true)) + syms("_", "©", "®", "™", "✓", "[", "]") + backspaceKey(1.5f),
            listOf(jaKey(), sym("<"), Key("1234", Type.CALC, special = true), spaceKey(4f), sym(">"), enterKey()),
        ),
        fourRows,
    )

    private val all = listOf(ja, en, num, calc, sym1, sym2)

    // ---- state ----

    var mode = Mode.JA
        private set
    var page = Page.KEYS
        private set

    private fun layout() = when (page) {
        Page.KEYS -> when (mode) {
            Mode.JA -> ja
            Mode.EN -> en
            Mode.NUM -> num
        }
        Page.CALC -> calc
        Page.SYM1 -> sym1
        Page.SYM2 -> sym2
    }

    /** Something is composed: 変換 and 確定 rather than 日本語 and the field's action. */
    var composing = false
        set(value) {
            if (field != value) {
                field = value
                invalidate()
            }
        }

    /** The Enter key's icon: the field's action (search, send, ...) or a new line. */
    var actionIcon = R.drawable.ic_return
        set(value) {
            field = value
            invalidate()
        }

    private var shift = false
    private var capsLock = false
    private var lastShiftTap = 0L

    private fun choose(m: Mode, p: Page = Page.KEYS) {
        mode = m
        page = p
        shift = false
        capsLock = false
        invalidate()
    }

    /** A new field: あ, or the calculator pad for a number. */
    fun reset(numbers: Boolean) {
        if (numbers) choose(Mode.JA, Page.CALC) else choose(Mode.JA)
    }

    /** Follow the IME turned on or off elsewhere (a hardware key). */
    fun syncEnabled(enabled: Boolean) {
        if (page != Page.KEYS) return
        if (enabled && mode != Mode.JA) choose(Mode.JA)
        if (!enabled && mode == Mode.JA) choose(Mode.EN)
    }

    /** The settings page's キーボード: vibrations, the bubble, the height. */
    var prefs = KeyboardPrefs()
        set(value) {
            val taller = value.height != field.height
            field = value
            if (taller) requestLayout()
        }

    /** A vibration as the settings page says: a tap's and a long press's apart. */
    private fun haptic(feedback: Int) {
        val on = if (feedback == HapticFeedbackConstants.LONG_PRESS) prefs.vibrateHold else prefs.vibrate
        if (on) performHapticFeedback(feedback)
    }

    // ---- geometry ----

    private val keysHeight get() = 266 * density * prefs.height / 100f
    private val padTop = 3 * density
    private val padBottom = 3 * density
    private val hGap = 2.7f * density
    private val vGap = 4.5f * density

    override fun onMeasure(widthMeasureSpec: Int, heightMeasureSpec: Int) {
        setMeasuredDimension(MeasureSpec.getSize(widthMeasureSpec), (keysHeight + padTop + padBottom).toInt())
    }

    override fun onSizeChanged(w: Int, h: Int, oldw: Int, oldh: Int) {
        super.onSizeChanged(w, h, oldw, oldh)
        for (l in all) place(l)
    }

    private fun place(l: Layout) {
        val unit = width / 10f
        val total = l.weights.sum()
        val heights = l.weights.map { it / total * keysHeight }
        var y = padTop
        l.rows.forEachIndexed { r, row ->
            var x = (width - unit * row.sumOf { it.w.toDouble() }.toFloat()) / 2
            for (k in row) {
                val h = heights.subList(r, (r + k.rows).coerceAtMost(heights.size)).sum()
                k.rect.set(x + hGap, y + vGap, x + unit * k.w - hGap, y + h - vGap)
                x += unit * k.w
            }
            y += heights[r]
        }
    }

    private fun keyAt(x: Float, y: Float): Key? {
        val keys = layout().rows.flatten().filter { it.type != Type.SPACER }
        val k = keys.firstOrNull {
            x >= it.rect.left - hGap && x <= it.rect.right + hGap && y >= it.rect.top - vGap && y <= it.rect.bottom + vGap
        } ?: keys.minByOrNull { k ->
            val dx = maxOf(k.rect.left - x, 0f, x - k.rect.right)
            val dy = maxOf(k.rect.top - y, 0f, y - k.rect.bottom)
            dx * dx + dy * dy
        }
        return k?.let { shown(it) }
    }

    /** A key as it is now: the globe is ? while something is composed. */
    private fun shown(k: Key): Key =
        if (k.type == Type.GLOBE && composing) question.also { it.rect.set(k.rect) } else k

    // ---- drawing ----

    private val keyPaint = Paint(Paint.ANTI_ALIAS_FLAG)
    private val brandPaint = Paint(Paint.ANTI_ALIAS_FLAG)
    private val textPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply { textAlign = Paint.Align.CENTER }
    private val radius = 10 * density

    private fun letter(k: Key) =
        k.type == Type.CHAR && k.label.length == 1 && k.label[0] in 'a'..'z'

    /** A number row key with Shift on (caps lock, as on a PC, is for the letters only). */
    private fun symbolNow(k: Key) = k.shifted.isNotEmpty() && shift && !capsLock

    /** What a key shows, when it shows text. */
    private fun label(k: Key): String = when (k.type) {
        Type.CHAR -> when {
            letter(k) && (shift || capsLock) -> k.label.uppercase()
            symbolNow(k) -> k.shifted
            else -> k.label
        }
        Type.SPACE -> k.label.ifEmpty { if (composing && mode == Mode.JA && page == Page.KEYS) "変換" else "Setsuna" }
        Type.ENTER -> if (composing) "確定" else ""
        else -> k.label
    }

    override fun onDraw(canvas: Canvas) {
        canvas.drawColor(palette.background)
        val held = HashSet<Key>()
        for (i in 0 until presses.size()) held.add(presses.valueAt(i).key)
        for (row in layout().rows) {
            for (key in row) {
                if (key.type == Type.SPACER) continue
                val k = shown(key)
                val r = if (k.pill) k.rect.height() / 2 else radius
                if (k.type == Type.ENTER) {
                    // the icon's gradient, darkened while held
                    brandPaint.shader = LinearGradient(
                        k.rect.left, k.rect.top, k.rect.right, k.rect.bottom,
                        palette.brand, null, Shader.TileMode.CLAMP,
                    )
                    canvas.drawRoundRect(k.rect, r, r, brandPaint)
                    if (k in held) {
                        keyPaint.color = 0x33000000
                        canvas.drawRoundRect(k.rect, r, r, keyPaint)
                    }
                } else {
                    keyPaint.color = when {
                        k in held -> palette.pressed
                        k.special -> palette.special
                        else -> palette.key
                    }
                    canvas.drawRoundRect(k.rect, r, r, keyPaint)
                }
                drawFace(canvas, k)
            }
        }
    }

    private fun drawFace(canvas: Canvas, k: Key) {
        when (k.type) {
            Type.MODE -> return drawMode(canvas, k)
            Type.COLUMN -> {
                textPaint.color = palette.text
                textPaint.textSize = sp(18f)
                val n = k.more.size
                k.more.forEachIndexed { i, s ->
                    val cy = k.rect.top + k.rect.height() * (i + 0.5f) / n
                    canvas.drawText(s, k.rect.centerX(), baseline(cy), textPaint)
                }
                return
            }
            Type.FLICK -> {
                textPaint.color = palette.text
                if (k.label.length == 1) {
                    textPaint.textSize = sp(24f)
                    canvas.drawText(k.label, k.rect.centerX(), baseline(k.rect.centerY() - 6 * density), textPaint)
                    textPaint.textSize = sp(11f)
                    textPaint.color = palette.dim
                    val hint = k.more.take(3).joinToString("")
                    canvas.drawText(hint, k.rect.centerX(), baseline(k.rect.centerY() + 14 * density), textPaint)
                } else {
                    textPaint.textSize = sp(20f)
                    canvas.drawText(k.label, k.rect.centerX(), baseline(k.rect.centerY()), textPaint)
                }
                return
            }
            else -> {}
        }
        val icon = when (k.type) {
            Type.SHIFT -> when {
                capsLock -> R.drawable.ic_caps
                shift -> R.drawable.ic_shift_on
                else -> R.drawable.ic_shift
            }
            Type.ENTER -> if (composing) 0 else actionIcon
            else -> k.icon
        }
        if (icon != 0) {
            val d = context.getDrawable(icon) ?: return
            val s = (22 * density).toInt()
            val cx = k.rect.centerX().toInt()
            val cy = k.rect.centerY().toInt()
            d.setBounds(cx - s / 2, cy - s / 2, cx + s / 2, cy + s / 2)
            d.setTint(
                when {
                    k.type == Type.ENTER -> palette.onBrand
                    k.type == Type.SHIFT && (shift || capsLock) -> palette.accent
                    else -> palette.text
                },
            )
            d.draw(canvas)
            return
        }
        val text = label(k)
        textPaint.color = if (k.type == Type.ENTER) palette.onBrand else palette.text
        if (k.type == Type.CALC && text == "1234") {
            // Gboard's 12 over 34
            textPaint.textSize = sp(14f)
            canvas.drawText("12", k.rect.centerX(), baseline(k.rect.centerY() - 8 * density), textPaint)
            canvas.drawText("34", k.rect.centerX(), baseline(k.rect.centerY() + 8 * density), textPaint)
            return
        }
        textPaint.textSize = when {
            text.length == 1 -> sp(23f)
            k.type == Type.SPACE || k.type == Type.ENTER -> sp(15f)
            else -> sp(16f)
        }
        canvas.drawText(text, k.rect.centerX(), baseline(k.rect.centerY()), textPaint)
    }

    /** あa1, the mode in use bright. */
    private fun drawMode(canvas: Canvas, k: Key) {
        val parts = listOf("あ" to Mode.JA, "a" to Mode.EN, "1" to Mode.NUM)
        textPaint.textSize = sp(19f)
        val widths = parts.map { textPaint.measureText(it.first) }
        var x = k.rect.centerX() - widths.sum() / 2
        val y = baseline(k.rect.centerY())
        parts.forEachIndexed { i, (s, m) ->
            textPaint.color = if (m == mode) palette.accent else palette.dim
            canvas.drawText(s, x + widths[i] / 2, y, textPaint)
            x += widths[i]
        }
    }

    private fun baseline(cy: Float) = cy - (textPaint.descent() + textPaint.ascent()) / 2

    // ---- touch ----

    /** A finger on a key: whether its key went out yet, the space drag, the column's symbol. */
    private class Press(val key: Key, val downX: Float, val downY: Float, val index: Int) {
        var fired = false
        var dragged = 0
    }

    private val presses = SparseArray<Press>()
    private val main = Handler(Looper.getMainLooper())
    private var repeating: Key? = null
    private val repeat = object : Runnable {
        override fun run() {
            val k = repeating ?: return
            fire(k, null)
            main.postDelayed(this, 50)
        }
    }

    /** The globe key held long: the input method picker. */
    private var holding: Press? = null
    private val hold = Runnable {
        val p = holding ?: return@Runnable
        p.fired = true
        holding = null
        haptic(HapticFeedbackConstants.LONG_PRESS)
        listener.onPickInputMethod()
    }

    private val flickDistance = 22 * density

    /** -1 for none, else 0 left, 1 up, 2 right, 3 down. */
    private fun direction(p: Press, x: Float, y: Float): Int {
        val dx = x - p.downX
        val dy = y - p.downY
        if (maxOf(abs(dx), abs(dy)) < flickDistance) return -1
        return if (abs(dx) > abs(dy)) (if (dx < 0) 0 else 2) else (if (dy < 0) 1 else 3)
    }

    private fun flicked(k: Key, dir: Int): String =
        if (dir < 0) k.text else k.more.getOrNull(dir)?.ifEmpty { null } ?: k.text

    private fun preview(k: Key, text: String?) {
        val shows = prefs.preview && (k.type == Type.CHAR || k.type == Type.TEXT || k.type == Type.FLICK)
        listener.onPreview(if (shows) text else null, if (shows) k.rect else null)
    }

    @SuppressLint("ClickableViewAccessibility")
    override fun onTouchEvent(e: MotionEvent): Boolean {
        when (e.actionMasked) {
            MotionEvent.ACTION_DOWN, MotionEvent.ACTION_POINTER_DOWN -> {
                val i = e.actionIndex
                val x = e.getX(i)
                val y = e.getY(i)
                val k = keyAt(x, y) ?: return true
                // typing with two thumbs: the key held by the other goes now,
                // so that the keys go out in the order they were touched
                for (j in 0 until presses.size()) {
                    val p = presses.valueAt(j)
                    if (!p.fired && (p.key.type == Type.CHAR || p.key.type == Type.TEXT)) {
                        fire(p.key, p)
                        p.fired = true
                    }
                }
                val index = if (k.type == Type.COLUMN) {
                    ((y - k.rect.top) / k.rect.height() * k.more.size).toInt().coerceIn(0, k.more.size - 1)
                } else {
                    0
                }
                val p = Press(k, x, y, index)
                presses.put(e.getPointerId(i), p)
                haptic(HapticFeedbackConstants.KEYBOARD_TAP)
                preview(k, if (k.type == Type.FLICK) k.text else label(k))
                when (k.type) {
                    Type.BACKSPACE -> {
                        fire(k, p)
                        p.fired = true
                        repeating = k
                        main.postDelayed(repeat, 400)
                    }
                    Type.GLOBE -> {
                        holding = p
                        main.postDelayed(hold, 500)
                    }
                    else -> {}
                }
            }
            MotionEvent.ACTION_MOVE -> {
                for (i in 0 until e.pointerCount) {
                    val p = presses.get(e.getPointerId(i)) ?: continue
                    when (p.key.type) {
                        Type.SPACE -> {
                            // a step of the caret (or the segment) per 18 dp dragged
                            val steps = ((e.getX(i) - p.downX) / (18 * density)).toInt()
                            while (p.dragged < steps) {
                                listener.onKey(Keysym.RIGHT, 0)
                                p.dragged++
                            }
                            while (p.dragged > steps) {
                                listener.onKey(Keysym.LEFT, 0)
                                p.dragged--
                            }
                            if (steps != 0) p.fired = true
                        }
                        Type.FLICK -> preview(p.key, flicked(p.key, direction(p, e.getX(i), e.getY(i))))
                        else -> {}
                    }
                }
            }
            MotionEvent.ACTION_UP, MotionEvent.ACTION_POINTER_UP -> {
                val i = e.actionIndex
                val id = e.getPointerId(i)
                val p = presses.get(id)
                presses.remove(id)
                listener.onPreview(null, null)
                if (p != null) {
                    if (p.key == repeating) {
                        main.removeCallbacks(repeat)
                        repeating = null
                    }
                    if (p == holding) {
                        main.removeCallbacks(hold)
                        holding = null
                    }
                    if (!p.fired && p.dragged == 0) {
                        if (p.key.type == Type.FLICK) {
                            listener.onText(flicked(p.key, direction(p, e.getX(i), e.getY(i))))
                        } else {
                            fire(p.key, p)
                        }
                    }
                }
            }
            MotionEvent.ACTION_CANCEL -> {
                presses.clear()
                listener.onPreview(null, null)
                main.removeCallbacks(repeat)
                repeating = null
                main.removeCallbacks(hold)
                holding = null
            }
        }
        invalidate()
        return true
    }

    private fun fire(k: Key, p: Press?) {
        when (k.type) {
            Type.CHAR -> {
                when {
                    letter(k) && (shift || capsLock) -> listener.onKey(k.sym - 0x20, Native.MOD_SHIFT)
                    symbolNow(k) -> listener.onKey(k.shifted[0].code, Native.MOD_SHIFT)
                    else -> listener.onKey(k.sym, 0)
                }
                if ((letter(k) || k.shifted.isNotEmpty()) && shift && !capsLock) shift = false
            }
            Type.TEXT -> listener.onText(k.text)
            Type.FLICK -> listener.onText(k.text)
            Type.COLUMN -> {
                val s = k.more[p?.index ?: 0]
                val c = s[0].code
                if (s.length == 1 && c in 0x21..0x7e) listener.onKey(c, 0) else listener.onText(s)
            }
            Type.SHIFT -> {
                val now = System.currentTimeMillis()
                when {
                    capsLock -> {
                        capsLock = false
                        shift = false
                    }
                    shift && now - lastShiftTap < 350 -> capsLock = true
                    else -> shift = !shift
                }
                lastShiftTap = now
            }
            Type.BACKSPACE, Type.ENTER, Type.SPACE, Type.LEFT, Type.RIGHT -> listener.onKey(k.sym, 0)
            Type.MODE -> {
                val next = when (mode) {
                    Mode.JA -> Mode.EN
                    Mode.EN -> Mode.NUM
                    Mode.NUM -> Mode.JA
                }
                choose(next)
                listener.onMode(next)
            }
            Type.JA -> {
                choose(Mode.JA)
                listener.onMode(Mode.JA)
            }
            Type.CALC -> page = Page.CALC
            Type.SYM1 -> page = Page.SYM1
            Type.SYM2 -> page = Page.SYM2
            Type.GLOBE -> listener.onSwitchInputMethod()
            Type.UNDO -> listener.onUndo()
            Type.SPACER -> {}
        }
        invalidate()
    }
}
