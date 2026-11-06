package io.github.esehehelp.setsuna

import android.annotation.SuppressLint
import android.content.Context
import android.graphics.drawable.GradientDrawable
import android.graphics.drawable.StateListDrawable
import android.os.Handler
import android.os.Looper
import android.text.TextUtils
import android.util.TypedValue
import android.view.Gravity
import android.view.HapticFeedbackConstants
import android.view.MotionEvent
import android.view.View
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.TextView

/** A view drawn as a key: rounded, darker while pressed. */
private fun keyLike(context: Context, palette: Palette, special: Boolean): StateListDrawable {
    val r = 10 * context.resources.displayMetrics.density
    fun shape(color: Int) = GradientDrawable().apply {
        cornerRadius = r
        setColor(color)
    }
    return StateListDrawable().apply {
        addState(intArrayOf(android.R.attr.state_pressed), shape(palette.pressed))
        addState(intArrayOf(), shape(if (special) palette.special else palette.key))
    }
}

private fun keyText(context: Context, palette: Palette, label: String, special: Boolean, sp: Float = 15f) =
    TextView(context).apply {
        text = label
        setTextColor(palette.text)
        setTextSize(TypedValue.COMPLEX_UNIT_SP, sp)
        gravity = Gravity.CENTER
        isSingleLine = true
        background = keyLike(context, palette, special)
        isClickable = true
    }

/**
 * Text editing, as Gboard has it: the caret moved by the arrows and to the
 * line's start and end, a selection made with 選択 held on, and select all,
 * cut, copy and paste. Arrows and the delete keys repeat while held.
 */
@SuppressLint("ViewConstructor")
class EditPanel(context: Context, private val onOp: (Op) -> Unit) : LinearLayout(context) {
    enum class Op { LEFT, RIGHT, UP, DOWN, HOME, END, SELECT, SELECT_ALL, CUT, COPY, PASTE, BACKSPACE, DELETE, KEYBOARD }

    private val palette = Palette(context)
    private val density = resources.displayMetrics.density
    private val gap = (3 * density).toInt()
    private val main = Handler(Looper.getMainLooper())
    private lateinit var selectKey: TextView

    /** A vibration on a press ([KeyboardPrefs.vibrate]). */
    var vibrate = true

    /** 選択 is on: the arrows extend the selection. */
    var selecting = false
        set(value) {
            field = value
            selectKey.setTextColor(if (value) palette.accent else palette.text)
        }

    init {
        orientation = VERTICAL
        setBackgroundColor(palette.background)
        setPadding(gap, gap, gap, gap)
        row(key("|◀ 行頭", Op.HOME, true), key("↑", Op.UP, repeat = true), key("行末 ▶|", Op.END, true), key("すべて選択", Op.SELECT_ALL, true))
        selectKey = key("選択", Op.SELECT, false)
        row(key("←", Op.LEFT, repeat = true), selectKey, key("→", Op.RIGHT, repeat = true), key("切り取り", Op.CUT, true))
        row(key("⌫", Op.BACKSPACE, true, repeat = true), key("↓", Op.DOWN, repeat = true), key("削除", Op.DELETE, true, repeat = true), key("コピー", Op.COPY, true))
        val back = key("キーボードに戻る", Op.KEYBOARD, true)
        addView(LinearLayout(context).apply {
            orientation = HORIZONTAL
            addView(back, LayoutParams(0, LayoutParams.MATCH_PARENT, 3f).apply { setMargins(gap, gap, gap, gap) })
            addView(key("貼り付け", Op.PASTE, true), LayoutParams(0, LayoutParams.MATCH_PARENT, 1f).apply { setMargins(gap, gap, gap, gap) })
        }, LayoutParams(LayoutParams.MATCH_PARENT, 0, 1f))
    }

    private fun row(vararg keys: View) {
        addView(LinearLayout(context).apply {
            orientation = HORIZONTAL
            for (k in keys) addView(k, LayoutParams(0, LayoutParams.MATCH_PARENT, 1f).apply { setMargins(gap, gap, gap, gap) })
        }, LayoutParams(LayoutParams.MATCH_PARENT, 0, 1f))
    }

    @SuppressLint("ClickableViewAccessibility")
    private fun key(label: String, op: Op, special: Boolean = false, repeat: Boolean = false): TextView {
        val v = keyText(context, palette, label, special, if (label.length == 1) 22f else 15f)
        if (!repeat) {
            v.setOnClickListener {
                if (vibrate) it.performHapticFeedback(HapticFeedbackConstants.KEYBOARD_TAP)
                onOp(op)
            }
            return v
        }
        val again = object : Runnable {
            override fun run() {
                onOp(op)
                main.postDelayed(this, 60)
            }
        }
        v.setOnTouchListener { view, e ->
            when (e.actionMasked) {
                MotionEvent.ACTION_DOWN -> {
                    view.isPressed = true
                    if (vibrate) view.performHapticFeedback(HapticFeedbackConstants.KEYBOARD_TAP)
                    onOp(op)
                    main.postDelayed(again, 400)
                }
                MotionEvent.ACTION_UP, MotionEvent.ACTION_CANCEL -> {
                    view.isPressed = false
                    main.removeCallbacks(again)
                }
            }
            true
        }
        return v
    }
}

/**
 * What was copied (`ClipHistory`), newest first: a tap pastes it, ✕ drops
 * it. Kept an hour, in memory only.
 */
@SuppressLint("ViewConstructor")
class ClipboardPanel(
    context: Context,
    private val onPaste: (String) -> Unit,
    private val onRemove: (ClipHistory.Clip) -> Unit,
    private val onClear: () -> Unit,
    private val onKeyboard: () -> Unit,
) : LinearLayout(context) {
    private val palette = Palette(context)
    private val density = resources.displayMetrics.density
    private fun dp(v: Int) = (v * density).toInt()
    private val list = LinearLayout(context).apply { orientation = VERTICAL }

    init {
        orientation = VERTICAL
        setBackgroundColor(palette.background)
        val header = LinearLayout(context).apply {
            orientation = HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
            setPadding(dp(12), 0, dp(4), 0)
        }
        header.addView(TextView(context).apply {
            text = "クリップボード（1時間）"
            setTextColor(palette.dim)
            setTextSize(TypedValue.COMPLEX_UNIT_SP, 14f)
        }, LayoutParams(0, LayoutParams.WRAP_CONTENT, 1f))
        header.addView(TextView(context).apply {
            text = "すべて消去"
            setTextColor(palette.text)
            setTextSize(TypedValue.COMPLEX_UNIT_SP, 14f)
            setPadding(dp(12), dp(8), dp(12), dp(8))
            setOnClickListener { onClear() }
        })
        header.addView(ImageView(context).apply {
            setImageResource(R.drawable.ic_keyboard)
            setColorFilter(palette.text)
            contentDescription = "キーボードに戻る"
            setPadding(dp(10), dp(8), dp(10), dp(8))
            setOnClickListener { onKeyboard() }
        }, LayoutParams(dp(48), dp(40)))
        addView(header, LayoutParams(LayoutParams.MATCH_PARENT, dp(40)))
        addView(ScrollView(context).apply { addView(list) }, LayoutParams(LayoutParams.MATCH_PARENT, 0, 1f))
    }

    fun show(clips: List<ClipHistory.Clip>) {
        list.removeAllViews()
        if (clips.isEmpty()) {
            list.addView(TextView(context).apply {
                text = "コピーしたテキストがここに出ます。\n1時間で消え、パスワードなどは残りません。"
                setTextColor(palette.dim)
                setTextSize(TypedValue.COMPLEX_UNIT_SP, 14f)
                setPadding(dp(16), dp(16), dp(16), dp(16))
            })
            return
        }
        for (clip in clips) {
            val row = LinearLayout(context).apply {
                orientation = HORIZONTAL
                gravity = Gravity.CENTER_VERTICAL
                background = keyLike(context, palette, false)
                isClickable = true
                setOnClickListener { onPaste(clip.text) }
            }
            row.addView(TextView(context).apply {
                text = clip.text
                setTextColor(palette.text)
                setTextSize(TypedValue.COMPLEX_UNIT_SP, 15f)
                maxLines = 2
                ellipsize = TextUtils.TruncateAt.END
                setPadding(dp(12), dp(10), dp(4), dp(10))
            }, LayoutParams(0, LayoutParams.WRAP_CONTENT, 1f))
            row.addView(ImageView(context).apply {
                setImageResource(R.drawable.ic_close)
                setColorFilter(palette.dim)
                contentDescription = "消す"
                setPadding(dp(12), dp(12), dp(12), dp(12))
                setOnClickListener { onRemove(clip) }
            }, LayoutParams(dp(44), dp(44)))
            list.addView(row, LayoutParams(LayoutParams.MATCH_PARENT, LayoutParams.WRAP_CONTENT).apply {
                setMargins(dp(6), dp(3), dp(6), dp(3))
            })
        }
    }
}
