package io.github.esehehelp.setsuna

import android.annotation.SuppressLint
import android.content.Context
import android.graphics.Canvas
import android.graphics.LinearGradient
import android.graphics.Paint
import android.graphics.Shader
import android.graphics.drawable.GradientDrawable
import android.text.TextUtils
import android.util.TypedValue
import android.view.Gravity
import android.view.View
import android.widget.FrameLayout
import android.widget.HorizontalScrollView
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.TextView

/**
 * The strip above the keys. While converting, the open candidate list (the
 * one picked highlighted); while typing, the conversions of what is typed;
 * a line of status when there is one. Otherwise the toolbar, its icons where
 * Gboard has its own: text editing (Gboard's menu), the clipboard, a chip
 * that pastes what was just copied (Gboard's proofreading), the settings.
 */
@SuppressLint("ViewConstructor")
class CandidateBar(
    context: Context,
    private val onPick: (Int) -> Unit,
    private val onTool: (Tool) -> Unit,
) : FrameLayout(context) {

    enum class Tool { EDIT, CLIPBOARD, PASTE_RECENT, SETTINGS }

    private val palette = Palette(context)
    private val density = resources.displayMetrics.density
    private fun dp(v: Float) = (v * density).toInt()

    private val scroll = HorizontalScrollView(context).apply { isHorizontalScrollBarEnabled = false }
    private val row = LinearLayout(context).apply { orientation = LinearLayout.HORIZONTAL }
    private val tools = FrameLayout(context)
    private val recentChip = TextView(context).apply {
        setTextColor(palette.text)
        setTextSize(TypedValue.COMPLEX_UNIT_SP, 14f)
        isSingleLine = true
        ellipsize = TextUtils.TruncateAt.END
        gravity = Gravity.CENTER_VERTICAL
        setPadding(dp(12f), 0, dp(12f), 0)
        background = GradientDrawable().apply {
            cornerRadius = dp(16f).toFloat()
            setColor(palette.special)
        }
        setOnClickListener { onTool(Tool.PASTE_RECENT) }
        visibility = View.GONE
    }
    private val toolViews = HashMap<Tool, ImageView>()

    /** Gboard's icons: 44 dp wide, one every 91.5 dp from the left edge. */
    private val pitch = 91.5f

    init {
        setBackgroundColor(palette.background)
        scroll.addView(row, LayoutParams(LayoutParams.WRAP_CONTENT, LayoutParams.MATCH_PARENT))
        addView(scroll, LayoutParams(LayoutParams.MATCH_PARENT, LayoutParams.MATCH_PARENT))
        tools.addView(icon(Tool.EDIT, R.drawable.ic_edit_text, "テキスト編集"), at(0))
        tools.addView(icon(Tool.CLIPBOARD, R.drawable.ic_clipboard, "クリップボード"), at(1))
        tools.addView(
            recentChip,
            LayoutParams(dp(pitch * 2 - 44f - 8f), dp(32f), Gravity.CENTER_VERTICAL).apply {
                leftMargin = dp(pitch + 44f + 4f)
            },
        )
        tools.addView(icon(Tool.SETTINGS, R.drawable.ic_settings, "設定"), at(3))
        addView(tools, LayoutParams(LayoutParams.MATCH_PARENT, LayoutParams.MATCH_PARENT))
        scroll.visibility = View.GONE
    }

    private fun at(slot: Int) =
        LayoutParams(dp(44f), LayoutParams.MATCH_PARENT).apply { leftMargin = dp(pitch * slot) }

    private fun icon(tool: Tool, res: Int, label: String) = ImageView(context).apply {
        setImageResource(res)
        setColorFilter(palette.text)
        contentDescription = label
        val pad = dp(10f)
        setPadding(pad, pad, pad, pad)
        setOnClickListener { onTool(tool) }
        toolViews[tool] = this
    }

    /** The tool whose panel is open, drawn lit; null for none. */
    fun setOpen(tool: Tool?) {
        for ((t, v) in toolViews) v.setColorFilter(if (t == tool) palette.accent else palette.text)
    }

    /** What was just copied, on the chip that pastes it; null for none. */
    fun setRecent(text: String?) {
        if (text == null) {
            recentChip.visibility = View.GONE
        } else {
            recentChip.text = text.replace('\n', ' ')
            recentChip.visibility = View.VISIBLE
        }
    }

    private var shown: List<String> = emptyList()
    private var shownSelected = -1
    private var shownNote = ""

    /**
     * Show `items`, `selected` highlighted when it is >= 0; `note` alone
     * when there are none; the toolbar when there is neither.
     */
    fun show(items: List<String>, selected: Int, note: String) {
        val toolbar = items.isEmpty() && note.isEmpty()
        tools.visibility = if (toolbar) View.VISIBLE else View.GONE
        scroll.visibility = if (toolbar) View.GONE else View.VISIBLE
        if (items == shown && selected == shownSelected && note == shownNote) return
        shown = items
        shownSelected = selected
        shownNote = note
        row.removeAllViews()
        if (items.isEmpty()) {
            if (note.isNotEmpty()) row.addView(cell(note, false, palette.dim), cellParams())
            scroll.scrollTo(0, 0)
            return
        }
        items.forEachIndexed { i, s ->
            val tv = cell(s, i == selected, palette.text)
            tv.setOnClickListener { onPick(i) }
            row.addView(tv, cellParams())
        }
        if (selected >= 0) {
            post {
                val child = row.getChildAt(selected) ?: return@post
                if (child.right > scroll.scrollX + width || child.left < scroll.scrollX) {
                    scroll.smoothScrollTo((child.left - width / 3).coerceAtLeast(0), 0)
                }
            }
        } else {
            scroll.scrollTo(0, 0)
        }
    }

    private fun cellParams() =
        LinearLayout.LayoutParams(LinearLayout.LayoutParams.WRAP_CONTENT, LinearLayout.LayoutParams.MATCH_PARENT)

    private fun cell(text: String, picked: Boolean, color: Int) = TextView(context).apply {
        this.text = text
        setTextColor(color)
        setTextSize(TypedValue.COMPLEX_UNIT_SP, 18f)
        gravity = Gravity.CENTER
        isSingleLine = true
        setPadding(dp(14f), 0, dp(14f), 0)
        if (picked) {
            background = GradientDrawable().apply {
                cornerRadius = dp(10f).toFloat()
                setColor(palette.accentSoft)
            }
        }
    }

    // ---- the stripe along the top: Setsuna's at a glance ----

    private val stripe = Paint(Paint.ANTI_ALIAS_FLAG)
    private val stripeHeight = 2.5f * density

    override fun onSizeChanged(w: Int, h: Int, oldw: Int, oldh: Int) {
        super.onSizeChanged(w, h, oldw, oldh)
        stripe.shader = LinearGradient(0f, 0f, w.toFloat(), 0f, palette.brand, null, Shader.TileMode.CLAMP)
    }

    override fun dispatchDraw(canvas: Canvas) {
        super.dispatchDraw(canvas)
        canvas.drawRect(0f, 0f, width.toFloat(), stripeHeight, stripe)
    }
}
