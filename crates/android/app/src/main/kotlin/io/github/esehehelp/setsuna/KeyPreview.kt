package io.github.esehehelp.setsuna

import android.graphics.Canvas
import android.graphics.ColorFilter
import android.graphics.Paint
import android.graphics.PixelFormat
import android.graphics.RectF
import android.graphics.drawable.Drawable
import android.util.DisplayMetrics
import android.util.TypedValue

/**
 * The bubble over a pressed key, as Gboard draws it: the key's label large,
 * just above the key. Drawn in the input view's overlay, so that it may
 * reach over the candidate bar.
 */
class KeyPreview(private val palette: Palette, private val metrics: DisplayMetrics) : Drawable() {
    private var label: String? = null
    private val box = RectF()
    private val fill = Paint(Paint.ANTI_ALIAS_FLAG)
    private val shadow = Paint(Paint.ANTI_ALIAS_FLAG)
    private val text = Paint(Paint.ANTI_ALIAS_FLAG).apply { textAlign = Paint.Align.CENTER }

    /** Show `label` over `key` (the overlay's coordinates); null hides it. */
    fun show(label: String?, key: RectF?) {
        this.label = label
        if (label != null && key != null) {
            val d = metrics.density
            val w = key.width() + 10 * d
            val h = key.height() * 1.1f
            val left = (key.centerX() - w / 2).coerceIn(2 * d, bounds.width() - w - 2 * d)
            val bottom = key.top + 4 * d
            box.set(left, (bottom - h).coerceAtLeast(0f), left + w, bottom)
        }
        invalidateSelf()
    }

    override fun draw(canvas: Canvas) {
        val l = label ?: return
        val d = metrics.density
        shadow.color = 0x40000000
        canvas.drawRoundRect(box.left, box.top + 2 * d, box.right, box.bottom + 2 * d, 8 * d, 8 * d, shadow)
        fill.color = palette.preview
        canvas.drawRoundRect(box, 8 * d, 8 * d, fill)
        text.color = palette.text
        text.textSize = TypedValue.applyDimension(TypedValue.COMPLEX_UNIT_SP, if (l.length > 1) 22f else 30f, metrics)
        canvas.drawText(l, box.centerX(), box.centerY() - (text.descent() + text.ascent()) / 2, text)
    }

    override fun setAlpha(alpha: Int) {}

    override fun setColorFilter(colorFilter: ColorFilter?) {}

    @Deprecated("Deprecated in Java")
    override fun getOpacity() = PixelFormat.TRANSLUCENT
}
