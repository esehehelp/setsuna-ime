package io.github.esehehelp.setsuna

import android.graphics.Rect
import android.graphics.RectF
import android.graphics.Typeface
import android.inputmethodservice.InputMethodService
import android.os.Build
import android.os.Handler
import android.os.Looper
import android.os.SystemClock
import android.text.InputType
import android.text.SpannableString
import android.text.Spanned
import android.text.style.BackgroundColorSpan
import android.text.style.StyleSpan
import android.text.style.UnderlineSpan
import android.view.KeyEvent
import android.view.View
import android.view.ViewGroup
import android.view.WindowInsets
import android.view.inputmethod.EditorInfo
import android.view.inputmethod.InputMethodManager
import android.widget.FrameLayout
import android.widget.LinearLayout
import org.json.JSONArray
import org.json.JSONObject

/**
 * The input method service: the keyboard and the candidate bar, and the
 * app's text. Each key goes to setsuna-core ([Native.key]) as its keysym;
 * the view that comes back is written into the app: the text to commit,
 * then the preedit as the composing text. A key the IME leaves goes to the
 * app as it is. The engine runs in a child process, started when a field is
 * focused; its results wake the main thread ([onEngineNotify]).
 */
class SetsunaIme : InputMethodService(), Native.Listener, KeyboardView.Listener {
    private var h = 0L
    private val main = Handler(Looper.getMainLooper())
    private var keyboard: KeyboardView? = null
    private var bar: CandidateBar? = null
    private var palette: Palette? = null

    /** The focused field's kind (`Native.FIELD_*`). */
    private var field = Native.FIELD_NORMAL
    private var editorInfo: EditorInfo? = null

    /** The app holds a composing text of ours. */
    private var composing = false

    /** Where it starts in the app's text, as the app last reported. */
    private var composeStart = -1
    private var selStart = 0

    /** Hardware keys whose down the IME took: their ups are its too. */
    private val heldKeys = HashSet<Int>()

    private val tick = Runnable {
        if (h != 0L && Native.tick(h)) render() else scheduleTick()
    }

    /** The IME is on (the session's view says). */
    private var enabled = true
    private var root: LinearLayout? = null
    private var preview: KeyPreview? = null

    private lateinit var clips: ClipHistory
    private var clipPanel: ClipboardPanel? = null
    private var editPanel: EditPanel? = null

    /** The panel in place of the keys, null for the keys. */
    private var open: CandidateBar.Tool? = null

    /** How long a copy stays on the toolbar's chip. */
    private val recentMs = 60_000L
    private val refreshRecent = Runnable { showRecent() }

    override fun onCreate() {
        super.onCreate()
        Assets.unpack(this)
        // the models are in files/models too (push-models.ps1)
        h = Native.create(filesDir.path, applicationInfo.nativeLibraryDir, filesDir.path, this)
        clips = ClipHistory(this)
        clips.onChange = {
            if (open == CandidateBar.Tool.CLIPBOARD) clipPanel?.show(clips.list())
            showRecent()
        }
        clips.start()
    }

    override fun onDestroy() {
        main.removeCallbacksAndMessages(null)
        clips.stop()
        if (h != 0L) Native.destroy(h)
        h = 0L
        super.onDestroy()
    }

    override fun onEngineNotify() {
        main.post { if (h != 0L && Native.applyAsync(h)) render() }
    }

    override fun onCreateInputView(): View {
        palette = Palette(this)
        val density = resources.displayMetrics.density
        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setBackgroundColor(Palette(context).background)
        }
        val bar = CandidateBar(this, { i -> pick(i) }, ::onTool)
        val keyboard = KeyboardView(this, this)
        // the panels over the keys, as tall as they are
        val stack = FrameLayout(this)
        stack.addView(keyboard, FrameLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT))
        val clipPanel = ClipboardPanel(
            this,
            onPaste = { paste(it) },
            onRemove = { clips.remove(it) },
            onClear = { clips.clear() },
            onKeyboard = { showPanel(null) },
        )
        val editPanel = EditPanel(this, ::onEdit)
        for (p in listOf(clipPanel, editPanel)) {
            p.visibility = View.GONE
            stack.addView(p, FrameLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.MATCH_PARENT))
        }
        root.addView(bar, LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, (46 * density).toInt()))
        root.addView(stack, LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT))
        this.clipPanel = clipPanel
        this.editPanel = editPanel
        // the pressed key's bubble, over the keys and the bar
        val preview = KeyPreview(Palette(this), resources.displayMetrics)
        root.overlay.add(preview)
        root.addOnLayoutChangeListener { v, _, _, _, _, _, _, _, _ -> preview.setBounds(0, 0, v.width, v.height) }
        this.root = root
        this.preview = preview
        // the keys above the navigation bar, which the window reaches under
        if (Build.VERSION.SDK_INT >= 30) {
            root.setOnApplyWindowInsetsListener { v, insets ->
                val nav = insets.getInsets(WindowInsets.Type.navigationBars())
                v.setPadding(nav.left, 0, nav.right, nav.bottom)
                insets
            }
        }
        this.bar = bar
        this.keyboard = keyboard
        return root
    }

    override fun onStartInput(attribute: EditorInfo, restarting: Boolean) {
        super.onStartInput(attribute, restarting)
        // a composition of the field before stays as it is shown
        if (composing) currentInputConnection?.finishComposingText()
        composing = false
        composeStart = -1
        selStart = attribute.initialSelStart.coerceAtLeast(0)
        editorInfo = attribute
        field = fieldOf(attribute)
        if (h != 0L) Native.configure(h, attribute.packageName ?: "")
    }

    override fun onStartInputView(info: EditorInfo, restarting: Boolean) {
        super.onStartInputView(info, restarting)
        val cls = info.inputType and InputType.TYPE_MASK_CLASS
        keyboard?.reset(
            cls == InputType.TYPE_CLASS_NUMBER || cls == InputType.TYPE_CLASS_PHONE ||
                cls == InputType.TYPE_CLASS_DATETIME,
        )
        keyboard?.actionIcon = actionIcon(info)
        // the settings page's キーボード, as last saved
        val prefs = KeyboardPrefs.load(this)
        keyboard?.prefs = prefs
        editPanel?.vibrate = prefs.vibrate
        showPanel(null)
        render()
    }

    // ---- the toolbar and its panels ----

    private fun onTool(tool: CandidateBar.Tool) {
        when (tool) {
            CandidateBar.Tool.CLIPBOARD, CandidateBar.Tool.EDIT ->
                showPanel(if (open == tool) null else tool)
            CandidateBar.Tool.PASTE_RECENT -> clips.recent(recentMs)?.let { paste(it.text) }
            CandidateBar.Tool.SETTINGS -> {
                showPanel(null)
                startActivity(
                    android.content.Intent(this, SettingsActivity::class.java)
                        .addFlags(android.content.Intent.FLAG_ACTIVITY_NEW_TASK),
                )
            }
        }
    }

    /** `tool`'s panel in place of the keys; null for the keys. */
    private fun showPanel(tool: CandidateBar.Tool?) {
        if (tool != null) commitComposition()
        open = tool
        keyboard?.visibility = if (tool == null) View.VISIBLE else View.INVISIBLE
        clipPanel?.visibility = if (tool == CandidateBar.Tool.CLIPBOARD) View.VISIBLE else View.GONE
        editPanel?.visibility = if (tool == CandidateBar.Tool.EDIT) View.VISIBLE else View.GONE
        if (tool == CandidateBar.Tool.CLIPBOARD) clipPanel?.show(clips.list())
        if (tool != CandidateBar.Tool.EDIT) editPanel?.selecting = false
        bar?.setOpen(tool)
        showRecent()
    }

    /** The chip for what was just copied, while it is recent. */
    private fun showRecent() {
        main.removeCallbacks(refreshRecent)
        val recent = if (open == null) clips.recent(recentMs) else null
        bar?.setRecent(recent?.text)
        if (recent != null) {
            main.postDelayed(refreshRecent, recentMs - (System.currentTimeMillis() - recent.time) + 50)
        }
    }

    /** What is composed goes in as it stands, before text from elsewhere. */
    private fun commitComposition() {
        if (composing && h != 0L) {
            Native.flush(h)
            render()
        }
    }

    private fun paste(text: String) {
        commitComposition()
        currentInputConnection?.commitText(text, 1)
    }

    private fun onEdit(op: EditPanel.Op) {
        commitComposition()
        val ic = currentInputConnection ?: return
        val selecting = editPanel?.selecting == true
        when (op) {
            EditPanel.Op.LEFT -> move(KeyEvent.KEYCODE_DPAD_LEFT, selecting)
            EditPanel.Op.RIGHT -> move(KeyEvent.KEYCODE_DPAD_RIGHT, selecting)
            EditPanel.Op.UP -> move(KeyEvent.KEYCODE_DPAD_UP, selecting)
            EditPanel.Op.DOWN -> move(KeyEvent.KEYCODE_DPAD_DOWN, selecting)
            EditPanel.Op.HOME -> move(KeyEvent.KEYCODE_MOVE_HOME, selecting)
            EditPanel.Op.END -> move(KeyEvent.KEYCODE_MOVE_END, selecting)
            EditPanel.Op.SELECT -> editPanel?.selecting = !selecting
            EditPanel.Op.SELECT_ALL -> ic.performContextMenuAction(android.R.id.selectAll)
            EditPanel.Op.CUT -> {
                ic.performContextMenuAction(android.R.id.cut)
                editPanel?.selecting = false
            }
            EditPanel.Op.COPY -> {
                ic.performContextMenuAction(android.R.id.copy)
                editPanel?.selecting = false
            }
            EditPanel.Op.PASTE -> ic.performContextMenuAction(android.R.id.paste)
            EditPanel.Op.BACKSPACE -> sendDownUpKeyEvents(KeyEvent.KEYCODE_DEL)
            EditPanel.Op.DELETE -> sendDownUpKeyEvents(KeyEvent.KEYCODE_FORWARD_DEL)
            EditPanel.Op.KEYBOARD -> showPanel(null)
        }
    }

    /**
     * A caret key; while selecting, inside a press of Shift, which extends
     * the selection (the editor takes Shift from its own key state, not
     * from the arrow's meta state alone).
     */
    private fun move(code: Int, shift: Boolean) {
        if (!shift) {
            sendDownUpKeyEvents(code)
            return
        }
        val ic = currentInputConnection ?: return
        val now = SystemClock.uptimeMillis()
        val meta = KeyEvent.META_SHIFT_ON or KeyEvent.META_SHIFT_LEFT_ON
        ic.sendKeyEvent(KeyEvent(now, now, KeyEvent.ACTION_DOWN, KeyEvent.KEYCODE_SHIFT_LEFT, 0, meta))
        ic.sendKeyEvent(KeyEvent(now, now, KeyEvent.ACTION_DOWN, code, 0, meta))
        ic.sendKeyEvent(KeyEvent(now, now, KeyEvent.ACTION_UP, code, 0, meta))
        ic.sendKeyEvent(KeyEvent(now, now, KeyEvent.ACTION_UP, KeyEvent.KEYCODE_SHIFT_LEFT, 0, 0))
    }

    override fun onFinishInput() {
        // the field loses focus: what is composed is committed
        commitComposition()
        super.onFinishInput()
    }

    override fun onFinishInputView(finishingInput: Boolean) {
        // the keyboard goes away (Back, the field's own hide): so does the
        // composition, which the app ends too, keeping its text; left in
        // the session, it was written again when the keyboard came back
        commitComposition()
        super.onFinishInputView(finishingInput)
    }

    override fun onUpdateSelection(
        oldSelStart: Int,
        oldSelEnd: Int,
        newSelStart: Int,
        newSelEnd: Int,
        candidatesStart: Int,
        candidatesEnd: Int,
    ) {
        super.onUpdateSelection(oldSelStart, oldSelEnd, newSelStart, newSelEnd, candidatesStart, candidatesEnd)
        selStart = newSelStart
        if (candidatesStart >= 0) composeStart = candidatesStart
        // the user put the caret elsewhere: what is composed is committed
        // (a report of the time before the composition, without one, is not
        // taken for that: the app reports edits late)
        if (composing && candidatesStart >= 0 &&
            (newSelStart < candidatesStart || newSelStart > candidatesEnd || newSelEnd != newSelStart)
        ) {
            if (h != 0L) {
                Native.flush(h)
                render()
            }
        }
    }

    // ---- keys ----

    override fun onKey(sym: Int, mods: Int) {
        if (!take(sym, mods)) sendToApp(sym)
    }

    override fun onText(text: String) {
        commitComposition()
        currentInputConnection?.commitText(text, 1)
    }

    /** The IME on in あ, off (keys as they are) in a and 1. */
    override fun onMode(mode: KeyboardView.Mode) {
        if (h == 0L) return
        val on = mode == KeyboardView.Mode.JA
        if (enabled != on) {
            Native.toggle(h)
            render()
        }
    }

    override fun onSwitchInputMethod() {
        if (Build.VERSION.SDK_INT >= 28 && shouldOfferSwitchingToNextInputMethod()) {
            switchToNextInputMethod(false)
        } else {
            onPickInputMethod()
        }
    }

    override fun onPickInputMethod() {
        (getSystemService(INPUT_METHOD_SERVICE) as InputMethodManager).showInputMethodPicker()
    }

    /** Ctrl+Z, which the app's editor takes for undo. */
    override fun onUndo() {
        commitComposition()
        val ic = currentInputConnection ?: return
        val now = SystemClock.uptimeMillis()
        val meta = KeyEvent.META_CTRL_ON or KeyEvent.META_CTRL_LEFT_ON
        ic.sendKeyEvent(KeyEvent(now, now, KeyEvent.ACTION_DOWN, KeyEvent.KEYCODE_Z, 0, meta))
        ic.sendKeyEvent(KeyEvent(now, now, KeyEvent.ACTION_UP, KeyEvent.KEYCODE_Z, 0, meta))
    }

    override fun onPreview(label: String?, rect: RectF?) {
        val p = preview ?: return
        val k = keyboard ?: return
        val r = root ?: return
        if (label == null || rect == null) {
            p.show(null, null)
            return
        }
        // the key's rect in the input view's coordinates
        val box = Rect()
        rect.roundOut(box)
        r.offsetDescendantRectToMyCoords(k, box)
        p.show(label, RectF(box))
    }

    override fun onKeyDown(keyCode: Int, event: KeyEvent): Boolean {
        val sym = Keysym.of(event)
        if (sym != 0 && take(sym, Keysym.mods(event))) {
            heldKeys.add(keyCode)
            return true
        }
        return super.onKeyDown(keyCode, event)
    }

    override fun onKeyUp(keyCode: Int, event: KeyEvent): Boolean {
        if (heldKeys.remove(keyCode)) return true
        return super.onKeyUp(keyCode, event)
    }

    /** Hand the key to setsuna-core; true when it took it. */
    private fun take(sym: Int, mods: Int): Boolean {
        if (h == 0L) return false
        // the left context, read when a composition may start
        val before = if (!composing && field == Native.FIELD_NORMAL) {
            currentInputConnection?.getTextBeforeCursor(64, 0)?.toString()
        } else {
            null
        }
        val took = Native.key(h, sym, mods, field, before)
        if (took) render()
        return took
    }

    private fun sendToApp(sym: Int) {
        when (sym) {
            Keysym.BACKSPACE -> sendDownUpKeyEvents(KeyEvent.KEYCODE_DEL)
            Keysym.RETURN -> enter()
            Keysym.LEFT -> sendDownUpKeyEvents(KeyEvent.KEYCODE_DPAD_LEFT)
            Keysym.RIGHT -> sendDownUpKeyEvents(KeyEvent.KEYCODE_DPAD_RIGHT)
            Keysym.UP -> sendDownUpKeyEvents(KeyEvent.KEYCODE_DPAD_UP)
            Keysym.DOWN -> sendDownUpKeyEvents(KeyEvent.KEYCODE_DPAD_DOWN)
            Keysym.YEN -> currentInputConnection?.commitText("¥", 1)
            else -> if (sym in 0x20..0x7e) sendKeyChar(sym.toChar())
        }
    }

    /** Enter in the app: the field's action (search, send, ...) or a new line. */
    private fun enter() {
        val info = editorInfo
        val action = info?.let { it.imeOptions and EditorInfo.IME_MASK_ACTION } ?: EditorInfo.IME_ACTION_NONE
        val noAction = info != null && (info.imeOptions and EditorInfo.IME_FLAG_NO_ENTER_ACTION) != 0
        val multiLine = info != null && (info.inputType and InputType.TYPE_TEXT_FLAG_MULTI_LINE) != 0
        if (!noAction && !multiLine && action != EditorInfo.IME_ACTION_NONE &&
            action != EditorInfo.IME_ACTION_UNSPECIFIED
        ) {
            currentInputConnection?.performEditorAction(action)
        } else {
            sendKeyChar('\n')
        }
    }

    private fun pick(index: Int) {
        if (h != 0L && Native.select(h, index)) render()
    }

    // ---- the view ----

    private fun strings(a: JSONArray?): List<String> =
        if (a == null) emptyList() else List(a.length()) { a.getString(it) }

    /** Write the session's view into the app and onto the keyboard. */
    private fun render() {
        if (h == 0L) return
        val v = JSONObject(Native.view(h) ?: return)
        val commits = strings(v.optJSONArray("commits"))
        val segments = strings(v.optJSONArray("preedit"))
        val ic = currentInputConnection
        if (ic != null) {
            ic.beginBatchEdit()
            var start = if (composing && composeStart >= 0) composeStart else selStart
            for (c in commits) {
                ic.commitText(c, 1)
                start += c.length
            }
            if (segments.isNotEmpty()) {
                val text = segments.joinToString("")
                ic.setComposingText(preedit(segments, v.optInt("focus", -1)), 1)
                if (!composing || commits.isNotEmpty()) composeStart = start
                composing = true
                val cursor = v.optInt("cursor", text.length)
                if (cursor < text.length && composeStart >= 0) {
                    ic.setSelection(composeStart + cursor, composeStart + cursor)
                }
            } else {
                if (composing && commits.isEmpty()) ic.setComposingText("", 1)
                ic.finishComposingText()
                composing = false
                composeStart = -1
            }
            ic.endBatchEdit()
        }

        val candidates = strings(v.optJSONArray("candidates"))
        val engine = v.optBoolean("engine", false)
        val note = when {
            v.optString("status").isNotEmpty() -> v.optString("status")
            !engine && v.optBoolean("enabled", true) -> getString(R.string.no_engine)
            else -> ""
        }
        if (candidates.isNotEmpty()) {
            bar?.show(candidates, v.optInt("selected", 0), note)
        } else {
            bar?.show(strings(v.optJSONArray("suggestions")), -1, note)
        }
        enabled = v.optBoolean("enabled", true)
        keyboard?.syncEnabled(enabled)
        keyboard?.composing = composing
        scheduleTick()
    }

    /** The preedit: underlined, the segment being edited marked. */
    private fun preedit(segments: List<String>, focus: Int): CharSequence {
        val text = segments.joinToString("")
        val s = SpannableString(text)
        s.setSpan(UnderlineSpan(), 0, text.length, Spanned.SPAN_COMPOSING or Spanned.SPAN_EXCLUSIVE_EXCLUSIVE)
        if (focus in segments.indices && segments.size > 1) {
            val from = segments.take(focus).sumOf { it.length }
            val to = from + segments[focus].length
            s.setSpan(BackgroundColorSpan(palette?.accentSoft ?: 0x33FF7066), from, to, Spanned.SPAN_EXCLUSIVE_EXCLUSIVE)
            s.setSpan(StyleSpan(Typeface.BOLD), from, to, Spanned.SPAN_EXCLUSIVE_EXCLUSIVE)
        }
        return s
    }

    /** A deferred preedit shows anyway at its deadline. */
    private fun scheduleTick() {
        main.removeCallbacks(tick)
        val ms = if (h != 0L) Native.deferredMs(h) else 0
        if (ms > 0) main.postDelayed(tick, ms.toLong())
    }

    /** The Enter key's icon for the field's action, as Gboard has them. */
    private fun actionIcon(info: EditorInfo): Int {
        if ((info.imeOptions and EditorInfo.IME_FLAG_NO_ENTER_ACTION) != 0) return R.drawable.ic_return
        if ((info.inputType and InputType.TYPE_TEXT_FLAG_MULTI_LINE) != 0) return R.drawable.ic_return
        return when (info.imeOptions and EditorInfo.IME_MASK_ACTION) {
            EditorInfo.IME_ACTION_GO, EditorInfo.IME_ACTION_NEXT -> R.drawable.ic_next
            EditorInfo.IME_ACTION_SEARCH -> R.drawable.ic_search
            EditorInfo.IME_ACTION_SEND -> R.drawable.ic_send
            EditorInfo.IME_ACTION_DONE -> R.drawable.ic_done
            else -> R.drawable.ic_return
        }
    }

    companion object {
        /**
         * What a field is to the IME: a password, a URL, an e-mail address
         * or a number takes keys as they are (as the input scopes do on
         * Windows); one that asks for no learning gets none.
         */
        fun fieldOf(info: EditorInfo): Int {
            val cls = info.inputType and InputType.TYPE_MASK_CLASS
            val variation = info.inputType and InputType.TYPE_MASK_VARIATION
            if (cls == InputType.TYPE_NULL) return Native.FIELD_DIRECT
            if (cls == InputType.TYPE_CLASS_NUMBER || cls == InputType.TYPE_CLASS_PHONE ||
                cls == InputType.TYPE_CLASS_DATETIME
            ) {
                return Native.FIELD_DIRECT
            }
            if (cls == InputType.TYPE_CLASS_TEXT && variation in setOf(
                    InputType.TYPE_TEXT_VARIATION_PASSWORD,
                    InputType.TYPE_TEXT_VARIATION_VISIBLE_PASSWORD,
                    InputType.TYPE_TEXT_VARIATION_WEB_PASSWORD,
                    InputType.TYPE_TEXT_VARIATION_URI,
                    InputType.TYPE_TEXT_VARIATION_EMAIL_ADDRESS,
                    InputType.TYPE_TEXT_VARIATION_WEB_EMAIL_ADDRESS,
                )
            ) {
                return Native.FIELD_DIRECT
            }
            if ((info.imeOptions and EditorInfo.IME_FLAG_NO_PERSONALIZED_LEARNING) != 0) {
                return Native.FIELD_PRIVATE
            }
            return Native.FIELD_NORMAL
        }
    }
}
