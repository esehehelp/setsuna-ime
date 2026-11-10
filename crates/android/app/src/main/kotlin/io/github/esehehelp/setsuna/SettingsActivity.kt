package io.github.esehehelp.setsuna

import android.annotation.SuppressLint
import android.app.Activity
import android.content.ActivityNotFoundException
import android.content.ComponentName
import android.content.Intent
import android.content.res.Configuration
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.provider.Settings
import android.view.ViewGroup
import android.view.WindowInsets
import android.view.inputmethod.InputMethodManager
import android.webkit.JavascriptInterface
import android.webkit.ValueCallback
import android.webkit.WebChromeClient
import android.webkit.WebResourceRequest
import android.webkit.WebView
import android.webkit.WebViewClient
import android.widget.FrameLayout
import org.json.JSONObject
import java.io.File

/**
 * The app's one screen (user 2026-11-11: "設定画面だけにしたら？", a first
 * screen of its own having been hidden behind this one): the desktop app's
 * settings page (crates/settings/web/index.html, an asset) in a WebView. Its
 * messages go to [Native.settings], which answers the config's and the user
 * dictionary's as the desktop app does, on the app's own config.toml; the
 * keyboard's (`keyboard_load`, `keyboard_save`) are this activity's
 * ([KeyboardPrefs]), and so are the app's (`app_load`, `app_download`,
 * `app_open_ime`, `app_pick_ime`: whether Setsuna is on, what models/ holds,
 * the downloads of [Downloads]). The page hides what the app does not have
 * (the rewriter, the candidate window, a PC keyboard's keys).
 */
class SettingsActivity : Activity() {
    private lateinit var web: WebView

    /** The page's file input waiting for the document picker. */
    private var picking: ValueCallback<Array<Uri>>? = null

    @SuppressLint("SetJavaScriptEnabled", "JavascriptInterface")
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        actionBar?.hide()
        web = WebView(this)
        web.settings.javaScriptEnabled = true
        web.addJavascriptInterface(Bridge(), "SetsunaAndroid")
        // the page and nothing else: no link leads away from it
        web.webViewClient = object : WebViewClient() {
            override fun shouldOverrideUrlLoading(view: WebView, request: WebResourceRequest) = true
        }
        // the user dictionary's 読み込む…: the system's document picker
        web.webChromeClient = object : WebChromeClient() {
            override fun onShowFileChooser(
                view: WebView,
                callback: ValueCallback<Array<Uri>>,
                params: FileChooserParams,
            ): Boolean {
                picking?.onReceiveValue(null)
                picking = callback
                val intent = Intent(Intent.ACTION_OPEN_DOCUMENT)
                    .addCategory(Intent.CATEGORY_OPENABLE)
                    .setType("*/*")
                return try {
                    startActivityForResult(intent, PICK)
                    true
                } catch (_: ActivityNotFoundException) {
                    picking = null
                    false
                }
            }
        }
        // drawn edge to edge (API 35): kept clear of the bars and of the
        // keyboard, which the page's fields bring up, by a frame (a WebView
        // draws its page under its own padding), in the page's background
        val night = resources.configuration.uiMode and Configuration.UI_MODE_NIGHT_MASK ==
            Configuration.UI_MODE_NIGHT_YES
        val frame = FrameLayout(this).apply {
            setBackgroundColor(if (night) 0xFF202020.toInt() else 0xFFF3F3F3.toInt())
            addView(web, FrameLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.MATCH_PARENT))
        }
        if (Build.VERSION.SDK_INT >= 30) {
            frame.setOnApplyWindowInsetsListener { v, insets ->
                val bars = insets.getInsets(
                    WindowInsets.Type.systemBars() or WindowInsets.Type.displayCutout() or WindowInsets.Type.ime(),
                )
                v.setPadding(bars.left, bars.top, bars.right, bars.bottom)
                insets
            }
        }
        setContentView(frame)
        web.loadUrl("file:///android_asset/index.html")
    }

    @Deprecated("the WebView's file input only")
    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        if (requestCode != PICK) {
            @Suppress("DEPRECATION")
            super.onActivityResult(requestCode, resultCode, data)
            return
        }
        picking?.onReceiveValue(WebChromeClient.FileChooserParams.parseResult(resultCode, data))
        picking = null
    }

    override fun onDestroy() {
        picking?.onReceiveValue(null)
        picking = null
        web.destroy()
        super.onDestroy()
    }

    override fun onResume() {
        super.onResume()
        Downloads.onChange = { pushApp() }
        pushApp()
    }

    override fun onPause() {
        Downloads.onChange = null
        super.onPause()
    }

    override fun onWindowFocusChanged(hasFocus: Boolean) {
        super.onWindowFocusChanged(hasFocus)
        // back from the input method picker or the system's settings
        if (hasFocus) pushApp()
    }

    /** [appState] to the page, from any thread (before the page is up, nothing). */
    private fun pushApp() {
        val state = appState()
        web.post { web.evaluateJavascript("window.setsuna && setsuna.receive($state)", null) }
    }

    /**
     * The app's own state for the page's first section: whether Setsuna is on
     * as an input method and the current one, what models/ holds, how the
     * download goes, and what each download would get (label, MB).
     */
    private fun appState(): String {
        // the input method settings themselves are closed to apps past API 33
        val imm = getSystemService(INPUT_METHOD_SERVICE) as InputMethodManager
        val ours = ComponentName(this, SetsunaIme::class.java)
        val enabled = imm.enabledInputMethodList.any { it.component == ours }
        val current = if (Build.VERSION.SDK_INT >= 34) {
            imm.currentInputMethodInfo?.component == ours
        } else {
            @Suppress("DEPRECATION")
            Settings.Secure.getString(contentResolver, Settings.Secure.DEFAULT_INPUT_METHOD) ==
                ours.flattenToShortString()
        }
        val models = File(filesDir, "models")
        val manifest = File(models, "onnx").listFiles()?.firstOrNull { it.name.endsWith("manifest.json") }
        val lexicon = File(models, "lexicon").listFiles()?.firstOrNull { it.name.endsWith(".lexicon") }
        val analyzer = File(models, "analyzer").listFiles()?.firstOrNull {
            File(it, "analyzer.manifest.json").isFile
        }
        val j = JSONObject()
            .put("type", "app")
            .put("enabled", enabled)
            .put("current", current)
            .put("model", manifest?.name?.removeSuffix(".manifest.json") ?: JSONObject.NULL)
            .put("lexicon", lexicon?.name ?: JSONObject.NULL)
            .put("analyzer", analyzer?.name ?: JSONObject.NULL)
            .put("download", Downloads.state)
            .put("running", Downloads.running)
        for ((key, what) in listOf("get_model" to Downloads.What.MODEL, "get_dicts" to Downloads.What.DICTIONARIES)) {
            runCatching { Downloads.describe(this, what) }.onSuccess { (label, size) ->
                j.put(key, JSONObject().put("label", label).put("mb", Math.round(size / 1048576.0)))
            }
        }
        return j.toString()
    }

    /** The page's `SetsunaAndroid.post`, called on the WebView's bridge thread. */
    private inner class Bridge {
        @JavascriptInterface
        fun post(message: String) {
            val msg = runCatching { JSONObject(message) }.getOrNull() ?: return
            val context = this@SettingsActivity
            val reply = when (msg.optString("type")) {
                "keyboard_load" -> keyboard(KeyboardPrefs.load(context), null)
                "keyboard_save" -> {
                    msg.optJSONObject("keyboard")?.let { KeyboardPrefs.save(context, it) }
                    keyboard(KeyboardPrefs.load(context), "保存しました。次の入力から反映されます。")
                }
                "app_load" -> appState()
                "app_download" -> {
                    val what = if (msg.optString("what") == "dictionaries") {
                        Downloads.What.DICTIONARIES
                    } else {
                        Downloads.What.MODEL
                    }
                    Downloads.start(context, what)
                    appState()
                }
                "app_open_ime" -> {
                    runOnUiThread { startActivity(Intent(Settings.ACTION_INPUT_METHOD_SETTINGS)) }
                    null
                }
                "app_pick_ime" -> {
                    runOnUiThread { (getSystemService(INPUT_METHOD_SERVICE) as InputMethodManager).showInputMethodPicker() }
                    null
                }
                else -> Native.settings(filesDir.path, message)
            } ?: return
            web.post { web.evaluateJavascript("setsuna.receive($reply)", null) }
        }

        private fun keyboard(prefs: KeyboardPrefs, status: String?): String {
            val j = JSONObject().put("type", "keyboard").put("keyboard", prefs.json())
            if (status != null) j.put("status", status)
            return j.toString()
        }
    }

    private companion object {
        const val PICK = 1
    }
}
