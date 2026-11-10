package io.github.esehehelp.setsuna

import android.app.Activity
import android.content.ComponentName
import android.content.Intent
import android.os.Build
import android.os.Bundle
import android.provider.Settings
import android.util.TypedValue
import android.view.ViewGroup
import android.view.WindowInsets
import android.view.inputmethod.InputMethodManager
import android.widget.Button
import android.widget.EditText
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.TextView
import java.io.File

/**
 * The app's first screen: whether Setsuna is on as an input method and has
 * a model, the downloads of the model and of the dictionaries ([Downloads]),
 * its settings ([SettingsActivity]), the way to turn it on, and a field to
 * try it in.
 */
class MainActivity : Activity() {
    private lateinit var status: TextView
    private lateinit var modelButton: Button
    private lateinit var dictionariesButton: Button

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val pad = (16 * resources.displayMetrics.density).toInt()
        val column = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(pad, pad, pad, pad)
        }
        column.addView(TextView(this).apply {
            text = getString(R.string.app_name)
            setTextSize(TypedValue.COMPLEX_UNIT_SP, 28f)
        })
        status = TextView(this).apply {
            setTextSize(TypedValue.COMPLEX_UNIT_SP, 15f)
            setPadding(0, pad, 0, pad)
        }
        column.addView(status)
        modelButton = Button(this).apply {
            setOnClickListener { Downloads.start(this@MainActivity, Downloads.What.MODEL) }
        }
        column.addView(modelButton)
        dictionariesButton = Button(this).apply {
            setOnClickListener { Downloads.start(this@MainActivity, Downloads.What.DICTIONARIES) }
        }
        column.addView(dictionariesButton)
        column.addView(Button(this).apply {
            text = getString(R.string.settings)
            setOnClickListener { startActivity(Intent(this@MainActivity, SettingsActivity::class.java)) }
        })
        column.addView(Button(this).apply {
            text = getString(R.string.open_settings)
            setOnClickListener { startActivity(Intent(Settings.ACTION_INPUT_METHOD_SETTINGS)) }
        })
        column.addView(Button(this).apply {
            text = getString(R.string.pick_ime)
            setOnClickListener {
                (getSystemService(INPUT_METHOD_SERVICE) as InputMethodManager).showInputMethodPicker()
            }
        })
        column.addView(EditText(this).apply {
            hint = getString(R.string.try_here)
            minLines = 3
        }, LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT))
        val root = ScrollView(this).apply { addView(column) }
        // drawn edge to edge (API 35): kept clear of the bars, the title our own
        actionBar?.hide()
        if (Build.VERSION.SDK_INT >= 30) {
            root.setOnApplyWindowInsetsListener { v, insets ->
                val bars = insets.getInsets(WindowInsets.Type.systemBars() or WindowInsets.Type.displayCutout())
                v.setPadding(bars.left, bars.top, bars.right, bars.bottom)
                insets
            }
        }
        setContentView(root)
    }

    override fun onResume() {
        super.onResume()
        Downloads.onChange = { runOnUiThread { refresh() } }
        refresh()
    }

    override fun onPause() {
        Downloads.onChange = null
        super.onPause()
    }

    override fun onWindowFocusChanged(hasFocus: Boolean) {
        super.onWindowFocusChanged(hasFocus)
        // back from the input method picker
        if (hasFocus) refresh()
    }

    private fun refresh() {
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
        status.text = buildString {
            appendLine(if (enabled) "✓ 入力方法として有効" else "✗ 入力方法として無効（下のボタンから有効に）")
            appendLine(if (current) "✓ 今の入力方法" else "・今の入力方法ではありません")
            appendLine(
                if (manifest != null) {
                    "✓ 変換モデル: ${manifest.name.removeSuffix(".manifest.json")}"
                } else {
                    "✗ 変換モデルがありません（下のボタンでダウンロード）"
                },
            )
            appendLine(if (lexicon != null) "✓ 読みの辞書: ${lexicon.name}" else "・読みの辞書なし")
            append(if (analyzer != null) "✓ 解析器: ${analyzer.name}" else "・解析器なし")
            if (Downloads.state.isNotEmpty()) append("\n${Downloads.state}")
        }
        // what each button gets as the catalog pins it and as things are (the full lexicon keeps the small one out)
        for ((button, what) in listOf(modelButton to Downloads.What.MODEL, dictionariesButton to Downloads.What.DICTIONARIES)) {
            button.text = runCatching { Downloads.describe(this, what) }.fold(
                { (label, size) -> getString(R.string.download, label, "%.0f".format(size / 1048576.0)) },
                { getString(if (what == Downloads.What.MODEL) R.string.download_model else R.string.download_dictionaries) },
            )
        }
    }
}
