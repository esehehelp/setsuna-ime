package io.github.esehehelp.setsuna

import android.content.Context
import android.net.Uri
import android.util.Log
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.io.IOException
import java.net.HttpURLConnection
import java.net.URL
import java.security.MessageDigest

/**
 * What the app downloads from Hugging Face (user 2026-11-10: the model the
 * APK carried did not come out of it on the phone): the conversion model,
 * the reading lexicon and the analyzer that crates/tsf/models.json pins
 * (assets/models.json, build.ps1 puts it there), each file checked by its
 * size and SHA-256 and put where setsuna-core looks for it, the app's
 * files/models: onnx/, lexicon/, analyzer/<name>/. The files arrive as
 * `.part` beside their place; once all of them are in, what each directory
 * held before goes (the engine takes the only manifest of onnx/, the
 * session the only lexicon and analyzer) and they are renamed into place.
 * A file installed already as pinned is not downloaded again. One download
 * at a time, on a thread of its own; [state] says how it goes.
 */
object Downloads {
    private const val TAG = "SetsunaDownloads"

    enum class What { MODEL, DICTIONARIES }

    /** A pinned file: its repository path, size and SHA-256, and where it goes. */
    private class Pin(val path: String, val size: Long, val sha256: String, val to: File)

    /** A repository's files at a revision, under `root` (whose other files go once they are in). */
    private class Part(val repository: String, val revision: String, val root: File, val pins: List<Pin>)

    /** What the download is doing, or how the last one ended ("" before any). */
    @Volatile
    var state: String = ""
        private set

    @Volatile
    var running = false
        private set

    /** Told (on the download's thread) when [state] changes. */
    @Volatile
    var onChange: (() -> Unit)? = null

    private fun set(s: String) {
        state = s
        onChange?.invoke()
    }

    /** Download [what] unless a download is going on. */
    fun start(context: Context, what: What) {
        synchronized(this) {
            if (running) return
            running = true
        }
        val app = context.applicationContext
        val label = if (what == What.MODEL) "変換モデル" else "読みの辞書と解析器"
        set("$label を確認中…")
        Thread {
            try {
                fetch(app, what, label)
                set("✓ $label を入れました（キーボードを開き直すと使います）")
            } catch (e: Exception) {
                Log.w(TAG, "download of $what failed", e)
                set("✗ $label のダウンロードに失敗: ${e.message ?: e.javaClass.simpleName}")
            } finally {
                running = false
                onChange?.invoke()
            }
        }.start()
    }

    /** The parts of [what] as the catalog pins them. */
    private fun parts(context: Context, what: What): List<Part> {
        val catalog = JSONObject(context.assets.open("models.json").bufferedReader().use { it.readText() })
        val models = File(context.filesDir, "models")
        return when (what) {
            What.MODEL -> {
                val onnx = File(models, "onnx")
                val files = catalog.getJSONObject("variants").getJSONObject("int8").getJSONArray("files")
                listOf(part(catalog, files, onnx) { File(onnx, it) })
            }
            What.DICTIONARIES -> {
                val components = catalog.getJSONObject("components")
                val lexicon = components.getJSONObject("lexicon")
                val lexiconDir = File(models, "lexicon")
                val analyzer = components.getJSONObject("analyzer")
                // the analyzer's files keep their paths below its manifest's directory
                val base = analyzer.getString("entry").substringBeforeLast('/', "")
                val analyzers = File(models, "analyzer")
                val analyzerDir = File(analyzers, base.ifEmpty { "analyzer" })
                listOf(
                    part(lexicon, lexicon.getJSONArray("files"), lexiconDir) {
                        File(lexiconDir, it.substringAfterLast('/'))
                    },
                    part(analyzer, analyzer.getJSONArray("files"), analyzers) {
                        File(analyzerDir, if (base.isEmpty()) it else it.removePrefix("$base/"))
                    },
                )
            }
        }
    }

    private fun part(source: JSONObject, files: JSONArray, root: File, place: (String) -> File): Part {
        val repository = source.getString("repository")
        val revision = source.getString("revision")
        require(Regex("[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*").matches(repository)) {
            "invalid repository $repository"
        }
        require(Regex("[a-f0-9]{40}").matches(revision)) { "revision not pinned: $revision" }
        val pins = (0 until files.length()).map { i ->
            val f = files.getJSONObject(i)
            val path = f.getString("path")
            require(path.isNotEmpty() && !path.startsWith("/") && path.split('/').none { it.isEmpty() || it == "." || it == ".." }) {
                "invalid path $path"
            }
            Pin(path, f.getLong("size"), f.getString("sha256"), place(path))
        }
        return Part(repository, revision, root, pins)
    }

    private fun fetch(context: Context, what: What, label: String) {
        val parts = parts(context, what)
        val total = parts.sumOf { p -> p.pins.sumOf { it.size } }
        var done = 0L
        var shown = 0L
        val progress = { n: Long ->
            done += n
            val now = System.currentTimeMillis()
            if (now - shown > 300) {
                shown = now
                set("$label をダウンロード中 ${mb(done)} / ${mb(total)} MB")
            }
        }
        // every file, as .part beside its place (none of them a manifest the engine would take)
        val got = mutableListOf<Pair<File, File>>()
        for (p in parts) {
            for (pin in p.pins) {
                if (verified(pin.to, pin)) {
                    progress(pin.size)
                    continue
                }
                val tmp = File(pin.to.path + ".part")
                val url = "https://huggingface.co/${p.repository}/resolve/${p.revision}/" +
                    pin.path.split('/').joinToString("/") { Uri.encode(it) }
                download(url, tmp, pin, progress)
                got.add(tmp to pin.to)
            }
        }
        // all in: what the directories held before goes, the new files take their places
        for (p in parts) {
            val keep = p.pins.map { it.to.canonicalFile }.toSet() + got.map { it.first.canonicalFile }
            p.root.walkBottomUp().forEach { f ->
                if (f == p.root) return@forEach
                if (f.isFile && f.canonicalFile !in keep) f.delete()
                if (f.isDirectory && f.list()?.isEmpty() == true) f.delete()
            }
        }
        for ((tmp, to) in got) {
            if (!tmp.renameTo(to)) throw IOException("cannot put ${to.name} in place")
        }
    }

    private fun mb(n: Long) = "%.1f".format(n / 1048576.0)

    /** The file is there with the pinned size and SHA-256. */
    private fun verified(f: File, pin: Pin): Boolean {
        if (!f.isFile || f.length() != pin.size) return false
        val md = MessageDigest.getInstance("SHA-256")
        f.inputStream().use { input ->
            val buf = ByteArray(1 shl 16)
            while (true) {
                val r = input.read(buf)
                if (r < 0) break
                md.update(buf, 0, r)
            }
        }
        return hex(md.digest()) == pin.sha256
    }

    private fun hex(b: ByteArray) = b.joinToString("") { "%02x".format(it) }

    /** `url` into `to`, checked against the pin as it comes (a mismatch leaves nothing). */
    private fun download(url: String, to: File, pin: Pin, progress: (Long) -> Unit) {
        to.parentFile?.mkdirs()
        val c = URL(url).openConnection() as HttpURLConnection
        c.connectTimeout = 30_000
        c.readTimeout = 60_000
        c.instanceFollowRedirects = true
        c.setRequestProperty("User-Agent", "setsuna-ime-android")
        try {
            if (c.responseCode != 200) throw IOException("HTTP ${c.responseCode}: ${pin.path}")
            val md = MessageDigest.getInstance("SHA-256")
            var n = 0L
            c.inputStream.use { input ->
                to.outputStream().use { out ->
                    val buf = ByteArray(1 shl 16)
                    while (true) {
                        val r = input.read(buf)
                        if (r < 0) break
                        out.write(buf, 0, r)
                        md.update(buf, 0, r)
                        n += r
                        progress(r.toLong())
                    }
                }
            }
            if (n != pin.size || hex(md.digest()) != pin.sha256) {
                to.delete()
                throw IOException("size or SHA-256 mismatch: ${pin.path}")
            }
        } catch (e: Exception) {
            to.delete()
            throw e
        } finally {
            c.disconnect()
        }
    }
}
