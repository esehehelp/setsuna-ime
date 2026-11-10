package io.github.esehehelp.setsuna

import android.content.Context
import java.io.File

/** What the APK carries as assets, unpacked where setsuna-core reads it. */
object Assets {
    /** The mixed input model (crates/ime-core/data/mixed). */
    private val MIXED = listOf("kana.tsv", "latin.tsv", "params.txt")

    /** The conversion model (assets/model: build.ps1 puts models.json's own there). */
    private const val MODEL = "model"

    /** When the app was installed or last updated: what an unpacking is stamped with. */
    private fun installed(context: Context): String =
        context.packageManager.getPackageInfo(context.packageName, 0).lastUpdateTime.toString()

    /** Copy the asset `path` to `to`, through a file beside it renamed over it. */
    private fun copy(context: Context, path: String, to: File): Boolean {
        val tmp = File(to.parentFile, "${to.name}.new")
        context.assets.open(path).use { input ->
            tmp.outputStream().use { input.copyTo(it) }
        }
        return tmp.renameTo(to)
    }

    /**
     * Unpack the mixed input model into `files/mixed` once per install of
     * the app (an update brings its own model), and the conversion model
     * into `files/models/onnx` (see [unpackModel]).
     */
    fun unpack(context: Context) {
        // a full disk leaves the model out, not the keyboard
        runCatching { unpackModel(context) }
        val dir = File(context.filesDir, "mixed")
        val stamp = File(dir, ".installed")
        val installed = installed(context)
        if (stamp.isFile && stamp.readText() == installed) return
        dir.mkdirs()
        for (name in MIXED) {
            copy(context, "mixed/$name", File(dir, name))
        }
        stamp.writeText(installed)
    }

    /**
     * The conversion model into `files/models/onnx`, where the engine takes
     * the only manifest there, once per install of the app: the one unpacked
     * before goes (the files `.bundled` lists after its stamp), and none is
     * unpacked where another model is (one push-models.ps1 copied there).
     */
    private fun unpackModel(context: Context) {
        val names = context.assets.list(MODEL)?.toList().orEmpty()
        if (names.isEmpty()) return
        val dir = File(context.filesDir, "models/onnx")
        val stamp = File(dir, ".bundled")
        val installed = installed(context)
        val lines = if (stamp.isFile) stamp.readLines() else emptyList()
        if (lines.firstOrNull() == installed) return
        for (name in lines.drop(1)) {
            if (name.isNotEmpty() && !name.contains('/')) File(dir, name).delete()
        }
        stamp.delete()
        if (dir.listFiles()?.any { it.name.endsWith(".manifest.json") } == true) return
        dir.mkdirs()
        // stamped once all of it is in: else tried again at the next start
        if (names.all { copy(context, "$MODEL/$it", File(dir, it)) }) {
            stamp.writeText((listOf(installed) + names).joinToString("\n"))
        }
    }
}
