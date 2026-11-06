package io.github.esehehelp.setsuna

import android.content.Context
import java.io.File

/** What the APK carries as assets, unpacked where setsuna-core reads it. */
object Assets {
    /** The mixed input model (crates/ime-core/data/mixed). */
    private val MIXED = listOf("kana.tsv", "latin.tsv", "params.txt")

    /**
     * Unpack the mixed input model into `files/mixed` once per install of
     * the app (an update brings its own model).
     */
    fun unpack(context: Context) {
        val dir = File(context.filesDir, "mixed")
        val stamp = File(dir, ".installed")
        val installed = context.packageManager
            .getPackageInfo(context.packageName, 0).lastUpdateTime.toString()
        if (stamp.isFile && stamp.readText() == installed) return
        dir.mkdirs()
        for (name in MIXED) {
            val tmp = File(dir, "$name.new")
            context.assets.open("mixed/$name").use { input ->
                tmp.outputStream().use { input.copyTo(it) }
            }
            tmp.renameTo(File(dir, name))
        }
        stamp.writeText(installed)
    }
}
