package io.github.esehehelp.setsuna

/**
 * The Rust half of the input method (crates/android/src/jni_api.rs): a
 * session of setsuna-core and the engine it converts with. Every call but
 * [Listener.onEngineNotify] and [settings] is made on the main thread.
 */
object Native {
    init {
        System.loadLibrary("setsuna_android")
    }

    /** What the field typed into is: a normal one. */
    const val FIELD_NORMAL = 0

    /** No context, no rewrite, nothing learned. */
    const val FIELD_PRIVATE = 1

    /** Direct input: keys that would start a composition go to the app. */
    const val FIELD_DIRECT = 2

    const val MOD_SHIFT = 1
    const val MOD_CTRL = 1 shl 1
    const val MOD_ALT = 1 shl 2
    const val MOD_SUPER = 1 shl 3

    interface Listener {
        /** An engine result is in; called from the engine client's thread. */
        fun onEngineNotify()
    }

    @JvmStatic
    external fun create(files: String, nativeLib: String, data: String, listener: Listener): Long

    @JvmStatic
    external fun destroy(h: Long)

    @JvmStatic
    external fun configure(h: Long, packageName: String)

    @JvmStatic
    external fun key(h: Long, sym: Int, mods: Int, field: Int, beforeCaret: String?): Boolean

    @JvmStatic
    external fun applyAsync(h: Long): Boolean

    @JvmStatic
    external fun deferredMs(h: Long): Int

    @JvmStatic
    external fun tick(h: Long): Boolean

    @JvmStatic
    external fun select(h: Long, index: Int): Boolean

    @JvmStatic
    external fun flush(h: Long)

    @JvmStatic
    external fun reset(h: Long)

    @JvmStatic
    external fun toggle(h: Long)

    @JvmStatic
    external fun view(h: Long): String?

    /**
     * A message of the settings page (JSON) for the config under [files]
     * (the app's files directory): the reply as JSON, null for none. Any
     * thread; no session needed.
     */
    @JvmStatic
    external fun settings(files: String, message: String): String?

    /** The version and the commit built ("0.2.3 (d4dbe59)"), as the settings page shows them. */
    @JvmStatic
    external fun version(): String
}
