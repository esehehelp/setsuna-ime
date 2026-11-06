//! The session the input method service keeps, and where its files are.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use setsuna_core::child_daemon;
use setsuna_core::config::{AppStart, Config};
use setsuna_core::engine_client::Notify;
use setsuna_core::keysym::Key;
use setsuna_core::session::{Field, Session};

/// The engine binary in the native library directory: packaged as a
/// library so that the system extracts it, executable, with the others.
pub const ENGINE_LIB: &str = "libsetsuna_engine.so";
/// ONNX Runtime, from Microsoft's onnxruntime-android package.
pub const ORT_LIB: &str = "libonnxruntime.so";

/// The app's directories, as its `Context` names them.
#[derive(Debug, Clone)]
pub struct Dirs {
    /// `getFilesDir()`: config.toml and the learning (`config/setsuna-ime`),
    /// the logs (`state/setsuna-ime`), and the mixed input model the app
    /// unpacks from its assets (`mixed`).
    pub files: PathBuf,
    /// `ApplicationInfo.nativeLibraryDir`: [`ENGINE_LIB`] and [`ORT_LIB`].
    pub native_lib: PathBuf,
    /// Where the models are: `models/onnx/<manifest>` and its files, the one
    /// `models/lexicon/*.lexicon`, the one
    /// `models/analyzer/<name>/analyzer.manifest.json`, and
    /// `models/names/latin-names.tsv`.
    pub data: PathBuf,
}

impl Dirs {
    /// Point the config and log directories at the app's own: off Windows
    /// and macOS, `config_dir` and `log_path` follow XDG_CONFIG_HOME and
    /// XDG_STATE_HOME. The engine, a child, inherits them.
    pub fn export(&self) {
        export_files(&self.files);
    }
}

/// [`Dirs::export`] from the files directory alone, as the settings page
/// has it (its activity may start before the service). A variable already
/// so is left alone: setting one while another thread reads the
/// environment is not safe.
pub fn export_files(files: &Path) {
    let set = |key: &str, value: PathBuf| {
        if std::env::var_os(key).as_deref() != Some(value.as_os_str()) {
            std::env::set_var(key, value);
        }
    };
    set("XDG_CONFIG_HOME", files.join("config"));
    set("XDG_STATE_HOME", files.join("state"));
    if std::env::var_os("HOME").is_none_or(|h| h.is_empty()) {
        std::env::set_var("HOME", files);
    }
}

/// `config.toml` over what the app supplies, each where config.toml sets
/// nothing: the engine and ONNX Runtime in the native library directory,
/// the analyzer, the reading lexicon and the Latin names dictionary under
/// `data/models`, and the mixed input model in `files/mixed`. The model
/// itself is found by the engine client in `data/models/onnx`.
pub fn with_app_defaults(mut cfg: Config, dirs: &Dirs) -> Config {
    let path = |p: PathBuf| p.to_string_lossy().into_owned();
    let models = dirs.data.join("models");
    let e = &mut cfg.engine;
    if e.engine_exe.is_none() {
        e.engine_exe = Some(dirs.native_lib.join(ENGINE_LIB));
    }
    if e.ort_dylib.is_none() {
        e.ort_dylib = Some(dirs.native_lib.join(ORT_LIB));
    }
    if e.analyzer_manifest.is_none() {
        e.analyzer_manifest = the_one(&models.join("analyzer"), |p| {
            let m = p.join("analyzer.manifest.json");
            m.is_file().then_some(m)
        });
    }
    if cfg.lexicon.file.is_none() {
        cfg.lexicon.file = the_one(&models.join("lexicon"), |p| {
            (p.is_file() && p.extension().is_some_and(|x| x == "lexicon")).then(|| p.to_path_buf())
        })
        .map(path);
    }
    if cfg.frontend.mixed_model.is_none() {
        let mixed = dirs.files.join("mixed");
        cfg.frontend.mixed_model = mixed.join("kana.tsv").is_file().then(|| path(mixed));
    }
    if cfg.frontend.name_dictionary_file.is_none() {
        let names = models.join("names").join("latin-names.tsv");
        cfg.frontend.name_dictionary_file = names.is_file().then(|| path(names));
    }
    cfg
}

/// The one entry of `dir` that `pick` takes: none when there is none or
/// several (which one is not guessed).
fn the_one(dir: &Path, pick: impl Fn(&Path) -> Option<PathBuf>) -> Option<PathBuf> {
    let mut found = std::fs::read_dir(dir)
        .ok()?
        .flatten()
        .filter_map(|e| pick(&e.path()));
    let first = found.next()?;
    found.next().is_none().then_some(first)
}

/// `key`'s `field`: what the field typed into is.
pub const FIELD_NORMAL: i32 = 0;
/// No context, no rewrite, nothing learned (`IME_FLAG_NO_PERSONALIZED_LEARNING`).
pub const FIELD_PRIVATE: i32 = 1;
/// Direct input (a password, a URL, an e-mail address, a number): keys that
/// would start a composition go to the app.
pub const FIELD_DIRECT: i32 = 2;

pub fn field(code: i32) -> Field {
    match code {
        FIELD_PRIVATE => Field::Private,
        FIELD_DIRECT => Field::Direct,
        _ => Field::Normal,
    }
}

/// The service's session and the engine it converts with.
pub struct Host {
    pub session: Session,
    dirs: Dirs,
    /// Wakes the service's main thread, from the engine client's reader
    /// thread; the service then takes the result (`apply_async`).
    wake: Arc<dyn Fn() + Send + Sync>,
}

impl Host {
    pub fn new(dirs: Dirs, wake: Arc<dyn Fn() + Send + Sync>) -> Self {
        dirs.export();
        setsuna_core::diagnostics::init("android.log");
        Self {
            session: Session::new(&Config::default(), None),
            dirs,
            wake,
        }
    }

    fn notify(&self) -> Notify {
        let wake = self.wake.clone();
        Box::new(move |_req_id| wake())
    }

    /// Take up config.toml for the app `package` (its `[[apps]]` settings
    /// over the rest, and its start state), dropping any composition, and
    /// start the engine when there is none or its daemon exited. Called when
    /// a field is focused, as the TSF DLL reloads on ActivateEx.
    pub fn configure(&mut self, package: &str) {
        let (cfg, app) = with_app_defaults(Config::load_default(), &self.dirs).for_app(package);
        let client = (!self.session.has_client())
            .then(|| child_daemon::spawn_with_config(&cfg, Some(&self.dirs.data), self.notify()))
            .flatten();
        if client.is_some() {
            setsuna_core::diagnostics::log("android engine started");
        }
        self.session.configure_with(&cfg, client);
        match app.map(|a| a.start) {
            Some(AppStart::On) => self.session.set_enabled(true),
            Some(AppStart::Off) => self.session.set_enabled(false),
            _ => {}
        }
    }

    /// A key press: the X keysym it types and `MOD_*` bits, into a field of
    /// `field` (`FIELD_*`), with the app's text before the caret when it was
    /// read. True when the IME took it.
    pub fn key(&mut self, sym: u32, mods: u32, field_code: i32, before: Option<&str>) -> bool {
        let key = Key { sym, code: 0, mods };
        self.session.key_in(&key, field(field_code), before)
    }

    /// The view as JSON, for the service to render.
    pub fn view_json(&self) -> String {
        view_json(&self.session)
    }

    /// [`Host::view_json`], its commits taken: they go into the app once.
    pub fn take_view_json(&mut self) -> String {
        let json = self.view_json();
        self.session.clear_commits();
        json
    }
}

/// `s`'s view as JSON. The caret (`cursor`) is in UTF-16 units into the
/// preedit's segments joined, as Java counts.
pub fn view_json(s: &Session) -> String {
    let v = s.view();
    let joined: String = v.preedit.concat();
    let cursor: usize = joined
        .get(..v.cursor().min(joined.len()))
        .map_or(0, |p| p.encode_utf16().count());
    let (form, form_key) = v.form.unwrap_or(("", ""));
    serde_json::json!({
        "commits": v.commits,
        "preedit": v.preedit,
        "focus": v.focus.map_or(-1, |f| f as i64),
        "cursor": cursor,
        "candidates": v.candidates,
        "comments": v.comments,
        "selected": v.selected,
        "suggestions": v.suggestions,
        "reading": v.reading,
        "form": form,
        "formKey": form_key,
        "status": v.status,
        "enabled": s.enabled(),
        "engine": s.has_client(),
        "pageSize": s.page_size(),
    })
    .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp(name: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("setsuna-android-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn what_the_app_carries_and_holds_are_the_defaults() {
        let root = temp("defaults");
        let dirs = Dirs {
            files: root.join("files"),
            native_lib: root.join("lib"),
            data: root.join("data"),
        };
        let models = dirs.data.join("models");
        std::fs::create_dir_all(models.join("analyzer/v1")).unwrap();
        std::fs::write(models.join("analyzer/v1/analyzer.manifest.json"), "{}").unwrap();
        std::fs::create_dir_all(models.join("lexicon")).unwrap();
        std::fs::write(models.join("lexicon/words.lexicon"), "LEXICON1").unwrap();
        std::fs::create_dir_all(dirs.files.join("mixed")).unwrap();
        std::fs::write(dirs.files.join("mixed/kana.tsv"), "").unwrap();

        let cfg = with_app_defaults(Config::default(), &dirs);
        assert_eq!(
            cfg.engine.engine_exe,
            Some(dirs.native_lib.join(ENGINE_LIB))
        );
        assert_eq!(cfg.engine.ort_dylib, Some(dirs.native_lib.join(ORT_LIB)));
        assert_eq!(
            cfg.engine.analyzer_manifest,
            Some(models.join("analyzer/v1/analyzer.manifest.json"))
        );
        assert_eq!(
            cfg.lexicon.file.map(PathBuf::from),
            Some(models.join("lexicon/words.lexicon"))
        );
        assert_eq!(
            cfg.frontend.mixed_model.map(PathBuf::from),
            Some(dirs.files.join("mixed"))
        );
        assert!(cfg.frontend.name_dictionary_file.is_none());

        // config.toml's own wins; two lexicons: which one is not guessed
        let mut own = Config::default();
        own.engine.ort_dylib = Some(PathBuf::from("/o.so"));
        std::fs::write(models.join("lexicon/other.lexicon"), "LEXICON1").unwrap();
        let cfg = with_app_defaults(own, &dirs);
        assert_eq!(cfg.engine.ort_dylib, Some(PathBuf::from("/o.so")));
        assert!(cfg.lexicon.file.is_none());
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn the_view_counts_its_caret_as_java_does() {
        let mut s = Session::new(&Config::default(), None);
        let key = |s: &mut Session, sym: u32| {
            s.key_in(
                &Key {
                    sym,
                    code: 0,
                    mods: 0,
                },
                Field::Normal,
                None,
            )
        };
        for c in "kana".chars() {
            assert!(key(&mut s, c as u32));
        }
        let v: serde_json::Value = serde_json::from_str(&view_json(&s)).unwrap();
        assert_eq!(v["preedit"], serde_json::json!(["かな"]));
        // two chars, six bytes
        assert_eq!(v["cursor"], 2);
        assert_eq!(v["focus"], -1);
        assert_eq!(v["engine"], false);
        // Return commits
        assert!(key(&mut s, 0xff0d));
        let v: serde_json::Value = serde_json::from_str(&view_json(&s)).unwrap();
        assert_eq!(v["commits"], serde_json::json!(["かな"]));
        assert_eq!(v["preedit"], serde_json::json!([]));
    }

    #[test]
    fn the_service_takes_the_commits_once() {
        let mut host = Host {
            session: Session::new(&Config::default(), None),
            dirs: Dirs {
                files: PathBuf::new(),
                native_lib: PathBuf::new(),
                data: PathBuf::new(),
            },
            wake: Arc::new(|| {}),
        };
        assert!(host.key('a' as u32, 0, FIELD_NORMAL, None));
        host.session.flush();
        let commits = |json: &str| -> serde_json::Value {
            serde_json::from_str::<serde_json::Value>(json).unwrap()["commits"].clone()
        };
        assert_eq!(commits(&host.take_view_json()), serde_json::json!(["あ"]));
        // read again before a key (the keyboard shown again): nothing to write
        assert_eq!(commits(&host.take_view_json()), serde_json::json!([]));
    }

    #[test]
    fn a_direct_field_leaves_letters_to_the_app() {
        let mut s = Session::new(&Config::default(), None);
        let typed = s.key_in(
            &Key {
                sym: 'a' as u32,
                code: 0,
                mods: 0,
            },
            field(FIELD_DIRECT),
            None,
        );
        assert!(!typed);
        assert!(s.view().preedit.is_empty());
    }
}
