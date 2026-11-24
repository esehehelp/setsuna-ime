//! The session the input method service keeps, and where its files are.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use setsuna_core::child_daemon;
use setsuna_core::config::{AppStart, Config};
use setsuna_core::engine_client::{EngineClient, Notify};
use setsuna_core::keysym::Key;
use setsuna_core::session::{Field, Session};

/// The engine binary in the native library directory: packaged as a
/// library so that the system extracts it, executable, with the others.
pub const ENGINE_LIB: &str = "libsetsuna_engine.so";

/// The app's directories, as its `Context` names them.
#[derive(Debug, Clone)]
pub struct Dirs {
    /// `getFilesDir()`: config.toml and the learning (`config/setsuna-ime`),
    /// the logs (`state/setsuna-ime`), and the mixed input model the app
    /// unpacks from its assets (`mixed`).
    pub files: PathBuf,
    /// `ApplicationInfo.nativeLibraryDir`: [`ENGINE_LIB`].
    pub native_lib: PathBuf,
    /// Where the models are: `models/onnx/<manifest>` and its files, the one
    /// `models/lexicon/*.lexicon`, the one
    /// `models/analyzer/<name>/analyzer.manifest.json`, and
    /// `models/names/latin-names.tsv`.
    pub data: PathBuf,
}

impl Dirs {
    /// Point the config and log directories at the app's own.
    pub fn export(&self) {
        export_files(&self.files);
    }
}

/// [`Dirs::export`] from the files directory alone, as the settings page
/// has it (its activity may start before the service): `config_dir` and
/// `log_path` under it, the engine (a child) given them as its
/// XDG_CONFIG_HOME and XDG_STATE_HOME. Never through this process's
/// environment, which ART's threads read with no lock while Rust would set
/// it.
pub fn export_files(files: &Path) {
    setsuna_core::config::set_app_dirs(setsuna_core::config::AppDirs {
        config_home: files.join("config"),
        state_home: files.join("state"),
        home: files.to_path_buf(),
    });
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
    if e.analyzer_manifest.is_none() {
        e.analyzer_manifest = the_one(&models.join("analyzer"), |p| {
            let m = p.join("analyzer.manifest.json");
            m.is_file().then_some(m)
        });
    }
    // the reading corrector the app downloads with the model: models/corrector/<name>/<name>.manifest.json
    if e.corrector_manifest.is_none() {
        e.corrector_manifest = the_one(&models.join("corrector"), |p| {
            let name = p.file_name()?.to_str()?;
            let m = p.join(format!("{name}.manifest.json"));
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
/// Direct input (a password, a number, a field of no type): keys that
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
    /// The engine daemon started with the service, off the main thread, for
    /// the first `configure` to take.
    started: Arc<Mutex<Started>>,
    /// The `[engine]` settings the running daemon was started with (as
    /// `engine_key` writes them): another daemon starts when they change.
    engine_key: Option<String>,
}

/// `[engine]` as compared for a restart of the daemon (the Linux addon's
/// way), with the manifests of the models under `data/models` it reads, by
/// their size and time: the model is found there, not named in `[engine]`,
/// and a download replaces it (or the corrector, or the analyzer) in place.
fn engine_key(cfg: &Config, data: &Path) -> String {
    // every file of the models, not their manifests only: a dictionary put back under a manifest
    // that stayed (a daemon that failed to load it keeps the error) starts another daemon too;
    // a `.part` being downloaded is none of them yet
    fn walk(dir: &Path, depth: usize, files: &mut Vec<String>) {
        let Ok(entries) = std::fs::read_dir(dir) else {
            return;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            let Ok(meta) = entry.metadata() else {
                continue;
            };
            if meta.is_dir() {
                if depth > 0 {
                    walk(&path, depth - 1, files);
                }
            } else if !path.to_string_lossy().ends_with(".part") {
                files.push(format!(
                    "{} {} {:?}",
                    path.display(),
                    meta.len(),
                    meta.modified().ok()
                ));
            }
        }
    }
    let mut files = Vec::new();
    for kind in ["onnx", "corrector", "analyzer"] {
        walk(&data.join("models").join(kind), 3, &mut files);
    }
    files.sort();
    format!("{:?} {files:?}", cfg.engine)
}

/// The daemon the warm-up thread starts: not yet, there to take, or taken
/// (or none started) — then `configure` starts one itself, and the warm-up
/// thread, coming later, none.
enum Started {
    Pending,
    /// The daemon, and the `engine_key` of the settings it was started with.
    Ready(child_daemon::ChildDaemonClient, String),
    Taken,
}

impl Host {
    pub fn new(dirs: Dirs, wake: Arc<dyn Fn() + Send + Sync>) -> Self {
        dirs.export();
        setsuna_core::diagnostics::init("android.log");
        // what the first field's `configure` would do on the main thread
        // (start the engine, whose fork of this process took ~96 ms on the
        // phone, and load the mixed input model's tables: 250 ms on a PC)
        // happens now, off it: the daemon for `configure` to take, the tables
        // in the caches every session shares
        let started = Arc::new(Mutex::new(Started::Pending));
        let (warm, ahead, waker) = (dirs.clone(), started.clone(), wake.clone());
        let _ = std::thread::Builder::new()
            .name("setsuna-warm".into())
            .spawn(move || {
                let cfg = with_app_defaults(Config::load_default(), &warm);
                {
                    let mut state = ahead.lock().unwrap_or_else(|e| e.into_inner());
                    if matches!(*state, Started::Pending) {
                        let notify: Notify = Box::new(move |_req_id| waker());
                        // the files' key before the daemon reads them: a model published while it
                        // starts makes the next field start another, never pass for the new one
                        let key = engine_key(&cfg, &warm.data);
                        *state = match child_daemon::spawn_child_with_config(
                            &cfg,
                            Some(&warm.data),
                            notify,
                        ) {
                            Some(c) => Started::Ready(c, key),
                            None => Started::Taken,
                        };
                    }
                }
                Session::new(&Config::default(), None).configure_with(&cfg, None);
            });
        Self {
            session: Session::new(&Config::default(), None),
            dirs,
            wake,
            started,
            engine_key: None,
        }
    }

    fn notify(&self) -> Notify {
        let wake = self.wake.clone();
        Box::new(move |_req_id| wake())
    }

    /// Take up config.toml for the app `package` (its `[[apps]]` settings
    /// over the rest, and its start state), dropping any composition, and
    /// start the engine when there is none, its daemon exited, or `[engine]`
    /// changed (the settings page's corrector threshold): the new daemon
    /// replaces the old one, which ends with its client; when it does not
    /// start, the old one stays and the next field tries again. Called when
    /// a field is focused, as the TSF DLL reloads on ActivateEx.
    pub fn configure(&mut self, package: &str) {
        let (cfg, app) = with_app_defaults(Config::load_default(), &self.dirs).for_app(package);
        let key = engine_key(&cfg, &self.dirs.data);
        let client = if self.session.has_client() && self.engine_key.as_ref() == Some(&key) {
            None
        } else {
            // the one started with the service when it was started so, else one now (none was,
            // its daemon exited, or the settings changed)
            let ahead = std::mem::replace(
                &mut *self.started.lock().unwrap_or_else(|e| e.into_inner()),
                Started::Taken,
            );
            match ahead {
                Started::Ready(c, k) if k == key => Some(Box::new(c) as Box<dyn EngineClient>),
                _ => child_daemon::spawn_with_config(&cfg, Some(&self.dirs.data), self.notify()),
            }
        };
        if client.is_some() {
            setsuna_core::diagnostics::log("android engine started");
            self.engine_key = Some(key);
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
        std::fs::create_dir_all(models.join("corrector/N-tiny")).unwrap();
        std::fs::write(models.join("corrector/N-tiny/N-tiny.manifest.json"), "{}").unwrap();
        std::fs::create_dir_all(models.join("lexicon")).unwrap();
        std::fs::write(models.join("lexicon/words.lexicon"), "LEXICON1").unwrap();
        std::fs::create_dir_all(dirs.files.join("mixed")).unwrap();
        std::fs::write(dirs.files.join("mixed/kana.tsv"), "").unwrap();

        let cfg = with_app_defaults(Config::default(), &dirs);
        assert_eq!(
            cfg.engine.engine_exe,
            Some(dirs.native_lib.join(ENGINE_LIB))
        );
        assert_eq!(
            cfg.engine.analyzer_manifest,
            Some(models.join("analyzer/v1/analyzer.manifest.json"))
        );
        assert_eq!(
            cfg.engine.corrector_manifest,
            Some(models.join("corrector/N-tiny/N-tiny.manifest.json"))
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
        own.engine.engine_exe = Some(PathBuf::from("/e.so"));
        std::fs::write(models.join("lexicon/other.lexicon"), "LEXICON1").unwrap();
        let cfg = with_app_defaults(own, &dirs);
        assert_eq!(cfg.engine.engine_exe, Some(PathBuf::from("/e.so")));
        assert!(cfg.lexicon.file.is_none());
        let _ = std::fs::remove_dir_all(&root);
    }

    /// A download replaces the model in place: the next field starts the
    /// engine again with it.
    #[test]
    fn a_model_replaced_in_place_changes_the_engine_key() {
        let root = temp("engine-key");
        let onnx = root.join("models/onnx");
        let corrector = root.join("models/corrector/N-tiny");
        std::fs::create_dir_all(&onnx).unwrap();
        std::fs::create_dir_all(&corrector).unwrap();
        let cfg = Config::default();
        let none = engine_key(&cfg, &root);
        std::fs::write(onnx.join("M.manifest.json"), "{}").unwrap();
        let model = engine_key(&cfg, &root);
        assert_ne!(none, model);
        assert_eq!(model, engine_key(&cfg, &root));
        std::fs::write(onnx.join("M.manifest.json"), r#"{"v":2}"#).unwrap();
        let replaced = engine_key(&cfg, &root);
        assert_ne!(model, replaced);
        std::fs::write(corrector.join("N-tiny.manifest.json"), "{}").unwrap();
        assert_ne!(replaced, engine_key(&cfg, &root));
        // what is no manifest (a weight file, a .part) changes nothing
        let with_corrector = engine_key(&cfg, &root);
        std::fs::write(onnx.join("M.encoder.safetensors.part"), "x").unwrap();
        assert_eq!(with_corrector, engine_key(&cfg, &root));
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
            started: Arc::new(Mutex::new(Started::Taken)),
            engine_key: None,
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
