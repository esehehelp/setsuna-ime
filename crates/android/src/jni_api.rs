//! The JNI calls of `io.github.esehehelp.setsuna.Native` (app/).
//!
//! Every call runs on the service's main thread, but the wake-up, which the
//! engine client's reader thread makes: it calls the listener's
//! `onEngineNotify()`, which posts to the main thread. A session is a
//! `Box<Host>` handed to Java as a `long`. Panics never cross the boundary:
//! each call catches them and returns its "nothing happened" value.

use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::PathBuf;
use std::sync::Arc;

use jni::objects::{JClass, JObject, JString};
use jni::sys::{jboolean, jint, jlong, jstring, JNI_FALSE, JNI_TRUE};
use jni::JNIEnv;

use crate::host::{Dirs, Host};

fn guard<R>(fallback: R, f: impl FnOnce() -> R) -> R {
    catch_unwind(AssertUnwindSafe(f)).unwrap_or_else(|_| {
        setsuna_core::diagnostics::log("panic in the Android frontend");
        fallback
    })
}

fn string(env: &mut JNIEnv, s: &JString) -> Option<String> {
    if s.is_null() {
        return None;
    }
    env.get_string(s).ok().map(String::from)
}

fn bool(b: bool) -> jboolean {
    if b {
        JNI_TRUE
    } else {
        JNI_FALSE
    }
}

/// # Safety
/// `h` is 0 or a live session from `create`.
unsafe fn with<R>(h: jlong, fallback: R, f: impl FnOnce(&mut Host) -> R) -> R {
    match (h as *mut Host).as_mut() {
        Some(host) => guard(fallback, || f(host)),
        None => fallback,
    }
}

/// A session: the app's files, native library and model directories, and
/// the listener whose `onEngineNotify()` the engine's results wake. No
/// engine runs until `configure`. 0 on failure.
#[no_mangle]
pub extern "system" fn Java_io_github_esehehelp_setsuna_Native_create<'l>(
    mut env: JNIEnv<'l>,
    _class: JClass<'l>,
    files: JString<'l>,
    native_lib: JString<'l>,
    data: JString<'l>,
    listener: JObject<'l>,
) -> jlong {
    let dir = |env: &mut JNIEnv, s: &JString| PathBuf::from(string(env, s).unwrap_or_default());
    let dirs = Dirs {
        files: dir(&mut env, &files),
        native_lib: dir(&mut env, &native_lib),
        data: dir(&mut env, &data),
    };
    let (Ok(vm), Ok(listener)) = (env.get_java_vm(), env.new_global_ref(&listener)) else {
        return 0;
    };
    let wake = Arc::new(move || {
        if let Ok(mut env) = vm.attach_current_thread_permanently() {
            let _ = env.call_method(listener.as_obj(), "onEngineNotify", "()V", &[]);
            if env.exception_check().unwrap_or(false) {
                let _ = env.exception_clear();
            }
        }
    });
    guard(0, || {
        Box::into_raw(Box::new(Host::new(dirs, wake))) as jlong
    })
}

/// Stop the engine and free the session. No wake-up follows.
///
/// # Safety
/// `h` came from `create` and is not used afterwards.
#[no_mangle]
pub unsafe extern "system" fn Java_io_github_esehehelp_setsuna_Native_destroy(
    _env: JNIEnv,
    _class: JClass,
    h: jlong,
) {
    if h != 0 {
        guard((), || drop(Box::from_raw(h as *mut Host)));
    }
}

/// Reload config.toml for the app `package`, dropping any composition; the
/// engine starts when there is none (a field was focused).
///
/// # Safety
/// `h` is a live session.
#[no_mangle]
pub unsafe extern "system" fn Java_io_github_esehehelp_setsuna_Native_configure<'l>(
    mut env: JNIEnv<'l>,
    _class: JClass<'l>,
    h: jlong,
    package: JString<'l>,
) {
    let package = string(&mut env, &package).unwrap_or_default();
    with(h, (), |host| host.configure(&package))
}

/// A key press: the X keysym it types, `MOD_*` bits and the field's kind
/// (`FIELD_*`), with the app's text before the caret (null when not read).
/// True when the IME took it; else the service hands it to the app.
///
/// # Safety
/// `h` is a live session.
#[no_mangle]
pub unsafe extern "system" fn Java_io_github_esehehelp_setsuna_Native_key<'l>(
    mut env: JNIEnv<'l>,
    _class: JClass<'l>,
    h: jlong,
    sym: jint,
    mods: jint,
    field: jint,
    before_caret: JString<'l>,
) -> jboolean {
    let before = string(&mut env, &before_caret);
    bool(with(h, false, |host| {
        host.key(sym as u32, mods as u32, field, before.as_deref())
    }))
}

/// Take the engine's latest result (after a wake-up); true when the view
/// changed.
///
/// # Safety
/// `h` is a live session.
#[no_mangle]
pub unsafe extern "system" fn Java_io_github_esehehelp_setsuna_Native_applyAsync(
    _env: JNIEnv,
    _class: JClass,
    h: jlong,
) -> jboolean {
    bool(with(h, false, |host| host.session.apply_async_result()))
}

/// How long (ms) until a deferred preedit is shown anyway (0: none waits);
/// the service then calls `tick`.
///
/// # Safety
/// `h` is a live session.
#[no_mangle]
pub unsafe extern "system" fn Java_io_github_esehehelp_setsuna_Native_deferredMs(
    _env: JNIEnv,
    _class: JClass,
    h: jlong,
) -> jint {
    with(h, 0, |host| host.session.deferred_ms() as jint)
}

/// The deferred preedit's deadline passed; true when the view changed.
///
/// # Safety
/// `h` is a live session.
#[no_mangle]
pub unsafe extern "system" fn Java_io_github_esehehelp_setsuna_Native_tick(
    _env: JNIEnv,
    _class: JClass,
    h: jlong,
) -> jboolean {
    bool(with(h, false, |host| host.session.tick()))
}

/// Pick candidate `index` of the open list, or of the suggestions while
/// typing (a tap on the candidate bar).
///
/// # Safety
/// `h` is a live session.
#[no_mangle]
pub unsafe extern "system" fn Java_io_github_esehehelp_setsuna_Native_select(
    _env: JNIEnv,
    _class: JClass,
    h: jlong,
    index: jint,
) -> jboolean {
    bool(with(h, false, |host| {
        host.session.select(index.max(0) as usize)
    }))
}

/// Commit the composition as it stands (the field lost focus).
///
/// # Safety
/// `h` is a live session.
#[no_mangle]
pub unsafe extern "system" fn Java_io_github_esehehelp_setsuna_Native_flush(
    _env: JNIEnv,
    _class: JClass,
    h: jlong,
) {
    with(h, (), |host| host.session.flush())
}

/// Drop the composition (the app moved the caret or changed the text).
///
/// # Safety
/// `h` is a live session.
#[no_mangle]
pub unsafe extern "system" fn Java_io_github_esehehelp_setsuna_Native_reset(
    _env: JNIEnv,
    _class: JClass,
    h: jlong,
) {
    with(h, (), |host| host.session.reset())
}

/// Turn the IME on or off (the keyboard's あ / A key), committing what is
/// composed.
///
/// # Safety
/// `h` is a live session.
#[no_mangle]
pub unsafe extern "system" fn Java_io_github_esehehelp_setsuna_Native_toggle(
    _env: JNIEnv,
    _class: JClass,
    h: jlong,
) {
    with(h, (), |host| host.session.toggle())
}

/// The view as JSON (`host::view_json`); null for no session.
///
/// # Safety
/// `h` is a live session.
#[no_mangle]
pub unsafe extern "system" fn Java_io_github_esehehelp_setsuna_Native_view<'l>(
    env: JNIEnv<'l>,
    _class: JClass<'l>,
    h: jlong,
) -> jstring {
    let json = with(h, None, |host| Some(host.view_json()));
    match json {
        Some(json) => env
            .new_string(json)
            .map_or(std::ptr::null_mut(), |s| s.into_raw()),
        None => std::ptr::null_mut(),
    }
}
