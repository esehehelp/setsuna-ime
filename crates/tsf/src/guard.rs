//! The panic backstop of the entry points the host calls (COM methods,
//! window procedures). A panic unwinding out of an `extern "system"`
//! function aborts the process, and the process is the user's app (a
//! browser, an editor): each entry point runs its body through [`run`] or
//! [`catch`], which stop the panic there. The service then drops what it
//! was composing (`SetsunaTextService_Impl::guarded`), so the next key
//! starts clean.
//!
//! RefCell borrows held when the panic hit are released as it unwinds
//! (a `RefCell` is not poisoned), so the state is usable afterwards.

use std::panic::{catch_unwind, AssertUnwindSafe};

/// Run `f`, the body of the entry point `what`: `None` when it panicked
/// (logged).
pub fn run<T>(what: &str, f: impl FnOnce() -> T) -> Option<T> {
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(v) => Some(v),
        Err(_) => {
            // where it happened is in the log already (`install_hook`)
            setsuna_core::diagnostics::log(format!("panic caught in {what}"));
            tracing::error!("panic caught in {what}");
            None
        }
    }
}

/// [`run`], with `fallback` as what the entry point returns after a panic.
pub fn catch<T>(what: &str, fallback: T, f: impl FnOnce() -> T) -> T {
    run(what, f).unwrap_or(fallback)
}

/// Log a panic where it happens: the place, and the message when it is a
/// fixed one. A formatted message may quote what was typed (a reading, a
/// candidate), so it goes to the debug level only.
pub fn install_hook() {
    std::panic::set_hook(Box::new(|info| {
        let at = info
            .location()
            .map(|l| format!("{}:{}", l.file(), l.line()))
            .unwrap_or_default();
        let fixed = info.payload().downcast_ref::<&str>().copied();
        setsuna_core::diagnostics::log(format!(
            "panic at {at}: {}",
            fixed.unwrap_or("(message at debug level)")
        ));
        tracing::error!("setsuna-tsf panic at {at}: {}", fixed.unwrap_or("-"));
        tracing::debug!("setsuna-tsf panic: {info}");
    }));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_panic_returns_the_fallback() {
        assert_eq!(catch("test", 7, || -> i32 { panic!("boom") }), 7);
        assert_eq!(catch("test", 7, || 3), 3);
        assert_eq!(run("test", || -> i32 { panic!("boom {}", 1) }), None);
    }

    #[test]
    fn a_borrow_held_by_the_panic_is_released() {
        let cell = std::cell::RefCell::new(1);
        let r = run("test", || {
            let mut b = cell.borrow_mut();
            *b = 2;
            let v: Vec<i32> = Vec::new();
            v[0]
        });
        assert_eq!(r, None);
        assert_eq!(*cell.try_borrow_mut().expect("released"), 2);
    }
}
