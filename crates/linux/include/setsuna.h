/*
 * C ABI of the Setsuna Linux frontend (crates/linux/src/ffi.rs).
 *
 * All calls happen on the fcitx5 event-loop thread, except the notify
 * callback, which the engine client calls from its own thread.
 */
#ifndef SETSUNA_H
#define SETSUNA_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Modifier bits of setsuna_session_key's `mods`. */
#define SETSUNA_MOD_SHIFT (1u << 0)
#define SETSUNA_MOD_CTRL (1u << 1)
#define SETSUNA_MOD_ALT (1u << 2)
#define SETSUNA_MOD_SUPER (1u << 3)

typedef struct SetsunaSession SetsunaSession;

/* Wakes the event loop when a conversion result is ready. */
typedef void (*SetsunaNotify)(void *user);

/* What to show; pointers valid until the next call on the session. */
typedef struct {
    const char *const *commits;
    size_t n_commits;
    const char *const *preedit; /* segments */
    size_t n_preedit;
    ptrdiff_t focus;            /* focused segment, -1 for none */
    size_t cursor;              /* caret, byte offset in the preedit */
    const char *const *candidates;
    const char *const *comments; /* per candidate, "" for none */
    size_t n_candidates;
    size_t selected;
    const char *reading;  /* being converted, "" while typing */
    const char *form;     /* F6-F10 form shown ("全角カタカナ"), "" for none */
    const char *form_key; /* its key ("F7") */
    /* Conversions of what is typed (the preedit keeps the kana), for a
     * suggestion window; none while converting. By index,
     * setsuna_session_select opens the conversion on one and
     * setsuna_session_pick commits it (the addon's click). */
    const char *const *suggestions;
    const char *const *suggestion_comments;
    size_t n_suggestions;
    const char *status; /* a line under the open list (書き直し中…), "" for none */
} SetsunaView;

/* Load config.toml and spawn the daemons it asks for (the engine; the
 * reranker and rewriter when set up). `module_dir` (may be NULL) is searched
 * last for them and the models. A failure there leaves the defaults; NULL
 * only when not even they could be set up (every call then does nothing and
 * setsuna_session_view returns NULL). */
SetsunaSession *setsuna_session_new(const char *module_dir, SetsunaNotify notify,
                                  void *user);
/* Stop the daemons; no notify follows once this returns. */
void setsuna_session_free(SetsunaSession *s);

/* Reload config.toml if it changed (and restart daemons whose settings
 * changed or that exited). */
void setsuna_session_configure(SetsunaSession *s);

/* App-specific initial mode from [[apps]] (setsuna_session_configure_app). */
#define SETSUNA_APP_START_KEEP 0
#define SETSUNA_APP_START_ON 1
#define SETSUNA_APP_START_OFF 2

/* Apply config.toml and [[apps]] settings for `program` (null or empty when
 * unknown), reloading only when the config file, effective app settings or
 * daemon liveness changed. Returns SETSUNA_APP_START_* without changing
 * the session's enabled flag (the frontend owns per-context on/off state via
 * setsuna_session_set_enabled). */
int setsuna_session_configure_app(SetsunaSession *s, const char *program);
/* What the field typed into is (setsuna_session_key's `field`). */
#define SETSUNA_FIELD_NORMAL 0u
#define SETSUNA_FIELD_PRIVATE 1u /* no context, no rewrite, nothing learned */
#define SETSUNA_FIELD_DIRECT 2u  /* a URL, a number, a password: keys go to the app */

/* A key press (X keysym, X keycode, SETSUNA_MOD_*) typed into `field`, with
 * `before_caret` the app's text before the caret (NULL when unknown): a key
 * starting a composition takes both up. True when taken. Keys in
 * [frontend] toggle_keys / on_keys / off_keys switch the IME on or off. */
bool setsuna_session_key(SetsunaSession *s, uint32_t sym, uint32_t code,
                        uint32_t mods, uint32_t field, const char *before_caret);
/* Apply the latest conversion result; true when the view changed. */
bool setsuna_session_apply_async(SetsunaSession *s);
/* Pick candidate `index` of the open list (moves the selection there). */
bool setsuna_session_select(SetsunaSession *s, size_t index);
/* Pick candidate `index` of the open list or suggestions and accept it:
 * commits it, or while editing segments fixes the focused segment and moves
 * to the next (committing after the last), matching digit keys. */
bool setsuna_session_pick(SetsunaSession *s, size_t index);
/* Commit what is currently shown without waiting for a pending conversion
 * (focus out, input method switch). */
void setsuna_session_flush(SetsunaSession *s);
/* Drop the composition (the app reset the input context). */
void setsuna_session_reset(SetsunaSession *s);

const SetsunaView *setsuna_session_view(const SetsunaSession *s);
size_t setsuna_session_page_size(const SetsunaSession *s);
/* How long (ms) until a deferred preedit is shown anyway (0: none waits);
 * then call setsuna_session_tick (true when the view changed). */
uint32_t setsuna_session_deferred_ms(const SetsunaSession *s);
bool setsuna_session_tick(SetsunaSession *s);
/* Turn the IME on or off, committing what is composed. */
void setsuna_session_toggle(SetsunaSession *s);
/* Set the IME on or off without committing (per-context state restore). */
void setsuna_session_set_enabled(SetsunaSession *s, bool on);
/* Whether the IME is on; off, every key goes to the app. */
bool setsuna_session_enabled(const SetsunaSession *s);
bool setsuna_session_has_engine(const SetsunaSession *s);
/* Rewrite mode is on (a rewriter runs): the mode shows 直. */
bool setsuna_session_rewrite_mode(const SetsunaSession *s);

#ifdef __cplusplus
}
#endif

#endif
