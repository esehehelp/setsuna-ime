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
     * suggestion window; none while converting. setsuna_session_select
     * picks one by index. */
    const char *const *suggestions;
    const char *const *suggestion_comments;
    size_t n_suggestions;
} SetsunaView;

/* Load config.toml and spawn the engine daemon. `module_dir` (may be NULL)
 * is searched last for setsuna-engine and models. Never returns NULL. */
SetsunaSession *setsuna_session_new(const char *module_dir, SetsunaNotify notify,
                                  void *user);
/* Stop the daemon; no notify follows once this returns. */
void setsuna_session_free(SetsunaSession *s);

/* Reload config.toml, dropping any composition. */
void setsuna_session_configure(SetsunaSession *s);
/* setsuna_session_configure for the application `program` (its [[apps]]
 * settings and start state); null or empty when unknown. */
void setsuna_session_configure_app(SetsunaSession *s, const char *program);
/* A key press (X keysym, X keycode, SETSUNA_MOD_*); true when taken.
 * ` and Zenkaku_Hankaku turn the IME on or off. */
bool setsuna_session_key(SetsunaSession *s, uint32_t sym, uint32_t code,
                        uint32_t mods);
/* Apply the latest conversion result; true when the view changed. */
bool setsuna_session_apply_async(SetsunaSession *s);
/* Pick candidate `index` of the open list (moves the selection there). */
bool setsuna_session_select(SetsunaSession *s, size_t index);
/* Commit the composition (focus out, input method switch). */
void setsuna_session_flush(SetsunaSession *s);
/* Drop the composition (the app reset the input context). */
void setsuna_session_reset(SetsunaSession *s);

const SetsunaView *setsuna_session_view(const SetsunaSession *s);
size_t setsuna_session_page_size(const SetsunaSession *s);
/* Whether the field typed into is private (a password): no context, no
 * rewrite, nothing learned. */
void setsuna_session_set_private(SetsunaSession *s, bool private_field);
/* Turn the IME on or off, committing what is composed. */
void setsuna_session_toggle(SetsunaSession *s);
/* Whether the IME is on; off, every key goes to the app. */
bool setsuna_session_enabled(const SetsunaSession *s);
bool setsuna_session_has_engine(const SetsunaSession *s);

#ifdef __cplusplus
}
#endif

#endif
