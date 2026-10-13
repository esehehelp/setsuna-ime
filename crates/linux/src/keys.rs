//! X keysym → engine key classification: the fcitx5 counterpart of the TSF
//! crate's `keymap.rs`.
//!
//! fcitx5 hands the addon keysyms resolved by the keyboard layout, so the
//! symbol keys are looked up by the character they type rather than by
//! position (`setsuna_core::config::symbol_name`): `;` is `semicolon`, `(`
//! is `shift_8`, `"` is `double_quote`, whatever key made it. `\` is
//! `oem102_backslash` on the JIS ろ key and `yen` elsewhere, as on Windows.

use std::collections::BTreeMap;

use setsuna_core::char_forms::Form;
use setsuna_core::engine_bridge::KeyKind;

/// Modifier bits of [`Key::mods`] (`SETSUNA_MOD_*` in `include/setsuna.h`).
pub const MOD_SHIFT: u32 = 1;
pub const MOD_CTRL: u32 = 1 << 1;
pub const MOD_ALT: u32 = 1 << 2;
pub const MOD_SUPER: u32 = 1 << 3;

/// A key press as fcitx5 reports it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Key {
    /// X keysym (`fcitx::KeySym`).
    pub sym: u32,
    /// X keycode (evdev code + 8), used only to tell the JIS ろ key apart.
    pub code: u32,
    /// `MOD_*` bits.
    pub mods: u32,
}

impl Key {
    fn shift(&self) -> bool {
        self.mods & MOD_SHIFT != 0
    }
}

const BACKSPACE: u32 = 0xff08;
const TAB: u32 = 0xff09;
const RETURN: u32 = 0xff0d;
const ESCAPE: u32 = 0xff1b;
const HOME: u32 = 0xff50;
const LEFT: u32 = 0xff51;
const UP: u32 = 0xff52;
const RIGHT: u32 = 0xff53;
const DOWN: u32 = 0xff54;
const PAGE_UP: u32 = 0xff55;
const PAGE_DOWN: u32 = 0xff56;
const END: u32 = 0xff57;
const INSERT: u32 = 0xff63;
const KP_ENTER: u32 = 0xff8d;
const ISO_LEFT_TAB: u32 = 0xfe20;
const F6: u32 = 0xffc3;
const F7: u32 = 0xffc4;
const F8: u32 = 0xffc5;
const F9: u32 = 0xffc6;
const F10: u32 = 0xffc7;
const DELETE: u32 = 0xffff;
const SPACE: u32 = 0x20;
const GRAVE: u32 = 0x60;
const YEN: u32 = 0xa5;
const KANJI: u32 = 0xff21;
const ZENKAKU_HANKAKU: u32 = 0xff2a;

/// X keycode of the JIS ろ key (evdev `KEY_RO` 89 + 8).
const KEYCODE_RO: u32 = 97;

/// Whether `key` turns the IME on or off: \` (with or without Alt), the key
/// in the JIS 半角/全角 position on other keyboards, and 半角/全角 itself
/// (`Zenkaku_Hankaku`, or `Kanji` under some layouts). \` therefore never
/// types; Shift+\` is still `~`. fcitx5's default trigger keys already take
/// `Zenkaku_Hankaku` to switch input methods; it only reaches the addon
/// when they don't.
pub fn is_toggle(key: &Key) -> bool {
    matches!(key.sym, GRAVE | KANJI | ZENKAKU_HANKAKU) && key.mods & (MOD_CTRL | MOD_SUPER) == 0
}

/// The form F6–F10, or their MS-IME aliases Ctrl+U / I / O / P / T,
/// convert to.
fn form_key(key: &Key) -> Option<Form> {
    let ctrl = key.mods & (MOD_CTRL | MOD_ALT | MOD_SUPER) == MOD_CTRL;
    let plain = key.mods & (MOD_CTRL | MOD_ALT | MOD_SUPER) == 0;
    Some(match key.sym {
        F6 if plain => Form::Hiragana,
        F7 if plain => Form::Katakana,
        F8 if plain => Form::HalfKatakana,
        F9 if plain => Form::FullAlnum,
        F10 if plain => Form::HalfAlnum,
        0x75 | 0x55 if ctrl => Form::Hiragana,     // u
        0x69 | 0x49 if ctrl => Form::Katakana,     // i
        0x6f | 0x4f if ctrl => Form::HalfKatakana, // o
        0x70 | 0x50 if ctrl => Form::FullAlnum,    // p
        0x74 | 0x54 if ctrl => Form::HalfAlnum,    // t
        _ => return None,
    })
}

/// The engine key `key` is, or `None` for a key the IME leaves to the app.
/// Ctrl / Alt / Super combinations are never the IME's, but for the
/// Ctrl+U / I / O / P / T conversions (taken only while composing).
pub fn classify(key: &Key, symbols: &BTreeMap<String, String>) -> Option<KeyKind> {
    if let Some(form) = form_key(key) {
        return Some(KeyKind::ConvertTo(form));
    }
    if key.mods & (MOD_CTRL | MOD_ALT | MOD_SUPER) != 0 {
        return None;
    }
    let shift = key.shift();
    match key.sym {
        BACKSPACE => Some(KeyKind::Backspace),
        DELETE => Some(KeyKind::Delete),
        HOME => Some(KeyKind::Home),
        END => Some(KeyKind::End),
        ISO_LEFT_TAB => Some(KeyKind::ShiftTab),
        TAB if shift => Some(KeyKind::ShiftTab),
        TAB => Some(KeyKind::Tab),
        RETURN | KP_ENTER => Some(KeyKind::Enter),
        ESCAPE => Some(KeyKind::Escape),
        SPACE if shift => Some(KeyKind::ShiftSpace),
        SPACE => Some(KeyKind::Space),
        PAGE_UP => Some(KeyKind::PageUp),
        PAGE_DOWN => Some(KeyKind::PageDown),
        LEFT if shift => Some(KeyKind::ShiftLeft),
        LEFT => Some(KeyKind::Left),
        RIGHT if shift => Some(KeyKind::ShiftRight),
        RIGHT => Some(KeyKind::Right),
        UP => Some(KeyKind::Up),
        DOWN => Some(KeyKind::Down),
        // chōon in romaji
        0x2d => Some(KeyKind::Char('-')),
        0x30..=0x39 => Some(KeyKind::Char(key.sym as u8 as char)),
        // Letters by Shift, not by the keysym's case: Shift+letter starts the
        // temporary alphabet mode, Caps Lock alone still types romaji.
        0x41..=0x5a | 0x61..=0x7a => {
            let c = (key.sym as u8 as char).to_ascii_lowercase();
            Some(KeyKind::Char(if shift {
                c.to_ascii_uppercase()
            } else {
                c
            }))
        }
        0x21..=0x7e => Some(KeyKind::Punctuation(setsuna_core::config::typed_symbol(
            key.sym as u8 as char,
            key.code == KEYCODE_RO,
            symbols,
        ))),
        YEN => Some(KeyKind::Punctuation(
            symbols.get("yen").cloned().unwrap_or_else(|| "￥".into()),
        )),
        _ => None,
    }
}

/// Whether a key the app receives (the IME not composing) may change the
/// text before the caret or move the caret, making the left context stale.
/// Modifiers alone, Escape, function keys and Ctrl+C leave it.
pub fn breaks_context(key: &Key) -> bool {
    // Shift_L ..= Hyper_R, ISO_Level3_Shift, Mode_switch
    if matches!(key.sym, 0xffe1..=0xffee | 0xfe03 | 0xff7e) {
        return false;
    }
    if key.mods & (MOD_CTRL | MOD_ALT) != 0 {
        // Ctrl+C copies; other shortcuts may edit
        return !matches!(key.sym, 0x43 | 0x63);
    }
    matches!(
        key.sym,
        BACKSPACE | TAB | RETURN | KP_ENTER | SPACE | HOME..=END | INSERT | DELETE
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use setsuna_core::config::default_symbol_map;

    fn key(sym: u32, mods: u32) -> Key {
        Key { sym, code: 0, mods }
    }

    fn classify_default(sym: u32, mods: u32) -> Option<KeyKind> {
        classify(&key(sym, mods), &default_symbol_map())
    }

    fn p(s: &str) -> Option<KeyKind> {
        Some(KeyKind::Punctuation(s.to_string()))
    }

    #[test]
    fn letters_follow_shift_not_the_keysym_case() {
        assert_eq!(classify_default('a' as u32, 0), Some(KeyKind::Char('a')));
        assert_eq!(
            classify_default('A' as u32, MOD_SHIFT),
            Some(KeyKind::Char('A'))
        );
        // Caps Lock alone
        assert_eq!(classify_default('A' as u32, 0), Some(KeyKind::Char('a')));
        assert_eq!(classify_default('7' as u32, 0), Some(KeyKind::Char('7')));
        assert_eq!(classify_default('-' as u32, 0), Some(KeyKind::Char('-')));
    }

    #[test]
    fn symbols_map_by_the_char_they_type() {
        assert_eq!(classify_default('(' as u32, MOD_SHIFT), p("（"));
        assert_eq!(classify_default('!' as u32, MOD_SHIFT), p("！"));
        assert_eq!(classify_default(';' as u32, 0), p(";"));
        assert_eq!(classify_default('[' as u32, 0), p("「"));
        assert_eq!(classify_default('/' as u32, 0), p("・"));
        assert_eq!(classify_default('_' as u32, MOD_SHIFT), p("_"));
        assert_eq!(classify_default(YEN, 0), p("￥"));
        // as Japanese text usually writes them
        assert_eq!(classify_default('@' as u32, 0), p("＠"));
        assert_eq!(classify_default('^' as u32, 0), p("^"));
        assert_eq!(classify_default('"' as u32, MOD_SHIFT), p("\""));
    }

    #[test]
    fn backslash_is_yen_except_on_the_ro_key() {
        let mut symbols = default_symbol_map();
        symbols.insert("yen".into(), "￥".into());
        let bs = |code| {
            classify(
                &Key {
                    sym: '\\' as u32,
                    code,
                    mods: 0,
                },
                &symbols,
            )
        };
        assert_eq!(bs(51), p("￥"));
        assert_eq!(bs(132), p("￥"));
        assert_eq!(bs(KEYCODE_RO), p("＼"));
    }

    #[test]
    fn custom_symbol_map_overrides_defaults() {
        let mut symbols = default_symbol_map();
        symbols.insert("slash".into(), "／".into());
        assert_eq!(classify(&key('/' as u32, 0), &symbols), p("／"));
    }

    #[test]
    fn editing_keys_and_shift_arrows() {
        assert_eq!(classify_default(RETURN, 0), Some(KeyKind::Enter));
        assert_eq!(classify_default(KP_ENTER, 0), Some(KeyKind::Enter));
        assert_eq!(classify_default(LEFT, MOD_SHIFT), Some(KeyKind::ShiftLeft));
        assert_eq!(classify_default(RIGHT, 0), Some(KeyKind::Right));
        assert_eq!(
            classify_default(SPACE, MOD_SHIFT),
            Some(KeyKind::ShiftSpace)
        );
        assert_eq!(classify_default(TAB, 0), Some(KeyKind::Tab));
        assert_eq!(
            classify_default(ISO_LEFT_TAB, MOD_SHIFT),
            Some(KeyKind::ShiftTab)
        );
        assert_eq!(classify_default(DELETE, 0), Some(KeyKind::Delete));
        assert_eq!(classify_default(HOME, 0), Some(KeyKind::Home));
        assert_eq!(classify_default(0xffbe, 0), None); // F1
    }

    #[test]
    fn function_keys_and_their_ctrl_aliases_convert() {
        let form = |sym, mods| match classify_default(sym, mods) {
            Some(KeyKind::ConvertTo(f)) => Some(f),
            _ => None,
        };
        assert_eq!(form(F6, 0), Some(Form::Hiragana));
        assert_eq!(form(F7, 0), Some(Form::Katakana));
        assert_eq!(form(F8, 0), Some(Form::HalfKatakana));
        assert_eq!(form(F9, 0), Some(Form::FullAlnum));
        assert_eq!(form(F10, 0), Some(Form::HalfAlnum));
        assert_eq!(form('i' as u32, MOD_CTRL), Some(Form::Katakana));
        assert_eq!(form('t' as u32, MOD_CTRL), Some(Form::HalfAlnum));
        assert_eq!(form('t' as u32, MOD_CTRL | MOD_ALT), None);
        assert_eq!(form(F7, MOD_ALT), None);
    }

    #[test]
    fn shortcuts_are_left_to_the_app() {
        assert_eq!(classify_default('c' as u32, MOD_CTRL), None);
        assert_eq!(classify_default('a' as u32, MOD_CTRL), None);
        assert_eq!(classify_default('x' as u32, MOD_ALT), None);
        assert_eq!(classify_default(SPACE, MOD_SUPER), None);
    }

    #[test]
    fn backquote_and_zenkaku_hankaku_toggle() {
        assert!(is_toggle(&key(GRAVE, 0)));
        assert!(is_toggle(&key(GRAVE, MOD_ALT)));
        assert!(is_toggle(&key(ZENKAKU_HANKAKU, 0)));
        assert!(is_toggle(&key(KANJI, 0)));
        // Shift+` types ～
        assert!(!is_toggle(&key('~' as u32, MOD_SHIFT)));
        assert_eq!(classify_default('~' as u32, MOD_SHIFT), p("～"));
        assert!(!is_toggle(&key(GRAVE, MOD_CTRL)));
        assert!(!is_toggle(&key(ZENKAKU_HANKAKU, MOD_SUPER)));
    }

    #[test]
    fn keys_that_edit_or_move_break_the_context() {
        for sym in [
            RETURN, TAB, SPACE, BACKSPACE, DELETE, LEFT, DOWN, HOME, END, PAGE_UP,
        ] {
            assert!(breaks_context(&key(sym, 0)), "{sym:#x}");
        }
        // Ctrl+V pastes, Ctrl+Z undoes
        assert!(breaks_context(&key('v' as u32, MOD_CTRL)));
        assert!(breaks_context(&key('z' as u32, MOD_CTRL)));
    }

    #[test]
    fn modifiers_escape_function_keys_and_copy_do_not() {
        for sym in [0xffe1, 0xffe3, 0xffe9, 0xffeb, ESCAPE, 0xffbe, 0xffc9] {
            assert!(!breaks_context(&key(sym, 0)), "{sym:#x}");
        }
        assert!(!breaks_context(&key('c' as u32, MOD_CTRL)));
        assert!(!breaks_context(&key(0xffe3, MOD_CTRL)));
    }
}
