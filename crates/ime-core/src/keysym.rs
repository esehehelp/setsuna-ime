//! X keysym → engine key classification, for the frontends that hand over
//! keys as X keysyms: the fcitx5 addon (`crates/linux`) and the Android
//! keyboard (`crates/android`, whose keys and a hardware keyboard's are
//! sent as the keysyms they type). The counterpart of the TSF crate's
//! `keymap.rs`.
//!
//! A keysym is resolved by the keyboard layout already, so the symbol keys
//! are looked up by the character they type rather than by position
//! (`crate::config::symbol_name`): `;` is `semicolon`, `(` is `shift_8`, `"`
//! is `double_quote`, whatever key made it. `\` is `oem102_backslash` on the
//! JIS ろ key and `yen` elsewhere, as on Windows.

use std::collections::BTreeMap;

use crate::char_forms::Form;
use crate::engine_bridge::KeyKind;

/// Modifier bits of [`Key::mods`] (`SETSUNA_MOD_*` in
/// `crates/linux/include/setsuna.h`).
pub const MOD_SHIFT: u32 = 1;
pub const MOD_CTRL: u32 = 1 << 1;
pub const MOD_ALT: u32 = 1 << 2;
pub const MOD_SUPER: u32 = 1 << 3;

/// A key press as the frontend reports it.
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
const KP_SPACE: u32 = 0xff80;
const KP_TAB: u32 = 0xff89;
const KP_ENTER: u32 = 0xff8d;
const KP_HOME: u32 = 0xff95;
const KP_LEFT: u32 = 0xff96;
const KP_UP: u32 = 0xff97;
const KP_RIGHT: u32 = 0xff98;
const KP_DOWN: u32 = 0xff99;
const KP_PAGE_UP: u32 = 0xff9a;
const KP_PAGE_DOWN: u32 = 0xff9b;
const KP_END: u32 = 0xff9c;
const KP_INSERT: u32 = 0xff9e;
const KP_DELETE: u32 = 0xff9f;
const KP_MULTIPLY: u32 = 0xffaa;
const KP_ADD: u32 = 0xffab;
const KP_SEPARATOR: u32 = 0xffac;
const KP_SUBTRACT: u32 = 0xffad;
const KP_DECIMAL: u32 = 0xffae;
const KP_DIVIDE: u32 = 0xffaf;
const KP_0: u32 = 0xffb0;
const KP_9: u32 = 0xffb9;
const KP_EQUAL: u32 = 0xffbd;
const ISO_LEFT_TAB: u32 = 0xfe20;
const F1: u32 = 0xffbe;
const F6: u32 = 0xffc3;
const F7: u32 = 0xffc4;
const F8: u32 = 0xffc5;
const F9: u32 = 0xffc6;
const F10: u32 = 0xffc7;
const CAPS_LOCK: u32 = 0xffe5;
const DELETE: u32 = 0xffff;
const SPACE: u32 = 0x20;
const GRAVE: u32 = 0x60;
const YEN: u32 = 0xa5;
const KANJI: u32 = 0xff21;
const MUHENKAN: u32 = 0xff22;
const HENKAN: u32 = 0xff23;
const HIRAGANA: u32 = 0xff25;
const KATAKANA: u32 = 0xff26;
const HIRAGANA_KATAKANA: u32 = 0xff27;
const ZENKAKU: u32 = 0xff28;
const HANKAKU: u32 = 0xff29;
const ZENKAKU_HANKAKU: u32 = 0xff2a;
const KANA_SHIFT: u32 = 0xff2e;
const EISU_SHIFT: u32 = 0xff2f;
const EISU_TOGGLE: u32 = 0xff30;

/// X keycode of the JIS ろ key (evdev `KEY_RO` 89 + 8).
const KEYCODE_RO: u32 = 97;

/// What a key of `[frontend] toggle_keys` / `on_keys` / `off_keys` does.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OnOff {
    Toggle,
    On,
    Off,
}

/// A key of the config: its X keysyms with exact `MOD_*` modifiers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KeySpec {
    pub syms: Vec<u32>,
    pub mods: u32,
}

/// Read a key spec (`Zenkaku`, `Alt+\``, `Ctrl+Space`, `Henkan`, `Muhenkan`,
/// `F12`, `0xff23`), mirroring `crates/tsf/src/keymap.rs`. Names ignore case;
/// `None` for one this does not know.
pub fn parse_key(spec: &str) -> Option<KeySpec> {
    let parts: Vec<String> = spec.split('+').map(|p| p.trim().to_lowercase()).collect();
    let (key, modifiers) = parts.split_last()?;
    let mut mods = 0u32;
    for m in modifiers {
        match m.as_str() {
            "ctrl" | "control" => mods |= MOD_CTRL,
            "shift" => mods |= MOD_SHIFT,
            "alt" => mods |= MOD_ALT,
            "win" | "windows" | "super" | "meta" => mods |= MOD_SUPER,
            _ => return None,
        }
    }
    let syms = match key.as_str() {
        "zenkaku" | "hankaku" | "hankaku/zenkaku" | "kanji" | "半角/全角" | "半角" | "全角" =>
        {
            vec![ZENKAKU_HANKAKU, KANJI, ZENKAKU, HANKAKU]
        }
        "henkan" | "convert" | "変換" => vec![HENKAN],
        "muhenkan" | "nonconvert" | "無変換" => vec![MUHENKAN],
        "eisu" | "英数" => vec![EISU_TOGGLE, EISU_SHIFT],
        "kana" | "hiragana" | "katakana" | "かな" | "カタカナ/ひらがな" => {
            vec![HIRAGANA_KATAKANA, HIRAGANA, KATAKANA, KANA_SHIFT]
        }
        // Windows-only VK_IME_ON / VK_IME_OFF (in the default config): accept
        // without warning, matching no X keysym.
        "imeon" | "ime_on" | "imeoff" | "ime_off" => Vec::new(),
        "capslock" => vec![CAPS_LOCK],
        "space" => vec![SPACE],
        "enter" | "return" => vec![RETURN, KP_ENTER],
        "tab" => vec![TAB, ISO_LEFT_TAB],
        "esc" | "escape" => vec![ESCAPE],
        "backspace" => vec![BACKSPACE],
        "`" | "backquote" | "grave" => vec![GRAVE],
        k if k.starts_with("0x") => vec![u32::from_str_radix(&k[2..], 16).ok()?],
        k if k.starts_with('f') && k.len() > 1 => {
            let n: u32 = k[1..].parse().ok()?;
            if !(1..=24).contains(&n) {
                return None;
            }
            vec![F1 + n - 1]
        }
        k if k.len() == 1 && k.chars().all(|c| c.is_ascii_alphabetic()) => {
            let b = k.as_bytes()[0];
            vec![
                u32::from(b.to_ascii_lowercase()),
                u32::from(b.to_ascii_uppercase()),
            ]
        }
        k if k.len() == 1 && k.chars().all(|c| c.is_ascii_digit()) => {
            vec![u32::from(k.as_bytes()[0])]
        }
        _ => return None,
    };
    Some(KeySpec { syms, mods })
}

/// The keys that turn the IME on and off, as `[frontend] toggle_keys` /
/// `on_keys` / `off_keys` name them.
#[derive(Debug, Clone, Default)]
pub struct OnOffKeys {
    keys: Vec<(KeySpec, OnOff)>,
}

impl OnOffKeys {
    /// Parse `[frontend] toggle_keys`, `on_keys`, `off_keys`; unrecognized
    /// specs are left out and returned for the log.
    pub fn from_config(f: &crate::config::FrontendCfg) -> (Self, Vec<String>) {
        let mut keys = Vec::new();
        let mut bad = Vec::new();
        for (specs, role) in [
            (&f.toggle_keys, OnOff::Toggle),
            (&f.on_keys, OnOff::On),
            (&f.off_keys, OnOff::Off),
        ] {
            for spec in specs {
                match parse_key(spec) {
                    Some(k) if !k.syms.is_empty() => keys.push((k, role)),
                    Some(_) => {}
                    None => bad.push(spec.clone()),
                }
            }
        }
        (Self { keys }, bad)
    }

    /// What `key` does, if it matches one of the configured keys with exact
    /// modifiers.
    pub fn role(&self, key: &Key) -> Option<OnOff> {
        self.keys
            .iter()
            .find(|(k, _)| k.mods == key.mods && k.syms.contains(&key.sym))
            .map(|(_, role)| *role)
    }
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

/// Map numeric keypad keysyms onto their main-keyboard counterparts so numpad
/// digits, operators and navigation keys work during composition and candidate
/// selection rather than leaking through to the application.
fn normalize_keypad(sym: u32) -> u32 {
    match sym {
        KP_SPACE => SPACE,
        KP_TAB => TAB,
        KP_ENTER => RETURN,
        KP_HOME => HOME,
        KP_LEFT => LEFT,
        KP_UP => UP,
        KP_RIGHT => RIGHT,
        KP_DOWN => DOWN,
        KP_PAGE_UP => PAGE_UP,
        KP_PAGE_DOWN => PAGE_DOWN,
        KP_END => END,
        KP_INSERT => INSERT,
        KP_DELETE => DELETE,
        KP_EQUAL => '=' as u32,
        KP_MULTIPLY => '*' as u32,
        KP_ADD => '+' as u32,
        KP_SEPARATOR => ',' as u32,
        KP_SUBTRACT => '-' as u32,
        KP_DECIMAL => '.' as u32,
        KP_DIVIDE => '/' as u32,
        KP_0..=KP_9 => '0' as u32 + (sym - KP_0),
        other => other,
    }
}

/// The engine key `key` is, or `None` for a key the IME leaves to the app.
/// Ctrl / Alt / Super combinations are never the IME's, but for the
/// Ctrl+U / I / O / P / T conversions (taken only while composing).
pub fn classify(key: &Key, symbols: &BTreeMap<String, String>) -> Option<KeyKind> {
    if let Some(form) = form_key(key) {
        return Some(KeyKind::ConvertTo(form));
    }
    let sym = normalize_keypad(key.sym);
    // Ctrl+Delete: forget or hide the candidate (taken only while converting)
    if sym == DELETE && key.mods & (MOD_CTRL | MOD_ALT | MOD_SUPER) == MOD_CTRL {
        return Some(KeyKind::ForgetCandidate);
    }
    if key.mods & (MOD_CTRL | MOD_ALT | MOD_SUPER) != 0 {
        return None;
    }
    let shift = key.shift();
    match sym {
        BACKSPACE => Some(KeyKind::Backspace),
        DELETE => Some(KeyKind::Delete),
        HOME => Some(KeyKind::Home),
        END => Some(KeyKind::End),
        ISO_LEFT_TAB => Some(KeyKind::ShiftTab),
        TAB if shift => Some(KeyKind::ShiftTab),
        TAB => Some(KeyKind::Tab),
        RETURN => Some(KeyKind::Enter),
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
        0x30..=0x39 => Some(KeyKind::Char(sym as u8 as char)),
        // Letters by Shift, not by the keysym's case: Shift+letter starts the
        // temporary alphabet mode, Caps Lock alone still types romaji.
        0x41..=0x5a | 0x61..=0x7a => {
            let c = (sym as u8 as char).to_ascii_lowercase();
            Some(KeyKind::Char(if shift {
                c.to_ascii_uppercase()
            } else {
                c
            }))
        }
        0x21..=0x7e => Some(KeyKind::Punctuation(crate::config::typed_symbol(
            sym as u8 as char,
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
        normalize_keypad(key.sym),
        BACKSPACE | TAB | RETURN | SPACE | HOME..=END | INSERT | DELETE
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::default_symbol_map;

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
    fn numpad_digits_operators_and_navigation_are_classified() {
        for d in 0..=9u32 {
            assert_eq!(
                classify_default(KP_0 + d, 0),
                Some(KeyKind::Char((b'0' + d as u8) as char))
            );
        }
        assert_eq!(classify_default(KP_SPACE, 0), Some(KeyKind::Space));
        assert_eq!(
            classify_default(KP_SPACE, MOD_SHIFT),
            Some(KeyKind::ShiftSpace)
        );
        assert_eq!(classify_default(KP_TAB, 0), Some(KeyKind::Tab));
        assert_eq!(classify_default(KP_TAB, MOD_SHIFT), Some(KeyKind::ShiftTab));
        assert_eq!(classify_default(KP_HOME, 0), Some(KeyKind::Home));
        assert_eq!(classify_default(KP_END, 0), Some(KeyKind::End));
        assert_eq!(classify_default(KP_LEFT, 0), Some(KeyKind::Left));
        assert_eq!(
            classify_default(KP_LEFT, MOD_SHIFT),
            Some(KeyKind::ShiftLeft)
        );
        assert_eq!(classify_default(KP_RIGHT, 0), Some(KeyKind::Right));
        assert_eq!(
            classify_default(KP_RIGHT, MOD_SHIFT),
            Some(KeyKind::ShiftRight)
        );
        assert_eq!(classify_default(KP_UP, 0), Some(KeyKind::Up));
        assert_eq!(classify_default(KP_DOWN, 0), Some(KeyKind::Down));
        assert_eq!(classify_default(KP_PAGE_UP, 0), Some(KeyKind::PageUp));
        assert_eq!(classify_default(KP_PAGE_DOWN, 0), Some(KeyKind::PageDown));
        assert_eq!(classify_default(KP_DELETE, 0), Some(KeyKind::Delete));
        assert_eq!(
            classify_default(KP_DELETE, MOD_CTRL),
            Some(KeyKind::ForgetCandidate)
        );
        assert_eq!(classify_default(KP_SUBTRACT, 0), Some(KeyKind::Char('-')));
        assert_eq!(classify_default(KP_DECIMAL, 0), p("。"));
        assert_eq!(classify_default(KP_SEPARATOR, 0), p("、"));
        assert_eq!(classify_default(KP_DIVIDE, 0), p("・"));
        assert_eq!(classify_default(KP_ADD, 0), classify_default('+' as u32, 0));
        assert_eq!(
            classify_default(KP_MULTIPLY, 0),
            classify_default('*' as u32, 0)
        );
        assert_eq!(
            classify_default(KP_EQUAL, 0),
            classify_default('=' as u32, 0)
        );
        for sym in [
            KP_ENTER, KP_SPACE, KP_TAB, KP_LEFT, KP_HOME, KP_DELETE, KP_INSERT,
        ] {
            assert!(breaks_context(&key(sym, 0)), "{sym:#x}");
        }
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
    fn default_on_off_keys_take_the_bare_grave_on_linux_and_never_the_shifted_one() {
        let (keys, bad) = OnOffKeys::from_config(&crate::config::FrontendCfg::default());
        assert!(bad.is_empty(), "{bad:?}");
        assert_eq!(keys.role(&key(ZENKAKU_HANKAKU, 0)), Some(OnOff::Toggle));
        assert_eq!(keys.role(&key(KANJI, 0)), Some(OnOff::Toggle));
        assert_eq!(
            keys.role(&key(ZENKAKU_HANKAKU, MOD_ALT)),
            Some(OnOff::Toggle)
        );
        assert_eq!(keys.role(&key(GRAVE, MOD_ALT)), Some(OnOff::Toggle));
        // Bare ` (US: the key where 半角/全角 is) toggles on Linux, where
        // Alt+` is the desktop's; elsewhere it stays free to type.
        assert_eq!(
            keys.role(&key(GRAVE, 0)),
            cfg!(target_os = "linux").then_some(OnOff::Toggle)
        );
        // Shift+@ (JIS backquote) stays free to type.
        assert_eq!(keys.role(&key(GRAVE, MOD_SHIFT)), None);
        // Shift+` on US types ～
        assert_eq!(keys.role(&key('~' as u32, MOD_SHIFT)), None);
        assert_eq!(classify_default('~' as u32, MOD_SHIFT), p("～"));
        assert_eq!(keys.role(&key(GRAVE, MOD_CTRL)), None);
        assert_eq!(keys.role(&key(ZENKAKU_HANKAKU, MOD_SUPER)), None);
    }

    #[test]
    fn custom_toggle_on_and_off_keys_parse() {
        let mut f = crate::config::FrontendCfg::default();
        f.toggle_keys = vec![
            "Ctrl+Space".into(),
            "`".into(),
            "Ctrl+A".into(),
            "NotAKey".into(),
        ];
        f.on_keys = vec!["Henkan".into(), "Kana".into()];
        f.off_keys = vec!["Muhenkan".into(), "Eisu".into()];
        let (keys, bad) = OnOffKeys::from_config(&f);
        assert_eq!(bad, vec!["NotAKey"]);
        assert_eq!(keys.role(&key(SPACE, MOD_CTRL)), Some(OnOff::Toggle));
        assert_eq!(keys.role(&key(SPACE, 0)), None);
        assert_eq!(keys.role(&key(GRAVE, 0)), Some(OnOff::Toggle));
        assert_eq!(keys.role(&key(GRAVE, MOD_SHIFT)), None);
        assert_eq!(keys.role(&key('a' as u32, MOD_CTRL)), Some(OnOff::Toggle));
        assert_eq!(keys.role(&key('A' as u32, MOD_CTRL)), Some(OnOff::Toggle));
        assert_eq!(keys.role(&key(HENKAN, 0)), Some(OnOff::On));
        assert_eq!(keys.role(&key(HIRAGANA_KATAKANA, 0)), Some(OnOff::On));
        assert_eq!(keys.role(&key(MUHENKAN, 0)), Some(OnOff::Off));
        assert_eq!(keys.role(&key(EISU_TOGGLE, 0)), Some(OnOff::Off));
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
