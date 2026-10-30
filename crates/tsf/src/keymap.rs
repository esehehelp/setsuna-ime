//! Virtual-key → engine key classification.

use setsuna_core::char_forms::Form;
use setsuna_core::config::typed_symbol;
use setsuna_core::engine_bridge::KeyKind;
use std::collections::BTreeMap;

/// Windows virtual-key codes we care about. The rest fall through to the host.
/// `typed` is the char the keyboard layout makes of the key (`layout_char`):
/// a symbol key is looked up by it, not by its position, so `:` is `colon`
/// on JIS (its own key) and on US (Shift+`;`) alike.
pub fn classify_vk(
    vk: u32,
    shift: bool,
    typed: Option<char>,
    symbols: &BTreeMap<String, String>,
) -> Option<KeyKind> {
    match vk {
        0x08 => Some(KeyKind::Backspace), // VK_BACK
        0x09 if shift => Some(KeyKind::ShiftTab),
        0x09 => Some(KeyKind::Tab),    // VK_TAB
        0x0D => Some(KeyKind::Enter),  // VK_RETURN
        0x1B => Some(KeyKind::Escape), // VK_ESCAPE
        0x20 if shift => Some(KeyKind::ShiftSpace),
        0x20 => Some(KeyKind::Space),                     // VK_SPACE
        0x23 => Some(KeyKind::End),                       // VK_END
        0x24 => Some(KeyKind::Home),                      // VK_HOME
        0x2E => Some(KeyKind::Delete),                    // VK_DELETE
        0x75 => Some(KeyKind::ConvertTo(Form::Hiragana)), // VK_F6
        0x76 => Some(KeyKind::ConvertTo(Form::Katakana)),
        0x77 => Some(KeyKind::ConvertTo(Form::HalfKatakana)),
        0x78 => Some(KeyKind::ConvertTo(Form::FullAlnum)),
        0x79 => Some(KeyKind::ConvertTo(Form::HalfAlnum)), // VK_F10
        0x21 => Some(KeyKind::PageUp),                     // VK_PRIOR
        0x22 => Some(KeyKind::PageDown),                   // VK_NEXT
        0x25 if shift => Some(KeyKind::ShiftLeft),
        0x25 => Some(KeyKind::Left), // VK_LEFT
        0x26 => Some(KeyKind::Up),   // VK_UP
        0x27 if shift => Some(KeyKind::ShiftRight),
        0x27 => Some(KeyKind::Right),               // VK_RIGHT
        0x28 => Some(KeyKind::Down),                // VK_DOWN
        0xBD if !shift => Some(KeyKind::Char('-')), // VK_OEM_MINUS: chōon
        _ if typed.is_some_and(|c| c.is_ascii_punctuation() || c == '¥') => {
            let c = typed.unwrap_or_default();
            Some(KeyKind::Punctuation(match c {
                '¥' => typed_symbol('\\', false, symbols),
                c => typed_symbol(c, vk == VK_OEM_102, symbols),
            }))
        }
        0x30..=0x39 => Some(KeyKind::Char((vk as u8) as char)), // '0'..='9'
        0x41..=0x5A => {
            // 'A'..='Z'
            let base = b'a' + (vk as u8 - 0x41);
            let ch = if shift {
                (base as char).to_ascii_uppercase()
            } else {
                base as char
            };
            Some(KeyKind::Char(ch))
        }
        _ => None,
    }
}

/// The form Ctrl+U / I / O / P / T (MS-IME's aliases of F6–F10) convert to.
/// The IME takes them only while composing; otherwise they are the app's.
pub fn ctrl_form(vk: u32) -> Option<Form> {
    Some(match vk {
        0x55 => Form::Hiragana,     // U
        0x49 => Form::Katakana,     // I
        0x4F => Form::HalfKatakana, // O
        0x50 => Form::FullAlnum,    // P
        0x54 => Form::HalfAlnum,    // T
        _ => return None,
    })
}

/// VK_OEM_102: the JIS ろ key (`\` / `_`).
const VK_OEM_102: u32 = 0xE2;

/// The char the thread's keyboard layout makes of `vk` with `shift` alone
/// held (Caps Lock, Kana lock and dead keys left out), or `None`.
#[cfg(windows)]
pub fn layout_char(vk: u32, shift: bool) -> Option<char> {
    use windows::Win32::UI::Input::KeyboardAndMouse::{
        GetKeyboardLayout, MapVirtualKeyExW, ToUnicodeEx, MAPVK_VK_TO_VSC,
    };
    let mut state = [0u8; 256];
    if shift {
        state[0x10] = 0x80; // VK_SHIFT
    }
    let mut buf = [0u16; 4];
    // flag 4: leave the kernel's keyboard state (dead keys) as it is
    let n = unsafe {
        let hkl = GetKeyboardLayout(0);
        let scan = MapVirtualKeyExW(vk, MAPVK_VK_TO_VSC, hkl);
        ToUnicodeEx(vk, scan, &state, &mut buf, 4, hkl)
    };
    if n != 1 {
        return None;
    }
    char::from_u32(u32::from(buf[0]))
}

/// What a key of `[frontend] toggle_keys` / `on_keys` / `off_keys` does.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OnOff {
    Toggle,
    On,
    Off,
}

/// The modifiers held with a key, or named in a key's spec.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Mods {
    pub ctrl: bool,
    pub shift: bool,
    pub alt: bool,
    pub win: bool,
}

impl Mods {
    /// `TF_PRESERVEDKEY::uModifiers` of these (`TF_MOD_ALT` 1,
    /// `TF_MOD_CONTROL` 2, `TF_MOD_SHIFT` 4); `None` with Win, which TSF has
    /// no modifier for.
    pub fn tf_modifiers(self) -> Option<u32> {
        (!self.win)
            .then_some(u32::from(self.alt) | u32::from(self.ctrl) << 1 | u32::from(self.shift) << 2)
    }
}

/// A key of the config: its virtual-key codes (one physical key may send
/// several: 半角/全角 sends VK_KANJI, or VK_OEM_AUTO / VK_OEM_ENLW by
/// turns) with exactly these modifiers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KeySpec {
    pub vks: Vec<u32>,
    pub mods: Mods,
}

/// Read a key spec: `Zenkaku`, `Alt+\``, `Ctrl+Space`, `Shift+Muhenkan`,
/// `F12`, `0x1C`. Names ignore case; `None` for one this does not know.
pub fn parse_key(spec: &str) -> Option<KeySpec> {
    let parts: Vec<String> = spec.split('+').map(|p| p.trim().to_lowercase()).collect();
    let (key, modifiers) = parts.split_last()?;
    let mut mods = Mods::default();
    for m in modifiers {
        match m.as_str() {
            "ctrl" | "control" => mods.ctrl = true,
            "shift" => mods.shift = true,
            "alt" => mods.alt = true,
            "win" | "windows" => mods.win = true,
            _ => return None,
        }
    }
    let vks = match key.as_str() {
        // 半角/全角: VK_KANJI, and VK_OEM_AUTO / VK_OEM_ENLW, which the
        // Japanese layout sends by turns
        "zenkaku" | "hankaku" | "hankaku/zenkaku" | "kanji" | "半角/全角" | "半角" | "全角" =>
        {
            vec![0x19, 0xF3, 0xF4]
        }
        "henkan" | "convert" | "変換" => vec![0x1C],
        "muhenkan" | "nonconvert" | "無変換" => vec![0x1D],
        // VK_DBE_ALPHANUMERIC
        "eisu" | "英数" => vec![0xF0],
        // カタカナ/ひらがな: VK_DBE_HIRAGANA, VK_KANA
        "kana" | "hiragana" | "katakana" | "かな" | "カタカナ/ひらがな" => {
            vec![0xF2, 0x15]
        }
        "imeon" | "ime_on" => vec![0x16],
        "imeoff" | "ime_off" => vec![0x1A],
        "capslock" => vec![0x14],
        "space" => vec![0x20],
        "enter" | "return" => vec![0x0D],
        "tab" => vec![0x09],
        "esc" | "escape" => vec![0x1B],
        "backspace" => vec![0x08],
        // the key left of 1 on the US layout (@ on the Japanese one)
        "`" | "backquote" | "grave" => vec![0xC0],
        k if k.starts_with("0x") => vec![u32::from_str_radix(&k[2..], 16).ok()?],
        k if k.starts_with('f') && k.len() > 1 => {
            let n: u32 = k[1..].parse().ok()?;
            if !(1..=24).contains(&n) {
                return None;
            }
            vec![0x70 + n - 1]
        }
        k if k.len() == 1 && k.chars().all(|c| c.is_ascii_alphanumeric()) => {
            vec![u32::from(k.as_bytes()[0].to_ascii_uppercase())]
        }
        _ => return None,
    };
    Some(KeySpec { vks, mods })
}

/// The keys that turn the IME on and off, as the config names them.
#[derive(Debug, Clone, Default)]
pub struct OnOffKeys {
    keys: Vec<(KeySpec, OnOff)>,
}

impl OnOffKeys {
    /// `[frontend] toggle_keys`, `on_keys`, `off_keys`; a spec that does not
    /// read is left out (and returned, for the log).
    pub fn from_config(f: &setsuna_core::config::FrontendCfg) -> (Self, Vec<String>) {
        let mut keys = Vec::new();
        let mut bad = Vec::new();
        for (specs, role) in [
            (&f.toggle_keys, OnOff::Toggle),
            (&f.on_keys, OnOff::On),
            (&f.off_keys, OnOff::Off),
        ] {
            for spec in specs {
                match parse_key(spec) {
                    Some(k) => keys.push((k, role)),
                    None => bad.push(spec.clone()),
                }
            }
        }
        (Self { keys }, bad)
    }

    /// What `vk` held with exactly `mods` does, if it is one of the keys.
    pub fn role(&self, vk: u32, mods: Mods) -> Option<OnOff> {
        self.keys
            .iter()
            .find(|(k, _)| k.mods == mods && k.vks.contains(&vk))
            .map(|(_, role)| *role)
    }

    /// Each key TSF can preserve: its role, virtual-key code and
    /// `TF_MOD_*` modifiers.
    pub fn preservable(&self) -> Vec<(OnOff, u32, u32)> {
        self.keys
            .iter()
            .filter_map(|(k, role)| Some((*role, k.mods.tf_modifiers()?, &k.vks)))
            .flat_map(|(role, mods, vks)| vks.iter().map(move |&vk| (role, vk, mods)))
            .collect()
    }
}

#[cfg(test)]
fn default_symbols() -> BTreeMap<String, String> {
    setsuna_core::config::default_symbol_map()
}

/// The chars of the symbol keys on the Japanese (106 / 109) layout, as
/// `layout_char` reads them from kbd106.dll.
#[cfg(test)]
fn jis(vk: u32, shift: bool) -> Option<char> {
    let (plain, shifted) = match vk {
        0xBA => (':', '*'),
        0xBB => (';', '+'),
        0xBC => (',', '<'),
        0xBD => ('-', '='),
        0xBE => ('.', '>'),
        0xBF => ('/', '?'),
        0xC0 => ('@', '`'),
        0xDB => ('[', '{'),
        0xDC => ('\\', '|'),
        0xDD => (']', '}'),
        0xDE => ('^', '~'),
        0xE2 => ('\\', '_'),
        0x31..=0x39 => {
            let d = char::from_u32(vk).unwrap_or('0');
            (d, "!\"#$%&'()".chars().nth(vk as usize - 0x31).unwrap_or(d))
        }
        0x30 => return (!shift).then_some('0'),
        0x41..=0x5A => {
            let c = char::from_u32(vk + 0x20).unwrap_or('a');
            (c, c.to_ascii_uppercase())
        }
        _ => return None,
    };
    Some(if shift { shifted } else { plain })
}

/// The same on the US layout, where they differ.
#[cfg(test)]
fn us(vk: u32, shift: bool) -> Option<char> {
    let (plain, shifted) = match vk {
        0xBA => (';', ':'),
        0xBB => ('=', '+'),
        0xBD => ('-', '_'),
        0xC0 => ('`', '~'),
        0xDE => ('\'', '"'),
        0x32 => ('2', '@'),
        0x38 => ('8', '*'),
        _ => return jis(vk, shift),
    };
    Some(if shift { shifted } else { plain })
}

#[cfg(test)]
fn classify_default(vk: u32, shift: bool) -> Option<KeyKind> {
    classify_vk(vk, shift, jis(vk, shift), &default_symbols())
}

#[cfg(test)]
fn p(s: &str) -> Option<KeyKind> {
    Some(KeyKind::Punctuation(s.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shifted_digits_map_to_configured_symbols() {
        assert_eq!(classify_default(0x31, true), p("！"));
        assert_eq!(classify_default(0x38, true), p("（"));
        assert_eq!(classify_default(0x39, true), p("）"));
    }

    #[test]
    fn common_oem_symbols_are_consumed() {
        assert_eq!(classify_default(0xDB, false), p("「"));
        assert_eq!(classify_default(0xDD, false), p("」"));
        assert_eq!(classify_default(0xBF, false), p("・"));
        assert_eq!(classify_default(0xDC, false), p("￥"));
        assert_eq!(classify_default(0xE2, false), p("＼"));
    }

    #[test]
    fn jis_keys_type_the_symbol_printed_on_them() {
        assert_eq!(classify_default(0xBA, false), p("："));
        assert_eq!(classify_default(0xBA, true), p("*"));
        assert_eq!(classify_default(0xBB, false), p(";"));
        assert_eq!(classify_default(0xBD, true), p("＝"));
        assert_eq!(classify_default(0xC0, false), p("＠"));
        assert_eq!(classify_default(0xDE, false), p("^"));
        assert_eq!(classify_default(0xDE, true), p("～"));
        assert_eq!(classify_default(0xE2, true), p("_"));
        assert_eq!(classify_default(0x32, true), p("\""));
        assert_eq!(classify_default(0x37, true), p("'"));
    }

    #[test]
    fn us_keys_type_the_symbol_printed_on_them() {
        let us = |vk, shift| classify_vk(vk, shift, us(vk, shift), &default_symbols());
        assert_eq!(us(0xBA, false), p(";"));
        assert_eq!(us(0xBA, true), p("："));
        assert_eq!(us(0xDE, true), p("\""));
        assert_eq!(us(0x32, true), p("＠"));
        assert_eq!(us(0x38, true), p("*"));
        assert_eq!(us(0xBD, true), p("_"));
    }

    #[test]
    fn a_key_the_layout_types_nothing_with_is_not_a_symbol() {
        // JIS Shift+0
        assert_eq!(classify_default(0x30, true), Some(KeyKind::Char('0')));
        assert_eq!(classify_vk(0xBA, false, None, &default_symbols()), None);
    }

    #[test]
    fn custom_symbol_map_overrides_defaults() {
        let mut symbols = default_symbols();
        symbols.insert("slash".into(), "／".into());
        assert_eq!(classify_vk(0xBF, false, Some('/'), &symbols), p("／"));
    }

    #[test]
    fn editing_and_function_keys() {
        assert_eq!(classify_default(0x09, false), Some(KeyKind::Tab));
        assert_eq!(classify_default(0x09, true), Some(KeyKind::ShiftTab));
        assert_eq!(classify_default(0x20, true), Some(KeyKind::ShiftSpace));
        assert_eq!(classify_default(0x2E, false), Some(KeyKind::Delete));
        assert_eq!(
            classify_default(0x76, false),
            Some(KeyKind::ConvertTo(Form::Katakana))
        );
        assert_eq!(
            classify_default(0x79, false),
            Some(KeyKind::ConvertTo(Form::HalfAlnum))
        );
    }

    #[test]
    fn ctrl_letters_alias_the_function_keys() {
        assert_eq!(ctrl_form(0x49), Some(Form::Katakana));
        assert_eq!(ctrl_form(0x54), Some(Form::HalfAlnum));
        assert_eq!(ctrl_form(0x43), None); // Ctrl+C stays the app's
    }

    #[test]
    fn key_specs_read() {
        let k = parse_key("Zenkaku").unwrap();
        assert_eq!(k.vks, [0x19, 0xF3, 0xF4]);
        assert_eq!(k.mods, Mods::default());
        let k = parse_key("ctrl + Space").unwrap();
        assert_eq!(k.vks, [0x20]);
        assert!(k.mods.ctrl && !k.mods.shift);
        assert_eq!(k.mods.tf_modifiers(), Some(2));
        assert_eq!(parse_key("Alt+`").unwrap().vks, [0xC0]);
        assert_eq!(parse_key("Alt+`").unwrap().mods.tf_modifiers(), Some(1));
        assert_eq!(parse_key("Shift+Muhenkan").unwrap().vks, [0x1D]);
        assert_eq!(parse_key("Henkan").unwrap().vks, [0x1C]);
        assert_eq!(parse_key("F12").unwrap().vks, [0x7B]);
        assert_eq!(parse_key("0x1c").unwrap().vks, [0x1C]);
        assert_eq!(parse_key("j").unwrap().vks, [0x4A]);
        assert_eq!(parse_key("Win+Space").unwrap().mods.tf_modifiers(), None);
        assert_eq!(parse_key("Hyper+Space"), None);
        assert_eq!(parse_key("F25"), None);
        assert_eq!(parse_key(""), None);
    }

    #[test]
    fn default_on_off_keys() {
        let (keys, bad) = OnOffKeys::from_config(&setsuna_core::config::FrontendCfg::default());
        assert!(bad.is_empty(), "{bad:?}");
        let none = Mods::default();
        let alt = Mods { alt: true, ..none };
        let ctrl = Mods { ctrl: true, ..none };
        for vk in [0x19, 0xF3, 0xF4] {
            assert_eq!(keys.role(vk, none), Some(OnOff::Toggle));
            assert_eq!(keys.role(vk, alt), Some(OnOff::Toggle));
        }
        assert_eq!(keys.role(0xC0, alt), Some(OnOff::Toggle));
        // ` / @ typed without Alt is a char
        assert_eq!(keys.role(0xC0, none), None);
        assert_eq!(keys.role(0x16, none), Some(OnOff::On));
        assert_eq!(keys.role(0x1A, none), Some(OnOff::Off));
        // Ctrl+Space stays the app's unless the config asks for it
        assert_eq!(keys.role(0x20, ctrl), None);
        let preserved = keys.preservable();
        assert!(preserved.contains(&(OnOff::Toggle, 0x19, 0)));
        assert!(preserved.contains(&(OnOff::Toggle, 0xC0, 1)));
        assert!(preserved.contains(&(OnOff::On, 0x16, 0)));
    }

    #[test]
    fn configured_on_off_keys() {
        let f = setsuna_core::config::FrontendCfg {
            toggle_keys: vec!["Ctrl+Space".into(), "nonsense".into()],
            on_keys: vec!["Henkan".into()],
            off_keys: vec!["Muhenkan".into()],
            ..Default::default()
        };
        let (keys, bad) = OnOffKeys::from_config(&f);
        assert_eq!(bad, ["nonsense"]);
        let ctrl = Mods {
            ctrl: true,
            ..Mods::default()
        };
        assert_eq!(keys.role(0x20, ctrl), Some(OnOff::Toggle));
        assert_eq!(keys.role(0x20, Mods::default()), None);
        assert_eq!(keys.role(0x1C, Mods::default()), Some(OnOff::On));
        assert_eq!(keys.role(0x1D, Mods::default()), Some(OnOff::Off));
        assert_eq!(keys.role(0x19, Mods::default()), None);
    }

    #[test]
    fn minus_still_feeds_romaji_choon_without_shift() {
        assert_eq!(classify_default(0xBD, false), Some(KeyKind::Char('-')));
        assert_eq!(classify_default(0xBD, true), p("＝"));
    }
}
