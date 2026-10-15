//! The romaji tables through the bridge. The table in effect is the
//! process's, so this runs as its own test binary, one test at a time.

use setsuna_core::config::FrontendCfg;
use setsuna_core::engine_bridge::{Action, EngineBridge};

fn commit(table: &str, keys: &[Key]) -> String {
    let frontend = FrontendCfg {
        romaji_table: table.to_string(),
        ..FrontendCfg::default()
    };
    let mut bridge = EngineBridge::with_frontend(None, &frontend);
    for k in keys {
        match k {
            Key::Char(c) if c.is_ascii_uppercase() => bridge.handle_shifted_alpha(*c),
            Key::Char(c) => bridge.handle_char(*c),
            Key::Punct(p) => bridge.handle_punctuation(p),
        };
    }
    bridge
        .handle_enter()
        .actions
        .into_iter()
        .find_map(|a| match a {
            Action::Commit { text } => Some(text),
            _ => None,
        })
        .unwrap_or_default()
}

enum Key {
    Char(char),
    Punct(&'static str),
}

fn chars(s: &str) -> Vec<Key> {
    s.chars().map(Key::Char).collect()
}

#[test]
fn tables_read_what_is_typed() {
    // AZIK: its shortcuts, and `;` (a punctuation key) as っ
    assert_eq!(commit("azik", &chars("kzsr")), "かんする");
    let mut keys = chars("ka");
    keys.push(Key::Punct("；"));
    keys.extend(chars("ta"));
    assert_eq!(commit("azik", &keys), "かった");
    // kana input: keys as kana, Shift+E ぃ, @ (typed as ＠) a ゛
    let mut keys = chars("t");
    keys.push(Key::Punct("＠"));
    keys.extend(chars("3fE"));
    assert_eq!(commit("kana_jis", &keys), "があはぃ");
    // back to the built-in table
    assert_eq!(commit("standard", &chars("kanji")), "かんじ");
    // a table that cannot be read leaves the built-in one
    assert_eq!(commit("Z:/no/such/table.tsv", &chars("ka")), "か");
}
