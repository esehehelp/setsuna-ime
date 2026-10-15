//! Emoji and kaomoji candidates (`data/emoji.tsv`, from Mozc's emoji and
//! emoticon data). A reading of an emoji puts up to `INLINE` of them
//! after the kana forms and the rest at the end of the list (as Mozc's
//! emoji rewriter puts them low); kaomoji go at the end only, never first.
//! えもじ / かおもじ list them all at the end. A reading of one followed by
//! a particle (にこにこを) gives it with the particle.

use std::collections::HashMap;
use std::sync::OnceLock;

const TABLE: &str = include_str!("../data/emoji.tsv");

/// Emoji put among the first candidates; the rest go at the end.
const INLINE: usize = 3;

/// Particles a reading of an emoji may end in.
const PARTICLES: &[&str] = &["を", "が", "は", "に", "で", "の", "と", "も", "や"];

#[derive(Default)]
struct Table {
    emoji: HashMap<String, Vec<&'static str>>,
    kaomoji: HashMap<String, Vec<&'static str>>,
    all_emoji: Vec<&'static str>,
    all_kaomoji: Vec<&'static str>,
}

fn table() -> &'static Table {
    static T: OnceLock<Table> = OnceLock::new();
    T.get_or_init(|| {
        let mut t = Table::default();
        for line in TABLE.lines() {
            if line.starts_with('#') {
                continue;
            }
            let mut cols = line.split('\t');
            let (Some(sym), Some(kind), Some(readings)) = (cols.next(), cols.next(), cols.next())
            else {
                continue;
            };
            let (map, all) = if kind == "k" {
                (&mut t.kaomoji, &mut t.all_kaomoji)
            } else {
                (&mut t.emoji, &mut t.all_emoji)
            };
            all.push(sym);
            for r in readings.split(' ') {
                map.entry(r.to_string()).or_default().push(sym);
            }
        }
        t
    })
}

/// The emoji and kaomoji of `reading`: (put among the first, at the end).
pub fn candidates(reading: &str) -> (Vec<String>, Vec<String>) {
    let t = table();
    match reading {
        "えもじ" => {
            return (
                Vec::new(),
                t.all_emoji.iter().map(|s| s.to_string()).collect(),
            )
        }
        "かおもじ" => {
            return (
                Vec::new(),
                t.all_kaomoji.iter().map(|s| s.to_string()).collect(),
            )
        }
        _ => {}
    }
    let (word, particle) = match (
        t.emoji.contains_key(reading) || t.kaomoji.contains_key(reading),
        reading,
    ) {
        (true, r) => (r, ""),
        (false, r) => match PARTICLES
            .iter()
            .find_map(|p| r.strip_suffix(p).map(|w| (w, *p)))
        {
            Some((w, p)) if w.chars().count() >= 2 => (w, p),
            _ => (r, ""),
        },
    };
    let with = |s: &&str| format!("{s}{particle}");
    let emoji: Vec<String> = t
        .emoji
        .get(word)
        .map(|v| v.iter().map(with).collect())
        .unwrap_or_default();
    let kaomoji: Vec<String> = t
        .kaomoji
        .get(word)
        .map(|v| v.iter().map(with).collect())
        .unwrap_or_default();
    // with a particle (かさの: ⛱の) they are a guess: at the end only
    let inline_n = if particle.is_empty() { INLINE } else { 0 };
    let inline: Vec<String> = emoji.iter().take(inline_n).cloned().collect();
    let mut end: Vec<String> = emoji.into_iter().skip(inline_n).collect();
    end.extend(kaomoji);
    (inline, end)
}

/// `candidates` with the emoji of `reading` in.
pub fn insert(reading: &str, candidates: Vec<String>) -> Vec<String> {
    let (inline, end) = self::candidates(reading);
    if inline.is_empty() && end.is_empty() {
        return candidates;
    }
    let mut out = crate::date_time::insert(reading, candidates, inline);
    for c in end {
        if !out.contains(&c) {
            out.push(c);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn list(v: &[&str]) -> Vec<String> {
        v.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn emoji_go_after_the_kana_and_kaomoji_last() {
        let out = insert("にこにこ", list(&["ニコニコ", "にこにこ", "にこ二個"]));
        assert_eq!(out[0], "ニコニコ");
        let kana = out.iter().position(|c| c == "にこにこ").unwrap();
        assert!(out[kana + 1].chars().all(|c| !c.is_ascii()) && out[kana + 1] != "にこ二個");
        // kaomoji last
        assert!(out.last().unwrap().contains('^'));
        assert!(out.contains(&"＼(^o^)／".to_string()));
    }

    #[test]
    fn a_particle_after_the_reading_stays() {
        // with a particle they go at the end only
        let (inline, end) = candidates("にこにこを");
        assert!(inline.is_empty() && !end.is_empty() && end.iter().all(|c| c.ends_with('を')));
        let (inline, end) = candidates("きょうを");
        assert!(inline.is_empty() && end.is_empty());
    }

    #[test]
    fn all_of_them_by_their_names() {
        let (_, all) = candidates("えもじ");
        assert!(all.len() > 1500);
        let (_, all) = candidates("かおもじ");
        assert!(all.len() > 300);
    }
}
