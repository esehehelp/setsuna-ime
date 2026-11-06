//! Japanese / English segmentation of typed romaji.
//!
//! The letters of a composition are either romaji (Japanese, shown as kana)
//! or English words (shown as typed). `MixedModel::segment` picks the split
//! with the lowest cost under two language models and a cost for switching
//! between them:
//!
//! - a Japanese span costs −ln P of its kana under a kana trigram model of
//!   corpus readings (Witten–Bell interpolation), plus `leftover` for each
//!   letter that reads as no romaji (`google` → ごおgぇ keeps a `g`) and
//!   `upper` for each letter typed with Shift;
//! - an English span is word pieces joined by symbols (`example.com`,
//!   `e-mail`); a piece costs −ln of a mixture of its frequency among the
//!   Latin words of Japanese and of English web text and, for an unseen
//!   word, a letter trigram model of those words;
//! - starting an English span inside Japanese costs `switch_in`, going back
//!   `switch_out`. The text before the caret sets the language the input
//!   starts in.
//!
//! A word the model knows, typed in lowercase, that reads better as a
//! commoner word and a form of する (`dlsite`: dl + して) is split so: the
//! sentence is preferred by `sentence` over the rarer word or name.
//!
//! `MixedModel::cased` then gives the English its case: a name of the names
//! dictionary typed in lowercase as the name is written (github → GitHub),
//! and an English sentence (two words or more) a capital at its start, `I`
//! as `I` and its `.` `,` `!` `?` as typed.
//!
//! The model files are counts: `kana.tsv` (`<n-gram>\t<count>`, n = 1..3,
//! `^` a reading's start, `0` any digit, `#` any other character) and
//! `latin.tsv` (`<word>\t<count in Japanese text>\t<count in English text>`),
//! with `params.txt` (`name=value` lines) overriding [`MixedParams`]. The
//! names dictionary (`<name as written>\t<weight>` lines, the weight in
//! (0, 1], 1 for the best known) is a file of its own.

use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};
use std::path::Path;
use std::sync::{Arc, Mutex, OnceLock};

use anyhow::{Context, Result};

use crate::composing_text::{pending_kana, romaji_units, UnitKind};
use crate::config::NameCase;
use crate::lexicon::Lexicon;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Lang {
    Ja,
    En,
}

/// Chars `start..end` of the typed text, in one language.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Span {
    pub lang: Lang,
    pub start: usize,
    pub end: usize,
}

#[derive(Debug, Clone)]
pub struct MixedParams {
    pub switch_in: f64,
    pub switch_out: f64,
    /// An English span right after another (no space between).
    pub en_en: f64,
    /// English as the first span with no text before the caret.
    pub start_en: f64,
    pub leftover: f64,
    /// Each letter left over after the first in one Japanese span (ぱyぱy
    /// for paypay): as often several slips as an English word.
    pub leftover_more: f64,
    pub upper: f64,
    /// Weight of the word frequencies against the letter model.
    pub known: f64,
    /// Weight of Japanese-text frequencies against English-text ones.
    pub ja_text: f64,
    pub symbol: f64,
    pub digit: f64,
    /// An English word going from two capitals to lowercase (`SCmo`: an
    /// acronym, then romaji).
    pub case_break: f64,
    /// The frequency of a name of the names dictionary of weight 1 among
    /// Latin words (its weight scales it).
    pub name: f64,
    /// What a sentence is preferred by over a word it overlaps, typed in
    /// lowercase (`dlsite`: dl + して over the name DLsite).
    pub sentence: f64,
    /// An English word typed in lowercase that is no word or name the model
    /// knows (nor the start of one, while it is typed): as often a slip of
    /// romaji (`kagriare` for かげりあれ) as a new word.
    pub oov: f64,
    /// A Latin word the model knows that romaji reads with a letter left
    /// (`bug` ぶg, `zip` じp, `kpi` kぴ) is English inside Japanese when that
    /// costs at most this over the best reading (0: never).
    pub letter_word: f64,
    /// The frequency among Latin words such a word needs: a rarer one
    /// (`furq`, `chk`) is likelier a slip of the romaji around it.
    pub letter_word_floor: f64,
    /// With the reading lexicon: a Latin word the model knows that romaji
    /// reads wholly as kana (`issue` いっすえ) is English inside Japanese
    /// when its kana cost as Japanese (the lexicon's cheapest words over
    /// them and `kana_word_window` kana each side, less those of the sides
    /// alone) more than this over the word's cost among Latin words.
    pub kana_word: f64,
    /// The frequency among Latin words such a word needs.
    pub kana_word_floor: f64,
    /// The kana each side of the word the lexicon reads it with (0: the
    /// whole Japanese span).
    pub kana_word_window: f64,
}

impl Default for MixedParams {
    fn default() -> Self {
        Self {
            switch_in: 6.0,
            switch_out: 6.0,
            en_en: 6.0,
            start_en: 3.0,
            leftover: 8.0,
            leftover_more: 0.0,
            upper: 8.0,
            known: 0.9,
            ja_text: 0.7,
            symbol: 3.0,
            digit: 1.0,
            case_break: 8.0,
            name: 1e-4,
            sentence: 1.0,
            oov: 0.0,
            letter_word: 4.0,
            letter_word_floor: 4e-5,
            kana_word: 8.0,
            kana_word_floor: 4e-5,
            kana_word_window: 4.0,
        }
    }
}

impl MixedParams {
    pub const NAMES: [&'static str; 20] = [
        "switch_in",
        "switch_out",
        "en_en",
        "start_en",
        "leftover",
        "leftover_more",
        "upper",
        "known",
        "ja_text",
        "symbol",
        "digit",
        "case_break",
        "name",
        "sentence",
        "oov",
        "letter_word",
        "letter_word_floor",
        "kana_word",
        "kana_word_floor",
        "kana_word_window",
    ];

    pub fn field(&mut self, name: &str) -> Option<&mut f64> {
        Some(match name {
            "switch_in" => &mut self.switch_in,
            "switch_out" => &mut self.switch_out,
            "en_en" => &mut self.en_en,
            "start_en" => &mut self.start_en,
            "leftover" => &mut self.leftover,
            "leftover_more" => &mut self.leftover_more,
            "upper" => &mut self.upper,
            "known" => &mut self.known,
            "ja_text" => &mut self.ja_text,
            "symbol" => &mut self.symbol,
            "digit" => &mut self.digit,
            "case_break" => &mut self.case_break,
            "name" => &mut self.name,
            "sentence" => &mut self.sentence,
            "oov" => &mut self.oov,
            "letter_word" => &mut self.letter_word,
            "letter_word_floor" => &mut self.letter_word_floor,
            "kana_word" => &mut self.kana_word,
            "kana_word_floor" => &mut self.kana_word_floor,
            "kana_word_window" => &mut self.kana_word_window,
            _ => return None,
        })
    }

    /// Set a parameter by name (`switch_in=7`).
    pub fn set(&mut self, assignment: &str) -> Result<()> {
        let (name, value) = assignment
            .split_once('=')
            .with_context(|| format!("{assignment}: expected name=value"))?;
        let value: f64 = value
            .trim()
            .parse()
            .with_context(|| assignment.to_string())?;
        *self
            .field(name.trim())
            .with_context(|| format!("unknown parameter {name}"))? = value;
        Ok(())
    }

    /// `name=value` lines.
    pub fn to_text(&self) -> String {
        let mut p = self.clone();
        Self::NAMES
            .iter()
            .map(|name| format!("{name}={}\n", p.field(name).map_or(0.0, |v| *v)))
            .collect()
    }
}

pub struct MixedModel {
    kana: CharLm,
    letters: CharLm,
    words: FastMap<String, (u32, u32)>,
    /// The words sorted, with running totals of their counts (Japanese,
    /// English text) before each: the mass of a prefix, for a word still
    /// being typed.
    sorted: Vec<String>,
    running: Vec<(u64, u64)>,
    ja_total: f64,
    en_total: f64,
    /// The names dictionary by lowercase form: the name as written, its
    /// weight and whether it is a common one (an acronym or an ordinary
    /// word too: OK, Word).
    names: FastMap<String, (String, f64, bool)>,
    pub params: MixedParams,
    /// The reading lexicon's cost of kana `kana_words` asked for, by the
    /// lexicon (its address) and the kana: typing asks again for the same
    /// at each key.
    lexicon_costs: Mutex<HashMap<(usize, String), f64>>,
}

/// Kana costs `MixedModel::lexicon_costs` keeps before it starts over.
const LEXICON_COSTS: usize = 4096;

impl std::fmt::Debug for MixedModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MixedModel")
            .field("words", &self.words.len())
            .field("names", &self.names.len())
            .field("params", &self.params)
            .finish_non_exhaustive()
    }
}

/// The model in `dir` with the names dictionary `names`, loaded once per
/// process; `None` (logged) when the model cannot be read (a names
/// dictionary that cannot be read is logged and left out).
pub fn shared(dir: &str, names: Option<&str>) -> Option<Arc<MixedModel>> {
    type Cache = HashMap<(String, Option<String>), Option<Arc<MixedModel>>>;
    static CACHE: OnceLock<Mutex<Cache>> = OnceLock::new();
    let mut cache = CACHE
        .get_or_init(Default::default)
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    cache
        .entry((dir.to_string(), names.map(str::to_string)))
        .or_insert_with(|| match MixedModel::load(Path::new(dir)) {
            Ok(mut model) => {
                if let Some(path) = names {
                    match std::fs::read_to_string(path) {
                        Ok(tsv) => model.add_names(&tsv),
                        Err(e) => crate::diagnostics::log(format!("names {path}: {e}")),
                    }
                }
                Some(Arc::new(model))
            }
            Err(e) => {
                crate::diagnostics::log(format!("mixed model {dir}: {e:#}"));
                None
            }
        })
        .clone()
}

/// Longest English span considered, in chars.
const MAX_EN: usize = 40;

/// Particles (and words used like them) that end an English word typed
/// right before them, as romaji: が を で の に は と も へ から より, and ら
/// before one (Judd + らは). 側 (gawa) starts with が, です with で. Not
/// まで: `Ollamade` would split before `ma` (Olla + まで); not ら alone:
/// `Sakura` would.
const PARTICLES: &[&str] = &[
    "ga", "wo", "de", "no", "ni", "ha", "to", "mo", "he", "kara", "yori", "raga", "rawo", "rade",
    "rano", "rani", "raha", "rato", "ramo",
];

/// The forms of する a Latin word goes on with (DLして, OKした, DLする),
/// two kana or more: し, さ, せ alone would read too much as one (these:
/// the + せ).
const SURU: &[&str] = &[
    "して", "した", "しな", "しま", "しよ", "しろ", "する", "すれ", "され", "させ", "せず", "せよ",
];

impl MixedModel {
    pub fn load(dir: &Path) -> Result<Self> {
        let read = |name: &str| {
            let path = dir.join(name);
            std::fs::read_to_string(&path).with_context(|| path.display().to_string())
        };
        let params = read("params.txt").ok();
        Self::from_texts(&read("kana.tsv")?, &read("latin.tsv")?, params.as_deref())
    }

    /// A model from the contents of its files.
    pub fn from_texts(kana_tsv: &str, latin_tsv: &str, params_txt: Option<&str>) -> Result<Self> {
        let mut kana = CharLm::default();
        for line in kana_tsv.lines() {
            if let Some((gram, count)) = line.split_once('\t') {
                kana.add(gram, count.parse().unwrap_or(0));
            }
        }
        kana.finish();
        let mut words = FastMap::default();
        let mut letters = CharLm::default();
        let (mut ja_total, mut en_total) = (0.0, 0.0);
        for line in latin_tsv.lines() {
            let mut cols = line.split('\t');
            let (Some(word), Some(ja), Some(en)) = (cols.next(), cols.next(), cols.next()) else {
                continue;
            };
            let (ja, en): (u32, u32) = (ja.parse().unwrap_or(0), en.parse().unwrap_or(0));
            ja_total += ja as f64;
            en_total += en as f64;
            // The letter model counts word types: what unseen words look like.
            let padded: Vec<char> = std::iter::once('^')
                .chain(word.chars().map(letter_class))
                .chain(std::iter::once('$'))
                .collect();
            for n in 1..=3 {
                for gram in padded.windows(n) {
                    letters.add_chars(gram, 1);
                }
            }
            words.insert(word.to_string(), (ja, en));
        }
        letters.finish();
        let mut sorted: Vec<String> = words.keys().cloned().collect();
        sorted.sort();
        let mut running = Vec::with_capacity(sorted.len() + 1);
        let (mut ja, mut en) = (0u64, 0u64);
        for word in &sorted {
            running.push((ja, en));
            let (j, e) = words[word];
            ja += j as u64;
            en += e as u64;
        }
        running.push((ja, en));
        // Tuned parameters, if any.
        let mut params = MixedParams::default();
        for line in params_txt
            .unwrap_or("")
            .lines()
            .filter(|l| !l.trim().is_empty())
        {
            params.set(line)?;
        }
        Ok(Self {
            kana,
            letters,
            words,
            sorted,
            running,
            ja_total: ja_total.max(1.0),
            en_total: en_total.max(1.0),
            names: FastMap::default(),
            params,
            lexicon_costs: Mutex::default(),
        })
    }

    /// Add the names of a names dictionary (`<name as written>\t<weight>`
    /// lines, `\tcommon` after a common one, `#` comments; a name of one
    /// word of ASCII letters, digits and joiners). Of two names with one
    /// lowercase form the heavier wins. A
    /// name that reads as romaji all through (`sushi`, `toyota`) is left
    /// out: typed in lowercase it is Japanese.
    pub fn add_names(&mut self, tsv: &str) {
        for line in tsv.lines().filter(|l| !l.starts_with('#')) {
            let mut cols = line.split('\t');
            let Some(name) = cols.next().filter(|n| {
                n.chars().any(|c| c.is_ascii_alphabetic())
                    && n.chars().all(|c| c.is_ascii_alphanumeric() || is_joiner(c))
            }) else {
                continue;
            };
            let lower: Vec<char> = name.chars().map(|c| c.to_ascii_lowercase()).collect();
            if romaji_units(&lower, true)
                .iter()
                .all(|u| matches!(u.kind, UnitKind::Kana(_)))
            {
                continue;
            }
            let weight: f64 = cols.next().and_then(|w| w.parse().ok()).unwrap_or(1.0);
            let common = cols.next() == Some("common");
            let entry = self
                .names
                .entry(name.to_ascii_lowercase())
                .or_insert_with(|| (name.to_string(), 0.0, common));
            if weight > entry.1 {
                *entry = (name.to_string(), weight, common);
            }
        }
    }

    /// The spans of `typed` (a space separates spans and belongs to none).
    /// `before` is the text before the caret; `fin` whether typing is done
    /// (a final `n` is ん).
    ///
    /// The kana model has seen Latin words as `@` in mixed text: what comes
    /// before an English span costs −ln P(@ | the kana before it), and the
    /// kana after one are read on from `@` (`google` + を is common,
    /// `melty` + ぺ is not).
    pub fn segment(&self, typed: &str, before: Option<&str>, fin: bool) -> Vec<Span> {
        self.segment_with(typed, before, fin, None)
    }

    /// `segment` with the reading lexicon, which reads the words romaji
    /// reads wholly as kana (`kana_words`).
    pub fn segment_with(
        &self,
        typed: &str,
        before: Option<&str>,
        fin: bool,
        lexicon: Option<&Lexicon>,
    ) -> Vec<Span> {
        let (spans, cost) = self.segment_masked(typed, before, fin, None);
        let spans = self.letter_words(typed, before, fin, spans, cost);
        match lexicon {
            Some(lexicon) => self.kana_words(lexicon, typed, fin, spans),
            None => spans,
        }
    }

    /// A Latin word the model knows (as often as `kana_word_floor`) that
    /// romaji reads wholly as kana inside a Japanese span of `spans`
    /// (`issue` いっすえ, `made` まで) is English where its kana cost as
    /// Japanese more than `kana_word` over its cost among Latin words: the
    /// reading lexicon's cheapest words over the kana with
    /// `kana_word_window` kana each side, less those over the sides alone
    /// (まで is a particle, cheap; いっすえ is 一 + 据え, dear). The kana
    /// model reads kana by threes and finds いっすえ about as likely as
    /// Japanese goes; a word over it is what tells. Of words that overlap,
    /// the one by the most.
    fn kana_words(&self, lexicon: &Lexicon, typed: &str, fin: bool, spans: Vec<Span>) -> Vec<Span> {
        let p = &self.params;
        let chars: Vec<char> = typed.chars().collect();
        let mut out = Vec::with_capacity(spans.len());
        for span in spans {
            if span.lang != Lang::Ja || span.end - span.start < 3 {
                out.push(span);
                continue;
            }
            let done = fin || span.end < chars.len();
            let units = romaji_units(&chars[span.start..span.end], done);
            // the kana before each unit, and whether the unit is a kana
            let mut kana: Vec<char> = Vec::new();
            let mut at: Vec<usize> = Vec::with_capacity(units.len() + 1);
            for u in &units {
                at.push(kana.len());
                match u.kind {
                    UnitKind::Kana(k) => kana.extend(k.chars()),
                    UnitKind::Char(c) => kana.push(c),
                    UnitKind::Pending => {}
                }
            }
            at.push(kana.len());
            let window = p.kana_word_window as usize;
            let mut flips: Vec<(f64, usize, usize)> = Vec::new();
            for i in 0..units.len() {
                let a = span.start + units[i].start;
                for j in i + 1..=units.len() {
                    let u = &units[j - 1];
                    if !matches!(u.kind, UnitKind::Kana(_)) {
                        break;
                    }
                    let b = span.start + u.end;
                    if b - a > 15 {
                        break;
                    }
                    if b - a < 3 || !chars[a..b].iter().all(|c| c.is_ascii_alphabetic()) {
                        continue;
                    }
                    let word: String = chars[a..b].iter().map(|c| c.to_ascii_lowercase()).collect();
                    let Some(freq) = self.known(&word).filter(|&f| f >= p.kana_word_floor) else {
                        continue;
                    };
                    let (ka, kb) = (at[i], at[j]);
                    // its kana as the word alone spells them: not a doubled
                    // consonant's first letter after it (`bit` of `bitto`)
                    let alone = romaji_units(&chars[a..b], true);
                    let mut spelled = String::new();
                    for u in &alone {
                        match u.kind {
                            UnitKind::Kana(k) => spelled.push_str(k),
                            _ => spelled.push('#'),
                        }
                    }
                    // nor a word whose kana start inside a Japanese one
                    // (`npo` of でんぽう, んぽ: ん and small kana end the
                    // kana before them)
                    if !spelled.chars().eq(kana[ka..kb].iter().copied())
                        || "んゃゅょぁぃぅぇぉっゎ".contains(kana[ka])
                    {
                        continue;
                    }
                    // nor one beside a letter the Japanese leaves: a slip of
                    // the romaji is there, which the word is likelier part
                    // of (げたをぬいdせお: not SEO)
                    let leftover = |u: Option<&crate::composing_text::RomajiUnit>| {
                        u.is_some_and(|u| match u.kind {
                            UnitKind::Char(c) => c.is_ascii_alphabetic(),
                            UnitKind::Pending => done,
                            UnitKind::Kana(_) => false,
                        })
                    };
                    if leftover(i.checked_sub(1).and_then(|k| units.get(k)))
                        || leftover(units.get(j))
                    {
                        continue;
                    }
                    let from = if window == 0 {
                        0
                    } else {
                        ka.saturating_sub(window)
                    };
                    let to = if window == 0 {
                        kana.len()
                    } else {
                        (kb + window).min(kana.len())
                    };
                    let all: String = kana[from..to].iter().collect();
                    let left: String = kana[from..ka].iter().collect();
                    let right: String = kana[kb..to].iter().collect();
                    let as_ja = self.lexicon_cost(lexicon, &all)
                        - self.lexicon_cost(lexicon, &left)
                        - self.lexicon_cost(lexicon, &right);
                    let by = as_ja + freq.ln() - p.kana_word;
                    if by > 0.0 {
                        flips.push((by, a, b));
                    }
                }
            }
            // the words by the most first, none over another
            flips.sort_by(|x, y| y.0.total_cmp(&x.0));
            let mut taken: Vec<(usize, usize)> = Vec::new();
            for &(_, a, b) in &flips {
                if taken.iter().all(|&(x, y)| b <= x || y <= a) {
                    taken.push((a, b));
                }
            }
            taken.sort_unstable();
            let mut pos = span.start;
            for (a, b) in taken {
                if pos < a {
                    out.push(Span {
                        lang: Lang::Ja,
                        start: pos,
                        end: a,
                    });
                }
                out.push(Span {
                    lang: Lang::En,
                    start: a,
                    end: b,
                });
                pos = b;
            }
            if pos < span.end {
                out.push(Span {
                    lang: Lang::Ja,
                    start: pos,
                    end: span.end,
                });
            }
        }
        out
    }

    /// The reading lexicon's cheapest conversion of `kana` (0 for none).
    fn lexicon_cost(&self, lexicon: &Lexicon, kana: &str) -> f64 {
        if kana.is_empty() {
            return 0.0;
        }
        let key = (lexicon as *const Lexicon as usize, kana.to_string());
        if let Some(&c) = self
            .lexicon_costs
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get(&key)
        {
            return c;
        }
        let cost = lexicon.convert(kana, 1).first().map_or(
            f64::from(crate::lexicon::KANA_COST) * kana.chars().count() as f64,
            |c| f64::from(c.1),
        );
        let mut costs = self.lexicon_costs.lock().unwrap_or_else(|e| e.into_inner());
        if costs.len() >= LEXICON_COSTS {
            costs.clear();
        }
        costs.insert(key, cost);
        cost
    }

    /// A letter the Japanese of `spans` (whose cost is `cost`) leaves (ぶg
    /// for `bug`, kぴ for `kpi`) read with the letters around it as a Latin
    /// word the model knows, English, where that costs at most
    /// `letter_word` over the best, the cheapest first, up to three: the
    /// letter model finds such a word about as likely as a slip of romaji,
    /// which leaves a letter too, but a slip rarely makes a word.
    fn letter_words(
        &self,
        typed: &str,
        before: Option<&str>,
        fin: bool,
        mut spans: Vec<Span>,
        mut cost: f64,
    ) -> Vec<Span> {
        let limit = self.params.letter_word;
        if limit <= 0.0 {
            return spans;
        }
        let chars: Vec<char> = typed.chars().collect();
        for _ in 0..3 {
            let mut best: Option<(f64, Vec<Span>, f64)> = None;
            for span in spans.iter().filter(|s| s.lang == Lang::Ja) {
                // the letters the Japanese shown leaves: only a word over one
                // is read again (most spans have none, and cost nothing)
                let done = fin || span.end < chars.len();
                let left: Vec<usize> = romaji_units(&chars[span.start..span.end], done)
                    .iter()
                    .filter(|u| match u.kind {
                        UnitKind::Char(c) => c.is_ascii_alphabetic(),
                        UnitKind::Pending => done,
                        UnitKind::Kana(_) => false,
                    })
                    .map(|u| span.start + u.start)
                    .collect();
                for &l in &left {
                    for a in l.saturating_sub(14).max(span.start)..=l {
                        for b in (a + 3).max(l + 1)..=(a + 15).min(span.end) {
                            if !chars[a..b].iter().all(|c| c.is_ascii_alphabetic()) {
                                break;
                            }
                            let word: String =
                                chars[a..b].iter().map(|c| c.to_ascii_lowercase()).collect();
                            if !self
                                .known(&word)
                                .is_some_and(|f| f >= self.params.letter_word_floor)
                            {
                                continue;
                            }
                            let mut mask = vec![None; chars.len()];
                            mask[span.start..span.end].fill(Some(Lang::Ja));
                            mask[a..b].fill(Some(Lang::En));
                            let (flipped, fcost) =
                                self.segment_masked(typed, before, fin, Some(&mask));
                            let gap = fcost - cost;
                            if gap <= limit && best.as_ref().is_none_or(|(g, _, _)| gap < *g) {
                                best = Some((gap, flipped, fcost));
                            }
                        }
                    }
                }
            }
            let Some((_, flipped, fcost)) = best else {
                break;
            };
            spans = flipped;
            cost = fcost;
        }
        spans
    }

    /// `segment` with the language of some chars fixed (`mask`, a language
    /// or `None` for each char), and the cost of the best segmentation so,
    /// before its spans are read again (particles split, words protected):
    /// what the bench's `gap` compares a case's expected reading with.
    #[doc(hidden)]
    pub fn segment_masked(
        &self,
        typed: &str,
        before: Option<&str>,
        fin: bool,
        mask: Option<&[Option<Lang>]>,
    ) -> (Vec<Span>, f64) {
        // The state after a span: Japanese, English, or Japanese that an
        // English span follows right away (its cost has the odds of `@`).
        const JA: usize = 0;
        const EN: usize = 1;
        const JA_EN: usize = 2;
        const INF: f64 = f64::INFINITY;
        let chars: Vec<char> = typed.chars().collect();
        let n = chars.len();
        let p = &self.params;
        // whether chars[a..b] may be read as `lang`
        let allowed = |lang: Lang, a: usize, b: usize| {
            mask.is_none_or(|m| m[a..b].iter().all(|l| l.is_none_or(|l| l == lang)))
        };
        let context = before.and_then(context_lang).map(|l| match l {
            Lang::Ja => JA,
            Lang::En => EN,
        });
        let context_spaced = before.is_some_and(|b| b.ends_with(char::is_whitespace));
        // The context's last two chars as the kana model reads them: the
        // odds of `@` after them, and the kana that go on from them.
        let context_kana: [char; 2] = {
            let tail: Vec<char> = before
                .unwrap_or("")
                .chars()
                .rev()
                .take(2)
                .map(context_class)
                .collect();
            [*tail.get(1).unwrap_or(&'^'), *tail.first().unwrap_or(&'^')]
        };
        // best[pos][state]: cost of chars[..pos] with a span ending at pos;
        // back[pos][state]: (span start, state before it, `None` at the
        // start of the input).
        let mut best = vec![[INF; 3]; n + 1];
        let mut back: Vec<[(usize, Option<usize>); 3]> = vec![[(0, None); 3]; n + 1];
        let upper_before: Vec<usize> = std::iter::once(0)
            .chain(chars.iter().scan(0, |acc, c| {
                *acc += c.is_ascii_uppercase() as usize;
                Some(*acc)
            }))
            .collect();
        for start in 0..n {
            if chars[start] == ' ' {
                continue;
            }
            // Spaces before `start` are skipped over.
            let mut from = start;
            while from > 0 && chars[from - 1] == ' ' {
                from -= 1;
            }
            let first = from == 0;
            let spaced = if first { context_spaced } else { from < start };
            // The states a span starting here may follow, with their cost;
            // the context's language stands before the input.
            let entry: Vec<(Option<usize>, f64)> = if first {
                vec![(context, 0.0)]
            } else {
                [JA, EN, JA_EN]
                    .into_iter()
                    .filter(|&s| best[from][s].is_finite())
                    .map(|s| (Some(s), best[from][s]))
                    .collect()
            };
            let seg_end = chars[start..]
                .iter()
                .position(|&c| c == ' ')
                .map_or(n, |k| start + k);
            // Romaji before a space is done (`think so`: no kana is coming
            // for the k).
            let seg_fin = fin || seg_end < n;
            // Sentence punctuation right after an English word, up to a
            // space or the end (`a pen.`, `hello,`): English goes on past
            // it. As Japanese it is the kana model's 。, likely after kana,
            // which turned `this is a pen.` into Japanese: here it costs what
            // the kana model gives it after a Latin word (Google。), with no
            // switch. After a sentence's end Japanese may start as freely.
            // Whether it shows as `.` or 。 is the sentence's
            // (`english_punctuation`).
            if !spaced
                && !first
                && is_sentence_punct(chars[start])
                && best[start][EN].is_finite()
                && allowed(Lang::En, start, start + 1)
            {
                let end = (start..n)
                    .find(|&k| !is_sentence_punct(chars[k]))
                    .unwrap_or(n);
                if end == n || chars[end] == ' ' {
                    let mut ctx = ['@', '@'];
                    let mut cost = best[start][EN];
                    for &c in &chars[start..end] {
                        // the corpus readings end and split sentences with
                        // 。、 only
                        let class = match c {
                            '.' | '!' | '?' => '。',
                            _ => '、',
                        };
                        cost += self.kana.cost(ctx[0], ctx[1], class);
                        ctx = [ctx[1], class];
                    }
                    let ends = chars[start..end]
                        .iter()
                        .any(|c| matches!(c, '.' | '!' | '?'));
                    for state in [EN, JA].into_iter().filter(|&s| s == EN || ends) {
                        if cost < best[end][state] {
                            best[end][state] = cost;
                            back[end][state] = (start, Some(EN));
                        }
                    }
                }
            }
            let mut update = |end: usize, state: usize, cost: f64, prev: Option<usize>| {
                if cost < best[end][state] {
                    best[end][state] = cost;
                    back[end][state] = (start, prev);
                }
            };
            // Japanese spans, read from a reading's start or, right after
            // English, from a Latin word.
            let mut fresh = None;
            let mut after_latin = None;
            let mut after_context = None;
            for &(prev, c0) in &entry {
                // 0: from a reading's start, 1: from a Latin word, 2: on
                // from the context's kana
                let (enter, read) = match prev {
                    None => (0.0, 0),
                    Some(JA) if first && !spaced => (0.0, 2),
                    // after a space Japanese starts over
                    Some(JA) if first || spaced => (0.0, 0),
                    Some(EN) => (p.switch_out, if spaced { 0 } else { 1 }),
                    // Japanese right after Japanese is one span; JA_EN
                    // is followed by English
                    _ => continue,
                };
                let part = &chars[start..seg_end];
                let costs: &Vec<(usize, f64, f64)> = match read {
                    1 => {
                        after_latin.get_or_insert_with(|| self.ja_costs(part, seg_fin, ['@', '@']))
                    }
                    2 => after_context
                        .get_or_insert_with(|| self.ja_costs(part, seg_fin, context_kana)),
                    _ => fresh.get_or_insert_with(|| self.ja_costs(part, seg_fin, ['^', '^'])),
                };
                for &(len, cost, to_latin) in costs {
                    let end = start + len;
                    if !allowed(Lang::Ja, start, end) {
                        continue;
                    }
                    let cost = c0
                        + enter
                        + cost
                        + p.upper * (upper_before[end] - upper_before[start]) as f64;
                    update(end, JA, cost, prev);
                    if end < seg_end {
                        update(end, JA_EN, cost + to_latin, prev);
                    }
                }
            }
            // English spans.
            let mut en_costs: Option<Vec<(usize, f64)>> = None;
            for &(prev, c0) in &entry {
                let enter = match prev {
                    None => p.start_en,
                    Some(EN) if spaced => 0.0,
                    Some(EN) => p.en_en,
                    // right after the context's Japanese: the odds of `@`
                    // after its kana
                    Some(JA) if first && !spaced => {
                        p.switch_in + self.kana.cost(context_kana[0], context_kana[1], '@')
                    }
                    Some(JA) if spaced => p.switch_in,
                    Some(JA_EN) if !spaced => p.switch_in,
                    _ => continue,
                };
                let costs = en_costs.get_or_insert_with(|| {
                    let to = seg_end.min(start + MAX_EN);
                    self.en_costs(&chars[start..to], !fin && to == n)
                });
                for &(len, cost) in costs.iter() {
                    if allowed(Lang::En, start, start + len) {
                        update(start + len, EN, c0 + enter + cost, prev);
                    }
                }
            }
        }
        // Trailing spaces belong to no span.
        let mut end = n;
        while end > 0 && chars[end - 1] == ' ' {
            end -= 1;
        }
        let mut state = if best[end][EN] < best[end][JA] {
            EN
        } else {
            JA
        };
        let total = best[end][state];
        let mut spans = Vec::new();
        while end > 0 {
            if !best[end][state].is_finite() {
                // Unreachable (only spaces): give up on segmenting.
                let all = Span {
                    lang: Lang::Ja,
                    start: 0,
                    end: n,
                };
                return (vec![all], total);
            }
            let (start, prev) = back[end][state];
            let lang = if state == EN { Lang::En } else { Lang::Ja };
            spans.push(Span { lang, start, end });
            let mut from = start;
            while from > 0 && chars[from - 1] == ' ' {
                from -= 1;
            }
            end = from;
            match prev {
                Some(s) if end > 0 => state = s,
                _ => break,
            }
        }
        spans.reverse();
        lone_i(&chars, &mut spans);
        let spans = self.split_particles(&chars, spans, fin);
        let mut spans = protect(&chars, spans);
        // the punctuation English went on past is Japanese's (、。) until
        // `cased` reads its sentence
        for s in &mut spans {
            if chars[s.start..s.end].iter().all(|&c| is_sentence_punct(c)) {
                s.lang = Lang::Ja;
            }
        }
        (spans, total)
    }

    /// An English span the model does not know as a word that ends in a
    /// particle read as romaji (`Ollamade`, `Qwenno`) is English up to it
    /// and Japanese from it (Ollama + で): the letter model alone finds
    /// such a word as likely as the split. Of several splits, the first
    /// (Ollama + のです, not Ollamano + です); a word it knows (`code`,
    /// `piano`) stays whole.
    fn split_particles(&self, chars: &[char], spans: Vec<Span>, fin: bool) -> Vec<Span> {
        let mut out: Vec<Span> = Vec::with_capacity(spans.len() + 1);
        let mut todo: std::collections::VecDeque<Span> = spans.into();
        while let Some(span) = todo.pop_front() {
            if span.lang != Lang::En {
                out.push(span);
                continue;
            }
            // Japanese right after it reads on from the particle.
            let next_ja = todo
                .front()
                .is_some_and(|n| n.lang == Lang::Ja && n.start == span.end);
            let rest_end = if next_ja {
                todo.front().map_or(span.end, |n| n.end)
            } else {
                span.end
            };
            match self.particle_split(chars, span, rest_end, fin) {
                Some((at, None)) => {
                    if next_ja {
                        todo.pop_front();
                    }
                    out.push(Span { end: at, ..span });
                    out.push(Span {
                        lang: Lang::Ja,
                        start: at,
                        end: rest_end,
                    });
                }
                // a capital after the particle starts English again
                // (Tealium + の + CDP), which may split in turn
                Some((at, Some(capital))) => {
                    out.push(Span { end: at, ..span });
                    out.push(Span {
                        lang: Lang::Ja,
                        start: at,
                        end: capital,
                    });
                    todo.push_front(Span {
                        start: capital,
                        ..span
                    });
                }
                None => match self.sentence_split(
                    chars,
                    span,
                    rest_end,
                    fin,
                    before_kana(&out, chars, span),
                ) {
                    Some(at) => {
                        if next_ja {
                            todo.pop_front();
                        }
                        out.push(Span { end: at, ..span });
                        out.push(Span {
                            lang: Lang::Ja,
                            start: at,
                            end: rest_end,
                        });
                    }
                    None => out.push(span),
                },
            }
        }
        out
    }

    /// Where a word the model knows, typed in lowercase, reads better as a
    /// commoner word and romaji (`dlsite`: dl + して, not the name DLsite):
    /// the word's cost against the shorter word's and the kana after a
    /// Latin word, with no switch back to Japanese counted (the sentence
    /// is the commoner text). Typed with Shift it stays (English meant).
    /// `before`: the kana right before the word (`^` none), as the kana
    /// model reads on from it.
    fn sentence_split(
        &self,
        chars: &[char],
        span: Span,
        rest_end: usize,
        fin: bool,
        before: char,
    ) -> Option<usize> {
        let p = &self.params;
        let word: String = chars[span.start..span.end].iter().collect();
        if !word.chars().all(|c| c.is_ascii_lowercase()) {
            return None;
        }
        let rest_fin = fin && rest_end == chars.len();
        // the kana of `chars[from..rest_end]`, but for romaji still being
        // typed; `None` when it does not read as kana
        let kana_from = |from: usize| {
            let units = romaji_units(&chars[from..rest_end], rest_fin);
            let mut kana = String::new();
            for (i, u) in units.iter().enumerate() {
                match u.kind {
                    UnitKind::Kana(k) => kana.push_str(k),
                    UnitKind::Pending
                        if !rest_fin && i + 1 == units.len() && rest_end == chars.len() => {}
                    _ => return None,
                }
            }
            Some(kana)
        };
        // the kana model's cost of `kana` after the Latin word
        let after_latin = |kana: &str| {
            let mut ctx = [before, '@'];
            let mut cost = 0.0;
            for c in kana.chars() {
                cost += self.kana.cost(ctx[0], ctx[1], c);
                ctx = [ctx[1], c];
            }
            cost
        };
        // what follows counts both ways (DLsite + で, not DL + してで)
        let whole = -(p.known * self.known(&word)?).ln() + after_latin(&kana_from(span.end)?);
        (span.start + 2..span.end.saturating_sub(1))
            .filter_map(|at| {
                let head: String = chars[span.start..at].iter().collect();
                let head = -(p.known * self.known(&head)?).ln();
                let kana = kana_from(at)?;
                // a form of する (DLして, OKした): a Latin word's commonest
                // use in Japanese
                if !SURU.iter().any(|f| kana.starts_with(f)) {
                    return None;
                }
                Some((at, head + after_latin(&kana) - p.sentence))
            })
            .filter(|&(_, cost)| cost < whole)
            .min_by(|a, b| a.1.total_cmp(&b.1))
            .map(|(at, _)| at)
    }

    /// Where `span` (English) splits before a particle, Japanese running on
    /// to `rest_end` or to a capital in the span (where English starts
    /// again, also returned); see `split_particles`.
    fn particle_split(
        &self,
        chars: &[char],
        span: Span,
        rest_end: usize,
        fin: bool,
    ) -> Option<(usize, Option<usize>)> {
        let word: String = chars[span.start..span.end].iter().collect();
        // the last run of letters: `Wi-Fino` splits in `Fino`
        let run_start = chars[span.start..span.end]
            .iter()
            .rposition(|c| !c.is_ascii_alphabetic())
            .map_or(span.start, |k| span.start + k + 1);
        let run: String = chars[run_start..span.end].iter().collect();
        if self.knows(&word) || self.knows(&run) {
            return None;
        }
        // at least two letters stay English
        (run_start + 2..span.end.saturating_sub(1)).find_map(|at| {
            // a particle typed in lowercase (Shift marks English)
            let typed: String = chars[at..span.end].iter().collect();
            if !PARTICLES.iter().any(|p| typed.starts_with(p)) {
                return None;
            }
            let capital = (at + 1..span.end).find(|&k| chars[k].is_ascii_uppercase());
            let end = capital.unwrap_or(rest_end);
            let rest_fin = capital.is_some() || fin && rest_end == chars.len();
            // the rest reads as kana, but for romaji still being typed
            let units = romaji_units(&chars[at..end], rest_fin);
            let reads = units.iter().enumerate().all(|(i, u)| match u.kind {
                UnitKind::Kana(_) => true,
                UnitKind::Pending => !rest_fin && i + 1 == units.len() && end == chars.len(),
                UnitKind::Char(_) => false,
            });
            // what stays is English (no joiner at its end) and, before a
            // capital, the word up to it is no word as a whole (Photo +
            // Shop is not Pho + と + Shop) and what follows is English
            let stays = self
                .en_costs(&chars[span.start..at], false)
                .last()
                .is_some_and(|(len, _)| span.start + len == at);
            let ok = match capital {
                None => true,
                Some(capital) => {
                    let from = (run_start..at)
                        .rev()
                        .find(|&k| chars[k].is_ascii_uppercase())
                        .unwrap_or(run_start);
                    let whole: String = chars[from..capital].iter().collect();
                    !self.knows(&whole)
                        && self
                            .en_costs(&chars[capital..span.end], !fin && span.end == chars.len())
                            .last()
                            .is_some_and(|(len, _)| capital + len == span.end)
                }
            };
            (reads && stays && ok).then_some((at, capital))
        })
    }

    /// For every `end` (1..=len): the cost of `chars[..end]` as Japanese,
    /// and of a Latin word right after it. The kana are read on from `ctx`
    /// (`^` a reading's start, `@` a Latin word).
    fn ja_costs(&self, chars: &[char], fin: bool, ctx: [char; 2]) -> Vec<(usize, f64, f64)> {
        let p = &self.params;
        let units = romaji_units(chars, fin);
        let mut out = Vec::with_capacity(chars.len());
        let mut ctx = ctx;
        let mut cost = 0.0;
        let mut pos = 0;
        let mut left_over = 0usize;
        for unit in units {
            let to_latin = self.kana.cost(ctx[0], ctx[1], '@');
            // An end inside this unit leaves its letters unread.
            for end in pos + 1..unit.end {
                out.push((end, cost + p.leftover * (end - unit.start) as f64, to_latin));
            }
            let mut add = |c: char| {
                let step = self.kana.cost(ctx[0], ctx[1], c);
                ctx = [ctx[1], c];
                step
            };
            cost += match unit.kind {
                UnitKind::Kana(kana) => kana.chars().map(&mut add).sum(),
                UnitKind::Char(c) if c.is_ascii_alphabetic() => {
                    left_over += 1;
                    p.leftover + p.leftover_more * (left_over - 1) as f64
                }
                // Digits are the same in either language; the kana after
                // them starts over (3じ).
                UnitKind::Char(c) if c.is_ascii_digit() || is_decimal_point(chars, unit.start) => {
                    ctx = ['^', '^'];
                    p.digit
                }
                // A sentence ends: the model read one sentence per reading,
                // so the next starts over (。 then こ is no rare pair).
                UnitKind::Char(c @ ('.' | '!' | '?')) => {
                    let end = add(kana_class(c)) + add('$');
                    ctx = ['^', '^'];
                    end
                }
                UnitKind::Char(c) => add(kana_class(c)),
                UnitKind::Pending if fin => p.leftover * (unit.end - unit.start) as f64,
                // Romaji still being typed: the odds of a kana it may become.
                UnitKind::Pending => {
                    let typed: String = chars[unit.start..unit.end].iter().collect();
                    let odds: f64 = pending_kana(&typed)
                        .into_iter()
                        .map(|k| (-self.kana.cost(ctx[0], ctx[1], k)).exp())
                        .sum();
                    -odds.max(f64::MIN_POSITIVE).ln()
                }
            };
            out.push((unit.end, cost, self.kana.cost(ctx[0], ctx[1], '@')));
            pos = unit.end;
        }
        out
    }

    /// The cost as English of every prefix `chars[..len]` that can be
    /// English, as `(len, cost)`. Each prefix is word pieces joined by
    /// symbols (`example.com`, `e-mail`), starting and ending with a letter
    /// or digit, with a letter in it. `growing`: the whole of `chars` may
    /// still be being typed (its last piece costs as a word's start).
    fn en_costs(&self, chars: &[char], growing: bool) -> Vec<(usize, f64)> {
        let p = &self.params;
        let mut out = Vec::new();
        if !chars.first().is_some_and(|c| c.is_ascii_alphanumeric()) {
            return out;
        }
        // completed pieces, and joiners so far
        let mut done = 0.0;
        let mut joiners = 0usize;
        let mut has_letter = false;
        let mut whole = String::new();
        let mut piece = Piece::default();
        for (k, &c) in chars.iter().enumerate() {
            if is_joiner(c) {
                // Joiners stand alone, but for a URL's `://` and `++`.
                if k > 0 && is_joiner(chars[k - 1]) {
                    let url = matches!(&chars[k - 1..], [':', '/', '/', ..])
                        || chars[k - 1] == '/' && c == '/'
                        || chars[k - 1] == '+' && c == '+';
                    if !url {
                        break;
                    }
                }
                if !piece.is_empty() {
                    done += self.piece_cost(&piece, false);
                    piece = Piece::default();
                }
                joiners += 1;
                whole.push(c);
                // `+` may end a word (NAD+, C++)
                if c == '+' && has_letter {
                    let mut cost = done + p.symbol * joiners as f64;
                    if let Some(known) = self.known(&whole) {
                        cost = cost.min(-(p.known * known).ln());
                    }
                    out.push((k + 1, cost));
                }
                continue;
            }
            // and `#` after a letter (C#, F#)
            if c == '#' && !piece.is_empty() && has_letter {
                let cost = done + self.piece_cost(&piece, false) + p.symbol * (joiners + 1) as f64;
                out.push((k + 1, cost));
                break;
            }
            if !c.is_ascii_alphanumeric() {
                break;
            }
            has_letter |= c.is_ascii_alphabetic();
            piece.push(c, self);
            whole.push(c.to_ascii_lowercase());
            if !has_letter {
                continue;
            }
            let len = k + 1;
            let mut cost = done
                + self.piece_cost(&piece, growing && len == chars.len())
                + p.symbol * joiners as f64;
            // A word with its joiners (e-mail, node.js) may be a word of its own.
            if joiners > 0 {
                if let Some(known) = self.known(&whole) {
                    cost = cost.min(-(p.known * known).ln());
                }
            }
            out.push((len, cost));
        }
        out
    }

    /// Cost of a word piece, or with `prefix` of any word it starts; with
    /// its case breaks.
    fn piece_cost(&self, piece: &Piece, prefix: bool) -> f64 {
        let p = &self.params;
        // a name as it is written has no case break (DLsite, as the casing
        // writes dlsite and the composition is read again on Space)
        let written = self
            .names
            .get(&piece.lower)
            .is_some_and(|(name, _, _)| *name == piece.typed);
        let breaks = if written {
            0.0
        } else {
            p.case_break * piece.breaks as f64
        };
        if piece.digits {
            return breaks + p.digit * piece.lower.len() as f64;
        }
        let (known, letters) = if prefix {
            (self.prefix_mass(&piece.lower), piece.letters)
        } else {
            let end = self.letters.cost(piece.ctx[0], piece.ctx[1], '$');
            (self.known(&piece.lower).unwrap_or(0.0), piece.letters + end)
        };
        let prob = p.known * known + (1.0 - p.known) * (-letters).exp();
        let oov = if known == 0.0 && !piece.typed.chars().any(|c| c.is_ascii_uppercase()) {
            p.oov
        } else {
            0.0
        };
        breaks + oov - prob.max(f64::MIN_POSITIVE).ln()
    }

    /// The frequency of the words starting with `prefix` (lowercase).
    fn prefix_mass(&self, prefix: &str) -> f64 {
        let p = &self.params;
        let from = self.sorted.partition_point(|w| w.as_str() < prefix);
        let to = from + self.sorted[from..].partition_point(|w| w.starts_with(prefix));
        let (ja, en) = (
            self.running[to].0 - self.running[from].0,
            self.running[to].1 - self.running[from].1,
        );
        p.ja_text * ja as f64 / self.ja_total + (1.0 - p.ja_text) * en as f64 / self.en_total
    }

    /// Whether `word` (any case) is a Latin word of the model's text or a
    /// name of its names dictionary.
    pub fn knows(&self, word: &str) -> bool {
        let lower = word.to_ascii_lowercase();
        self.words.contains_key(&lower) || self.name_weight(&lower).is_some()
    }

    /// `known` of `word` in any case (for the bench).
    #[doc(hidden)]
    pub fn frequency(&self, word: &str) -> Option<f64> {
        self.known(&word.to_ascii_lowercase())
    }

    /// The frequency of a lowercase word among Latin words, if seen; else
    /// a name's (`name_weight`).
    fn known(&self, word: &str) -> Option<f64> {
        let p = &self.params;
        match self.words.get(word) {
            Some(&(ja, en)) => Some(
                p.ja_text * ja as f64 / self.ja_total
                    + (1.0 - p.ja_text) * en as f64 / self.en_total,
            ),
            None => self.name_weight(word).map(|weight| p.name * weight),
        }
    }

    /// The weight of a name (lowercase) that tells English from romaji: of
    /// four chars or more and not an acronym (UIC, RAAD, Py would make
    /// English of the romaji around them: かんとUIC); the others only give
    /// their case.
    fn name_weight(&self, word: &str) -> Option<f64> {
        self.name_weight_of(word)
    }

    /// −ln P of `kana` as a reading by the kana model (start and end
    /// included): how Japanese-like it reads (typo correction's judge).
    pub fn kana_cost(&self, kana: &str) -> f64 {
        let chars: Vec<char> = std::iter::once('^')
            .chain(std::iter::once('^'))
            .chain(kana.chars().map(kana_class))
            .chain(std::iter::once('$'))
            .collect();
        chars
            .windows(3)
            .map(|w| self.kana.cost(w[0], w[1], w[2]))
            .sum()
    }

    fn name_weight_of(&self, word: &str) -> Option<f64> {
        self.names
            .get(word)
            .filter(|(name, _, _)| word.len() >= 4 && name.chars().any(|c| c.is_ascii_lowercase()))
            .map(|&(_, weight, _)| weight)
    }

    /// How `word` (lowercase) is written when typed so, as `names` says: as
    /// its name, unless the name is lowercase (the dictionary gives an
    /// ordinary word, apple, its lowercase form when the word is better
    /// known than the name) or a common one (an acronym, a word too: OK,
    /// Word), which no dictionary decides (`acronym` does the acronyms).
    fn name_case(&self, word: &str, names: NameCase) -> Option<&str> {
        let (name, _, common) = self.names.get(word)?;
        (names != NameCase::Off && !common && name != word).then_some(name.as_str())
    }

    /// `typed` segmented (`segment`) and cased as shown, as `casing` says:
    /// the English spans' letters with their case, and an English
    /// sentence's punctuation as English (see the module docs). Char for
    /// char: a position in it is one in `typed`.
    pub fn cased(
        &self,
        typed: &str,
        before: Option<&str>,
        fin: bool,
        casing: Casing,
    ) -> (Vec<char>, Vec<Span>) {
        self.cased_with(typed, before, fin, casing, None)
    }

    /// `cased` segmenting with the reading lexicon (`segment_with`).
    pub fn cased_with(
        &self,
        typed: &str,
        before: Option<&str>,
        fin: bool,
        casing: Casing,
        lexicon: Option<&Lexicon>,
    ) -> (Vec<char>, Vec<Span>) {
        let mut chars: Vec<char> = typed.chars().collect();
        let mut spans = self.segment_with(typed, before, fin, lexicon);
        let sentence = if casing.sentences {
            english_sentence(&chars, &spans, before)
        } else {
            0
        };
        for (k, span) in spans.iter().enumerate().filter(|(_, s)| s.lang == Lang::En) {
            // a unit after a number stays (100km, 173cm, m2), and so does
            // what follows a dot (sitemap1.xml)
            if span.start > 0 && matches!(chars[span.start - 1], '0'..='9' | '.') {
                continue;
            }
            let word = &mut chars[span.start..span.end];
            if word.iter().any(|c| c.is_ascii_uppercase()) {
                continue;
            }
            let lower: String = word.iter().collect();
            // a word with joiners as a whole only: example.com, not
            // Example.com
            if let Some(name) = self.name_case(&lower, casing.names) {
                // in an English sentence only a name with a capital inside
                // (iPhone, GitHub): `the word`, `it works` are words
                let inner = name.chars().skip(1).any(|c| c.is_ascii_uppercase())
                    && name.chars().any(|c| c.is_ascii_lowercase());
                if k >= sentence || inner {
                    for (c, n) in word.iter_mut().zip(name.chars()) {
                        *c = n;
                    }
                }
            } else if casing.names == NameCase::Acronyms
                // a name the dictionary writes in lowercase is one (npm, tmux)
                && self.names.get(&lower).is_none_or(|(name, _, _)| name != &lower)
                && acronym(word, k < sentence)
            {
                word.iter_mut().for_each(|c| c.make_ascii_uppercase());
            }
        }
        english_case(&mut chars, &mut spans[..sentence]);
        if casing.sentences {
            english_punctuation(&chars, &mut spans, before.unwrap_or(""));
        }
        (chars, spans)
    }
}

/// What `MixedModel::cased` cases (`[frontend] name_case`, `sentence_case`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Casing {
    pub names: NameCase,
    pub sentences: bool,
}

impl Default for Casing {
    fn default() -> Self {
        Self {
            names: NameCase::Acronyms,
            sentences: true,
        }
    }
}

/// Acronyms that read as words but Japanese text writes in capitals (ok,
/// api): no spelling tells them from a word.
const READ_ACRONYMS: &[&str] = &[
    "ok", "api", "ai", "id", "it", "url", "os", "ui", "ux", "iot", "seo", "gif", "diy", "ip",
    "ram", "lan",
];

/// Words of no vowel, or that no rule below reads, that are no acronyms:
/// interjections, placeholders (xx), and units, written in lowercase (or
/// as kW, which the typed letters do not tell).
const NOT_ACRONYMS: &[&str] = &[
    "hmm", "shh", "nth", "brr", "psst", "pst", "tsk", "zzz", "grr", "hm", "mm", "sh", "ok", "xx",
    "xxx", "yy", "km", "cm", "mm", "kg", "mg", "ml", "dl", "kl", "kw", "kwh", "mw", "hz", "khz",
    "mhz", "ghz", "kcal", "kb", "kbps", "mbps", "gbps", "dpi", "fps", "rpm", "ppm", "lx", "db",
];

/// Consonants an English word may start with (bl, str: `pronounceable`).
const ONSETS: &[&str] = &[
    "bl", "br", "ch", "cl", "cr", "dr", "dw", "fl", "fr", "gl", "gn", "gr", "kn", "kr", "ph", "pl",
    "pr", "ps", "sc", "sh", "sk", "sl", "sm", "sn", "sp", "st", "sw", "th", "tr", "tw", "wh", "wr",
    "chr", "sch", "scr", "shr", "sph", "spl", "spr", "squ", "str", "thr",
];

/// Consonants an English word may end with (nd, rst), a plural `s` aside.
const CODAS: &[&str] = &[
    "bb", "ch", "ck", "ct", "dd", "ff", "ft", "gh", "gg", "ld", "lf", "lk", "ll", "lm", "ln", "lp",
    "lt", "lv", "mb", "mn", "mp", "nc", "nd", "ng", "nk", "nn", "nt", "nx", "ph", "pt", "rb", "rc",
    "rd", "rf", "rg", "rk", "rl", "rm", "rn", "rp", "rr", "rt", "rv", "sh", "sk", "sm", "sp", "ss",
    "st", "th", "tt", "wd", "wl", "wn", "xt", "zz", "lch", "nch", "rch", "tch", "nth", "rth",
    "mpt", "rst", "dth", "ght", "nct", "lth",
];

/// Whether a run of lowercase letters can be read as an English word: a
/// vowel (y but at the start), the consonants before the first and after
/// the last a word may start and end with, and no q but before u.
fn pronounceable(w: &[char]) -> bool {
    let vowel = |i: usize| matches!(w[i], 'a' | 'e' | 'i' | 'o' | 'u') || w[i] == 'y' && i > 0;
    let (Some(first), Some(last)) = (
        (0..w.len()).find(|&i| vowel(i)),
        (0..w.len()).rev().find(|&i| vowel(i)),
    ) else {
        return false;
    };
    if w.iter()
        .enumerate()
        .any(|(i, &c)| c == 'q' && w.get(i + 1) != Some(&'u'))
    {
        return false;
    }
    let onset: String = w[..first].iter().collect();
    let mut coda: String = w[last + 1..].iter().collect();
    if coda.len() >= 2 && coda.ends_with('s') {
        coda.pop();
    }
    (onset.len() <= 1 || ONSETS.contains(&onset.as_str()))
        && (coda.len() <= 1 || CODAS.contains(&coda.as_str()))
}

/// Whether `word` (an English span typed in lowercase) is an acronym: two
/// to four letters (five with no vowel: https), a number after them or not
/// (mp3, gpt-4), that cannot be read as an English word (pc, sns, kpi,
/// usb, faq) or are a written one (`READ_ACRONYMS`: ok, api; not in an
/// English sentence, where it, id are words). A longer word that cannot be
/// read is a name more often (ctags, zylqa).
pub fn acronym(word: &[char], in_sentence: bool) -> bool {
    let letters = word.iter().take_while(|c| c.is_ascii_lowercase()).count();
    let rest = &word[letters..];
    let number = rest.strip_prefix(&['-']).unwrap_or(rest);
    let vowelless = || !word[..letters].iter().any(|c| "aeiouy".contains(*c));
    if !(2..=5).contains(&letters)
        || letters == 5 && !vowelless()
        || !number.iter().all(|c| c.is_ascii_digit())
    {
        return false;
    }
    let w: String = word[..letters].iter().collect();
    if NOT_ACRONYMS.contains(&w.as_str()) {
        return !in_sentence && READ_ACRONYMS.contains(&w.as_str());
    }
    !pronounceable(&word[..letters]) || !in_sentence && READ_ACRONYMS.contains(&w.as_str())
}

impl Casing {
    pub fn of(frontend: &crate::config::FrontendCfg) -> Self {
        Self {
            names: frontend.name_case,
            sentences: frontend.sentence_case,
        }
    }
}

/// Whether `candidate` has the Latin words of `reading` as they are, in
/// order, with the English punctuation right after them (`pen.`, `hello,`:
/// the reading has `,` `.` only where they are English, 、。 elsewhere). The
/// engine passes them through; a candidate that changed one is a
/// misconversion.
pub fn keeps_latin(reading: &str, candidate: &str) -> bool {
    // a full-width word is the same word (`symbol_width = "auto"`)
    let candidate: String = candidate.chars().map(crate::postprocess::to_half).collect();
    let mut rest = candidate.as_str();
    for word in reading
        .split(|c: char| !(c.is_ascii_alphanumeric() || is_joiner(c) || is_sentence_punct(c)))
        .filter(|w| w.chars().any(|c| c.is_ascii_alphabetic()))
    {
        // as a word of its own: no letter or digit glued on (Hhello)
        let alone = |i: usize| {
            !rest[..i].ends_with(|c: char| c.is_ascii_alphanumeric())
                && !rest[i + word.len()..].starts_with(|c: char| c.is_ascii_alphanumeric())
        };
        match rest.match_indices(word).map(|(i, _)| i).find(|&i| alone(i)) {
            Some(i) => rest = &rest[i + word.len()..],
            None => return false,
        }
    }
    true
}

/// A word piece of an English span read so far.
#[derive(Default)]
struct Piece {
    lower: String,
    /// As typed, its case kept.
    typed: String,
    /// The letter model's cost of its chars, and its context after them.
    letters: f64,
    ctx: [char; 2],
    digits: bool,
    /// The last two chars, for case breaks: two capitals then lowercase
    /// (`SCmo`, an acronym then romaji), but for a plural `s` at the end
    /// (`PCs`), counted once the piece goes on; lowercase then two capitals
    /// (`gaDJ`, romaji then an acronym).
    last: [char; 2],
    breaks: usize,
    s_break: bool,
}

impl Piece {
    fn is_empty(&self) -> bool {
        self.lower.is_empty()
    }

    fn push(&mut self, c: char, model: &MixedModel) {
        if self.lower.is_empty() {
            self.ctx = ['^', '^'];
            self.digits = true;
            self.last = [NONE, NONE];
        }
        if self.s_break {
            self.breaks += 1;
            self.s_break = false;
        }
        if self.last[0].is_ascii_uppercase()
            && self.last[1].is_ascii_uppercase()
            && c.is_ascii_lowercase()
        {
            if c == 's' {
                self.s_break = true;
            } else {
                self.breaks += 1;
            }
        }
        // lowercase, then capitals (`gaDJ`, `sateLEC`: romaji, then an
        // acronym)
        if self.last[0].is_ascii_lowercase()
            && self.last[1].is_ascii_uppercase()
            && c.is_ascii_uppercase()
        {
            self.breaks += 1;
        }
        self.last = [self.last[1], c];
        self.digits &= c.is_ascii_digit();
        let lower = c.to_ascii_lowercase();
        self.lower.push(lower);
        self.typed.push(c);
        let class = letter_class(lower);
        self.letters += model.letters.cost(self.ctx[0], self.ctx[1], class);
        self.ctx = [self.ctx[1], class];
    }
}

/// A lone `i` (at the input's start or after a space) is English when a space
/// and English follow it (`i think so`, `and i love it`): い before an
/// English word is next to never meant. Followed by nothing or by Japanese
/// it stays い.
fn lone_i(chars: &[char], spans: &mut [Span]) {
    for k in 0..spans.len().saturating_sub(1) {
        let (span, next) = (spans[k], spans[k + 1]);
        if span.lang == Lang::Ja
            && chars[span.start..span.end] == ['i']
            && (span.start == 0 || chars[span.start - 1] == ' ')
            && chars.get(span.end) == Some(&' ')
            && next.lang == Lang::En
        {
            spans[k].lang = Lang::En;
        }
    }
}

/// The ranges of `chars` never read as romaji (Meltype's protected spans, as
/// an idea): a URL (`https://…`, `www.…`), an e-mail address, a `@mention`,
/// a path (`C:\…`, `C:/…`, `\\host`, `~/…`, `./…`, `../…`, `/a/b`), each
/// from a word's start to the next space.
pub fn protected_ranges(chars: &[char]) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    let mut i = 0;
    while i < chars.len() {
        if chars[i] == ' ' {
            i += 1;
            continue;
        }
        let end = chars[i..]
            .iter()
            .position(|&c| c == ' ')
            .map_or(chars.len(), |k| i + k);
        let word: String = chars[i..end].iter().collect();
        let lower = word.to_ascii_lowercase();
        // a URL from its scheme on, romaji before it or not (詳細はhttps://…)
        let scheme_at = lower.find("://").and_then(|k| {
            ["https", "http", "ftp", "file", "ssh", "git"]
                .iter()
                .find(|s| lower[..k].ends_with(*s))
                .map(|s| lower[..k - s.len()].chars().count())
        });
        if let Some(at) = scheme_at {
            out.push((i + at, end));
            i = end;
            continue;
        }
        let email = word.split_once('@').is_some_and(|(local, domain)| {
            !local.is_empty() && domain.contains('.') && !domain.ends_with('.')
        });
        let mention = word.len() > 1
            && word.starts_with('@')
            && word[1..]
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-');
        let b = word.as_bytes();
        let drive = b.len() > 2
            && b[0].is_ascii_alphabetic()
            && b[1] == b':'
            && matches!(b[2], b'\\' | b'/');
        let path = drive
            || word.starts_with("\\\\")
            || word.starts_with("~/")
            || word.starts_with("./")
            || word.starts_with("../")
            || (word.starts_with('/') && word[1..].contains('/'));
        if lower.starts_with("www.") || email || mention || path {
            out.push((i, end));
        }
        i = end;
    }
    out
}

/// `spans` with each protected range (`protected_ranges`) one English span
/// of its own: never cut, never romaji.
fn protect(chars: &[char], spans: Vec<Span>) -> Vec<Span> {
    let ranges = protected_ranges(chars);
    if ranges.is_empty() {
        return spans;
    }
    let mut out: Vec<Span> = Vec::new();
    for s in spans {
        // the parts of `s` outside every range
        let mut start = s.start;
        for &(a, b) in &ranges {
            if b <= start || a >= s.end {
                continue;
            }
            if a > start {
                out.push(Span {
                    lang: s.lang,
                    start,
                    end: a,
                });
            }
            start = start.max(b);
        }
        if start < s.end {
            out.push(Span {
                lang: s.lang,
                start,
                end: s.end,
            });
        }
    }
    for &(a, b) in &ranges {
        out.push(Span {
            lang: Lang::En,
            start: a,
            end: b,
        });
    }
    out.sort_by_key(|s| s.start);
    out
}

/// Whether `typed` ends in a lone `i` (at its start or after a space): Space
/// there types a space, as after English, for the word after it to tell
/// whether it is I (`lone_i`); Space again converts い.
pub fn ends_in_lone_i(typed: &str) -> bool {
    typed
        .strip_suffix('i')
        .is_some_and(|rest| rest.is_empty() || rest.ends_with(' '))
}

/// The spans of an English sentence at the start of `spans` (0: none): two
/// English words or more at the start of the input, the text before it
/// ending a sentence (or none), with only spaces between them or `.` `,`
/// `!` `?` `;` `:` right after a word, up to the input's end or to a `.`
/// `!` `?` (English followed right away by Japanese, `pull request` + を, is
/// no sentence).
fn english_sentence(chars: &[char], spans: &[Span], before: Option<&str>) -> usize {
    let punct = |s: &Span, chars: &[char]| {
        s.lang == Lang::Ja
            && chars[s.start..s.end]
                .iter()
                .all(|c| matches!(c, '.' | ',' | '!' | '?' | ';' | ':'))
    };
    let ends_sentence = |s: &[char]| s.iter().any(|c| matches!(c, '.' | '!' | '?'));
    let before_ends = before.is_none_or(|b| {
        b.ends_with('\n')
            || b.trim_end()
                .chars()
                .last()
                .is_none_or(|c| matches!(c, '.' | '!' | '?' | '。' | '！' | '？'))
    });
    let Some(first) = spans.first() else {
        return 0;
    };
    if !before_ends || first.lang != Lang::En || chars[..first.start].iter().any(|&c| c != ' ') {
        return 0;
    }
    // the run: English words, and punctuation right after one
    let mut end = 1;
    while end < spans.len() {
        let (prev, next) = (spans[end - 1], spans[end]);
        let gap = &chars[prev.end..next.start];
        let joined = match next.lang {
            // a word after a space (`a.b` is no sentence)
            Lang::En => !gap.is_empty() && gap.iter().all(|&c| c == ' '),
            Lang::Ja => gap.is_empty() && prev.lang == Lang::En && punct(&next, chars),
        };
        if !joined {
            break;
        }
        end += 1;
    }
    let words = spans[..end].iter().filter(|s| s.lang == Lang::En).count();
    let last = spans[end - 1];
    let complete =
        end == spans.len() || last.lang == Lang::Ja && ends_sentence(&chars[last.start..last.end]);
    if words < 2 || !complete {
        return 0;
    }
    end
}

/// An English sentence's case (`english_sentence`): its punctuation becomes
/// English (as typed, not 、。), its words typed in lowercase start with a
/// capital at its start and after `.` `!` `?`, and `i` (`i'm`) is `I`.
fn english_case(chars: &mut [char], sentence: &mut [Span]) {
    let ends_sentence = |s: &[char]| s.iter().any(|c| matches!(c, '.' | '!' | '?'));
    let mut capital = true;
    for span in sentence {
        if span.lang == Lang::Ja {
            span.lang = Lang::En;
            capital |= ends_sentence(&chars[span.start..span.end]);
            continue;
        }
        let word = &mut chars[span.start..span.end];
        if !word.iter().any(|c| c.is_ascii_uppercase()) {
            let i = word[0] == 'i' && (word.len() == 1 || word.get(1) == Some(&'\''));
            if capital || i {
                word[0] = word[0].to_ascii_uppercase();
            }
        }
        capital = false;
    }
}

/// The last kana of the Japanese span right before `span` in `done` (no
/// space between), else `^`.
fn before_kana(done: &[Span], chars: &[char], span: Span) -> char {
    done.last()
        .filter(|s| s.lang == Lang::Ja && s.end == span.start)
        .and_then(|s| {
            romaji_units(&chars[s.start..s.end], true)
                .iter()
                .rev()
                .find_map(|u| match u.kind {
                    UnitKind::Kana(k) => k.chars().last(),
                    _ => None,
                })
        })
        .unwrap_or('^')
}

/// Punctuation that ends or splits a sentence, typed after a word.
fn is_sentence_punct(c: char) -> bool {
    matches!(c, '.' | ',' | '!' | '?' | ';' | ':')
}

/// Japanese text: kana, kanji and the CJK symbols (、。「」).
fn is_japanese(c: char) -> bool {
    matches!(c, '\u{3000}'..='\u{9FFF}' | '\u{F900}'..='\u{FAFF}' | '\u{FF61}'..='\u{FF9F}')
}

/// The sentence the text `before` the caret ends in: whether it has
/// Japanese in it, its English words, and whether the sentence before it
/// was Japanese (its end a 。 or kana).
pub(crate) fn sentence_of(before: &str) -> (bool, usize, bool) {
    let chars: Vec<char> = before.chars().collect();
    // the end of the last sentence: 。！？ or a line break, or `.` `!` `?`
    // before a space (not example.com)
    let end = (0..chars.len())
        .rev()
        .find(|&i| match chars[i] {
            '。' | '！' | '？' | '\n' => true,
            '.' | '!' | '?' => chars.get(i + 1).is_some_and(|c| c.is_whitespace()),
            _ => false,
        })
        .map_or(0, |i| i + 1);
    let tail: String = chars[end..].iter().collect();
    let japanese = tail.chars().any(is_japanese);
    let words = tail
        .split_whitespace()
        .filter(|w| w.chars().any(|c| c.is_ascii_alphabetic()))
        .count();
    let prev_ja = chars[..end]
        .iter()
        .rev()
        .find(|c| !c.is_whitespace())
        .is_some_and(|&c| is_japanese(c));
    (japanese, words, prev_ja)
}

/// Whether the sentence punctuation typed right after `before` is English
/// (`.` `,` `!` `?` as typed, not 、。): `before` ends in an English word
/// and its sentence has no Japanese in it. A sentence of one English word
/// after a Japanese one is taken as Japanese's (ありがとう。OK。).
pub fn english_after(before: &str) -> bool {
    if !before.ends_with(|c: char| c.is_ascii_alphabetic()) {
        return false;
    }
    let (japanese, words, prev_ja) = sentence_of(before);
    !japanese && (words >= 2 || !prev_ja)
}

/// The punctuation right after an English word is English when its
/// sentence is (`english_after`, the text before the caret and the spans
/// before it read as one text): `this is a pen.`, `hello, world.`, `.`
/// after an English text committed; これはgoogle。 stays.
fn english_punctuation(chars: &[char], spans: &mut [Span], before: &str) {
    let (mut japanese, mut words, mut prev_ja) = sentence_of(before);
    let after_before = before.ends_with(|c: char| c.is_ascii_alphabetic());
    for k in 0..spans.len() {
        let s = spans[k];
        let text = &chars[s.start..s.end];
        let ends = |c: &char| matches!(c, '.' | '!' | '?');
        if text.iter().all(|&c| is_sentence_punct(c)) {
            let after_word = if s.start == 0 {
                after_before
            } else {
                k > 0 && spans[k - 1].end == s.start && spans[k - 1].lang == Lang::En
            };
            let english = s.lang == Lang::En || after_word && !japanese && (words >= 2 || !prev_ja);
            if english {
                spans[k].lang = Lang::En;
            }
            if text.iter().any(ends) {
                (japanese, words, prev_ja) = (false, 0, !english);
            }
        } else if s.lang == Lang::En {
            words += 1;
        } else if let Some(last) = text.iter().rposition(ends) {
            // a Japanese span with a sentence's end in it (けんさく。)
            japanese = text[last + 1..].iter().any(|c| c.is_ascii_alphabetic());
            (words, prev_ja) = (0, true);
        } else if text
            .iter()
            .any(|&c| c.is_ascii_alphabetic() || is_japanese(c))
        {
            japanese = true;
        }
    }
}

/// Whether `chars[at]` is a `.` right after a digit (2.8, and 2. before the
/// next digit is typed): a 。 there is next to never meant.
fn is_decimal_point(chars: &[char], at: usize) -> bool {
    chars[at] == '.' && at > 0 && chars[at - 1].is_ascii_digit()
}

/// Symbols an English span may have between its pieces.
fn is_joiner(c: char) -> bool {
    matches!(c, '.' | '-' | '+' | '\'' | '@' | '/' | ':' | '_')
}

/// The language the text before the caret ends in.
fn context_lang(before: &str) -> Option<Lang> {
    let c = before.trim_end().chars().last()?;
    if c.is_ascii_alphabetic() {
        Some(Lang::En)
    } else if c.is_ascii() {
        None
    } else {
        Some(Lang::Ja)
    }
}

/// A char of the context as the kana model sees it: katakana as hiragana,
/// a Latin word's letter as `@`.
fn context_class(c: char) -> char {
    match c {
        '\u{30A1}'..='\u{30F6}' => char::from_u32(c as u32 - 0x60).unwrap_or(c),
        c if c.is_ascii_alphabetic() => '@',
        c => kana_class(c),
    }
}

/// A non-letter as the kana model sees it.
fn kana_class(c: char) -> char {
    match c {
        // kana already in the composition (frozen by a cursor edit)
        '\u{3041}'..='\u{3096}' | 'ー' | '、' | '。' => c,
        '0'..='9' => '0',
        ',' => '、',
        '.' => '。',
        _ => '#',
    }
}

fn letter_class(c: char) -> char {
    match c {
        'a'..='z' | '\'' => c,
        '0'..='9' => '0',
        _ => '#',
    }
}

/// A hash map with a fast multiply-rotate hash (FxHash): the models look
/// up small keys (char trigrams, short words) many times per keystroke.
type FastMap<K, V> = HashMap<K, V, BuildHasherDefault<FastHasher>>;

#[derive(Default)]
struct FastHasher(u64);

impl FastHasher {
    fn add(&mut self, word: u64) {
        self.0 = (self.0.rotate_left(5) ^ word).wrapping_mul(0x51_7c_c1_b7_27_22_0a_95);
    }
}

impl Hasher for FastHasher {
    fn write(&mut self, bytes: &[u8]) {
        let mut chunks = bytes.chunks_exact(8);
        for chunk in &mut chunks {
            self.add(u64::from_le_bytes(chunk.try_into().unwrap_or_default()));
        }
        for &b in chunks.remainder() {
            self.add(b as u64);
        }
    }

    fn write_u8(&mut self, n: u8) {
        self.add(n as u64);
    }

    fn write_u32(&mut self, n: u32) {
        self.add(n as u64);
    }

    fn write_u64(&mut self, n: u64) {
        self.add(n);
    }

    fn write_usize(&mut self, n: usize) {
        self.add(n as u64);
    }

    fn finish(&self) -> u64 {
        self.0
    }
}

/// A char trigram model with Witten–Bell interpolation, from counts of
/// 1- to 3-grams.
#[derive(Default)]
struct CharLm {
    counts: FastMap<(char, char, char), u32>,
    /// Per context (1 or 2 chars, the first `\0` for one): total count and
    /// number of distinct chars that follow it.
    contexts: FastMap<(char, char), (u64, u32)>,
    unigram_total: u64,
    vocab: u32,
}

const NONE: char = '\0';

impl CharLm {
    fn add(&mut self, gram: &str, count: u32) {
        let chars: Vec<char> = gram.chars().collect();
        self.add_chars(&chars, count);
    }

    fn add_chars(&mut self, chars: &[char], count: u32) {
        let key = match *chars {
            [c] => (NONE, NONE, c),
            [b, c] => (NONE, b, c),
            [a, b, c] => (a, b, c),
            _ => return,
        };
        *self.counts.entry(key).or_insert(0) += count;
    }

    fn finish(&mut self) {
        self.contexts.clear();
        self.unigram_total = 0;
        self.vocab = 0;
        for (&(a, b, _), &count) in &self.counts {
            if b == NONE {
                self.unigram_total += count as u64;
                self.vocab += 1;
            } else {
                let e = self.contexts.entry((a, b)).or_insert((0, 0));
                e.0 += count as u64;
                e.1 += 1;
            }
        }
    }

    /// −ln P(c | a b).
    fn cost(&self, a: char, b: char, c: char) -> f64 {
        let count = |k: (char, char, char)| *self.counts.get(&k).unwrap_or(&0) as f64;
        let p1 =
            (count((NONE, NONE, c)) + 1.0) / (self.unigram_total as f64 + self.vocab as f64 + 1.0);
        let interpolate = |ctx: (char, char), key: (char, char, char), lower: f64| match self
            .contexts
            .get(&ctx)
        {
            Some(&(total, types)) => {
                (count(key) + types as f64 * lower) / (total as f64 + types as f64)
            }
            None => lower,
        };
        let p2 = interpolate((NONE, b), (NONE, b, c), p1);
        let p3 = interpolate((a, b), (a, b, c), p2);
        -p3.ln()
    }
}

/// What the spans of `chars` show, and the romaji still pending at its end
/// (shown after it, not yet part of the reading): Japanese as kana with its
/// ASCII symbols as `symbols` maps them (else `,` `.` as 、。 and the others
/// full width), English as typed, chars between spans as they are. With
/// `fin`, a final `n` is ん and other pending romaji is text. Romaji is
/// pending at the end of `chars` only: a Japanese span with more after it
/// is done, its `n` ん and a letter no kana took text where it was
/// (`gboard`, `g` and English `board`: not `board` with `g` pending after).
pub fn render_parts(
    chars: &[char],
    spans: &[Span],
    fin: bool,
    symbols: &HashMap<char, String>,
) -> (String, String) {
    let mut text = String::new();
    let mut pending = String::new();
    let mut pos = 0;
    // the text ends in Latin: an English span's letter or digit, or a digit
    // (a letter romaji left over is a slip, not Latin: つよしh。)
    let mut latin_end = false;
    for span in spans {
        if pos < span.start {
            text.extend(&chars[pos..span.start]);
            latin_end = false;
        }
        let part = &chars[span.start..span.end];
        match span.lang {
            Lang::En => {
                text.extend(part);
                latin_end = part.last().is_some_and(|c| c.is_ascii_alphanumeric());
            }
            Lang::Ja => {
                let done = fin || span.end < chars.len();
                let laughs = laughter(part);
                for unit in romaji_units(part, done) {
                    let after_latin = std::mem::replace(&mut latin_end, false);
                    if laughs[unit.start] {
                        text.extend(&part[unit.start..unit.end]);
                        continue;
                    }
                    match unit.kind {
                        // a hyphen after a digit or a Latin letter is one
                        // (3-5-2, NEWS-), not ー (らーめん)
                        UnitKind::Kana("ー") if after_latin => text.push('-'),
                        UnitKind::Kana(kana) => text.push_str(kana),
                        UnitKind::Char(c) if !c.is_ascii_punctuation() => {
                            text.push(c);
                            latin_end = c.is_ascii_digit();
                        }
                        UnitKind::Char(c) if is_decimal_point(part, unit.start) => text.push(c),
                        UnitKind::Char('.')
                            if period_as_typed(part, unit.start, &text, after_latin) =>
                        {
                            text.push('.')
                        }
                        UnitKind::Char(c) => match symbols.get(&c) {
                            Some(s) => text.push_str(s),
                            None => text.push(match c {
                                ',' => '、',
                                '.' => '。',
                                _ => char::from_u32(c as u32 - 0x21 + 0xff01).unwrap_or(c),
                            }),
                        },
                        UnitKind::Pending if done => text.extend(&part[unit.start..unit.end]),
                        UnitKind::Pending => pending.extend(&part[unit.start..unit.end]),
                    }
                }
            }
        }
        pos = span.end;
    }
    text.extend(&chars[pos..]);
    (text, pending)
}

/// The chars of `part` in a laugh: two `w` or more that no romaji goes on
/// from (the end, a space, a symbol after them), as typed (`ww`, not っw);
/// a vowel typed after makes them romaji again (`wwa`, っわ).
fn laughter(part: &[char]) -> Vec<bool> {
    let mut out = vec![false; part.len()];
    let mut i = 0;
    while i < part.len() {
        if !part[i].eq_ignore_ascii_case(&'w') {
            i += 1;
            continue;
        }
        let j = (i..part.len())
            .find(|&j| !part[j].eq_ignore_ascii_case(&'w'))
            .unwrap_or(part.len());
        let ends = part.get(j).is_none_or(|c| !c.is_ascii_alphabetic());
        if j - i >= 2 && ends {
            out[i..j].fill(true);
        }
        i = j;
    }
    out
}

/// A period of Japanese text that stays one: in a run of periods (`...`,
/// not 。。。) or after an initial (`J.`, `A.R.C.`: a Latin letter
/// `after_latin`, with none before it). After a word it is the sentence's
/// end (Google。).
fn period_as_typed(part: &[char], at: usize, shown: &str, after_latin: bool) -> bool {
    let run = at > 0 && part[at - 1] == '.' || part.get(at + 1) == Some(&'.');
    let mut back = shown.chars().rev();
    let initial = after_latin
        && back.next().is_some_and(|c| c.is_ascii_alphabetic())
        && !back.next().is_some_and(|c| c.is_ascii_alphabetic());
    run || initial
}

/// What the spans of `typed` show (`render_parts`, default symbols, the
/// pending romaji appended).
pub fn render(typed: &str, spans: &[Span], fin: bool) -> String {
    let chars: Vec<char> = typed.chars().collect();
    let (text, pending) = render_parts(&chars, spans, fin, &HashMap::new());
    text + &pending
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// A small model: the kana of a few readings, a few Latin words with
    /// their counts in Japanese and English text, tuned-like parameters.
    pub(crate) fn model() -> MixedModel {
        let readings = [
            "きょうはいいてんきですね",
            "あしたまでにきてください",
            "そのためにべんきょうする",
            "でけんさくした。",
            "とうきょうにすむ",
            "かいぎはさんじからです",
            "これはぺんです。",
            "わたしはがくせいです。",
            "ありがとうございます。",
            "けんさくしてみた",
            "にほんごをはなす",
            "ばいになる",
            // Latin words as `@`, as in mixed text
            "@をつかう",
            "@のせってい",
            "@でけんさくした。",
            "@をひかくした。",
        ];
        let mut grams: HashMap<String, u32> = HashMap::new();
        for r in readings {
            let chars: Vec<char> = format!("^{r}$").chars().collect();
            for n in 1..=3 {
                for g in chars.windows(n) {
                    *grams.entry(g.iter().collect()).or_insert(0) += 1;
                }
            }
        }
        let kana: String = grams.iter().map(|(g, c)| format!("{g}\t{c}\n")).collect();
        let latin = "google\t500\t100\ngithub\t100\t20\nmade\t5\t500\ntokyo\t50\t50\n\
                     hello\t100\t300\nexample\t50\t200\ncom\t300\t500\nexample.com\t20\t50\n\
                     https\t100\t100\ngo\t50\t2000\nk\t300\t300\nmelty\t100\t10\n";
        let params = "switch_in=1\nswitch_out=5.7\nen_en=8.4\nstart_en=3\nleftover=30.7\n\
                      upper=30.7\nknown=0.95\nja_text=0.99\nsymbol=5.9\ndigit=2.7\ncase_break=11.2\n";
        MixedModel::from_texts(&kana, latin, Some(params)).unwrap()
    }

    fn shown(model: &MixedModel, typed: &str, before: Option<&str>) -> String {
        render(typed, &model.segment(typed, before, true), true)
    }

    #[test]
    fn english_words_among_romaji_stay_as_typed() {
        let m = model();
        assert_eq!(
            shown(&m, "kyouhagoogledekensakushita", None),
            "きょうはgoogleでけんさくした"
        );
        assert_eq!(shown(&m, "google", None), "google");
        assert_eq!(
            shown(&m, "https://example.com", None),
            "https://example.com"
        );
    }

    #[test]
    fn romaji_spelling_an_english_word_stays_japanese() {
        let m = model();
        assert_eq!(shown(&m, "ashitamadenikite", None), "あしたまでにきて");
    }

    #[test]
    fn an_unknown_word_typed_in_lowercase_costs_oov() {
        // a slip of the fingers among romaji is no word the model knows
        let mut m = model();
        m.params.leftover = 10.0;
        let typed = "kaigihasanjikarakwdsu";
        assert_eq!(shown(&m, typed, None), "かいぎはさんじからkwdsu");
        m.params.oov = 8.0;
        assert_eq!(shown(&m, typed, None), "かいぎはさんじからkwdす");
        // typed with Shift it is English still
        assert_eq!(
            shown(&m, "kaigihasanjikaraKwdsu", None),
            "かいぎはさんじからKwdsu"
        );
    }

    #[test]
    fn shift_marks_english() {
        let m = model();
        assert_eq!(shown(&m, "Tokyonisumu", None), "Tokyoにすむ");
        assert_eq!(shown(&m, "toukyounisumu", None), "とうきょうにすむ");
    }

    #[test]
    fn the_text_before_decides_a_word_of_both() {
        let m = model();
        assert_eq!(shown(&m, "go", Some("日本")), "ご");
        assert_eq!(shown(&m, "go", Some("Let's ")), "go");
    }

    #[test]
    fn a_sentence_after_another_starts_over() {
        let m = model();
        let cost = |s: &str| {
            m.ja_costs(&s.chars().collect::<Vec<_>>(), true, ['^', '^'])
                .last()
                .unwrap()
                .1
        };
        // The kana after 。 cost what they cost at a reading's start: the
        // model read one sentence per reading, and 。こ was a rare pair
        // (`desu.kore` read as `です。k` + おれ).
        let after = cost("desu.kore") - cost("desu.");
        assert!(
            (after - cost("kore")).abs() < 1e-9,
            "{after} vs {}",
            cost("kore")
        );
        assert_eq!(
            shown(&m, "korehapendesu.korehapendesu.", None),
            "これはぺんです。これはぺんです。"
        );
        assert_eq!(
            shown(&m, "googledekensakushita.korehapendesu", None),
            "googleでけんさくした。これはぺんです"
        );
    }

    #[test]
    fn kana_after_an_english_word_read_on_from_it() {
        let m = model();
        let cost = |s: &str, ctx: [char; 2]| {
            m.ja_costs(&s.chars().collect::<Vec<_>>(), true, ctx)
                .last()
                .unwrap()
                .1
        };
        // を after a Latin word is common, at a sentence start it is not
        assert!(cost("wo", ['@', '@']) < cost("wo", ['^', '^']));
    }

    #[test]
    fn decimals_and_pending_romaji() {
        let m = model();
        assert_eq!(shown(&m, "2.8baininaru", None), "2.8ばいになる");
        // a period after a digit stays one before the next digit is typed
        assert_eq!(shown(&m, "2.", None), "2.");
        assert_eq!(shown(&m, "desu.", None), "です。");
        let chars: Vec<char> = "kyouk".chars().collect();
        let spans = m.segment("kyouk", None, false);
        assert_eq!(
            render_parts(&chars, &spans, false, &HashMap::new()),
            ("きょう".to_string(), "k".to_string())
        );
    }

    /// Romaji is pending at the end of the input only: before an English
    /// word, a letter no kana took stays where it was typed and an n is ん.
    #[test]
    fn romaji_before_english_is_done() {
        let span = |lang, start, end| Span { lang, start, end };
        let parts = |typed: &str, spans: &[Span]| {
            let chars: Vec<char> = typed.chars().collect();
            render_parts(&chars, spans, false, &HashMap::new())
        };
        let shown = |text: &str, pending: &str| (text.to_string(), pending.to_string());
        assert_eq!(
            parts("gboard", &[span(Lang::Ja, 0, 1), span(Lang::En, 1, 6)]),
            shown("gboard", "")
        );
        assert_eq!(
            parts("honBook", &[span(Lang::Ja, 0, 3), span(Lang::En, 3, 7)]),
            shown("ほんBook", "")
        );
        // at the end it is still being typed
        assert_eq!(
            parts("Bookk", &[span(Lang::En, 0, 4), span(Lang::Ja, 4, 5)]),
            shown("Book", "k")
        );
    }

    /// Japanese text's ASCII punctuation as text writes it beside Latin:
    /// a hyphen after a digit or a letter, a run of periods, an initial's
    /// period; ー and 。 elsewhere.
    #[test]
    fn punctuation_beside_latin_stays_ascii() {
        let span = |lang, start, end| Span { lang, start, end };
        let shown = |typed: &str, spans: &[Span]| render(typed, spans, true);
        let ja = |typed: &str| shown(typed, &[span(Lang::Ja, 0, typed.chars().count())]);
        assert_eq!(ja("3-5-2no"), "3-5-2の");
        assert_eq!(ja("ra-men"), "らーめん");
        assert_eq!(ja("chotto..."), "ちょっと...");
        assert_eq!(ja("chotto."), "ちょっと。");
        // after a letter romaji left over (a slip): 。 and ー
        assert_eq!(ja("tsuyosih."), "つよしh。");
        assert_eq!(ja("as-bisu"), "あsーびす");
        // a laugh as typed, a double consonant as っ
        assert_eq!(ja("shinjimasennww"), "しんじませんww");
        assert_eq!(ja("kusawww,hai"), "くさwww、はい");
        assert_eq!(ja("wwwa"), "っっわ");
        assert_eq!(
            shown(
                "NEWS-hyakunen",
                &[span(Lang::En, 0, 4), span(Lang::Ja, 4, 13)]
            ),
            "NEWS-ひゃくねん"
        );
        assert_eq!(
            shown("J.seruko", &[span(Lang::En, 0, 1), span(Lang::Ja, 1, 8)]),
            "J.せるこ"
        );
        assert_eq!(
            shown(
                "Google.kyou",
                &[span(Lang::En, 0, 6), span(Lang::Ja, 6, 11)]
            ),
            "Google。きょう"
        );
    }

    #[test]
    fn a_particle_ends_an_unknown_english_word() {
        let m = model();
        assert_eq!(shown(&m, "Ollamade", None), "Ollamaで");
        assert_eq!(shown(&m, "Qwenwotsukau", None), "Qwenをつかう");
        // the first split: Ollama + のです, not Ollamano + です
        assert_eq!(shown(&m, "Ollamanodesu", None), "Ollamaのです");
        assert_eq!(shown(&m, "Wi-Fino", None), "Wi-Fiの");
        // a known word stays whole
        assert_eq!(shown(&m, "Made", None), "Made");
        // romaji still being typed after it: not yet
        let spans = m.segment("Ollamad", None, false);
        assert_eq!(render("Ollamad", &spans, false), "Ollamad");
        // up to a capital, where English starts again
        assert_eq!(
            shown(&m, "TealiumnoCDPwotsukau", None),
            "TealiumのCDPをつかう"
        );
        assert_eq!(shown(&m, "Juddraha", None), "Juddらは");
        let mut m = m;
        m.add_names("photo\t1\n");
        assert_eq!(shown(&m, "PhotoShop", None), "PhotoShop");
    }

    #[test]
    fn a_word_may_end_in_plus_or_sharp() {
        let m = model();
        assert_eq!(shown(&m, "NAD+nohokyuu", None), "NAD+のほきゅう");
        assert_eq!(shown(&m, "C++wotsukau", None), "C++をつかう");
        assert_eq!(shown(&m, "C#wotsukau", None), "C#をつかう");
    }

    /// What `typed` shows, cased (`MixedModel::cased`).
    fn cased(model: &MixedModel, typed: &str, before: Option<&str>) -> String {
        let (chars, spans) = model.cased(typed, before, true, Casing::default());
        let (text, pending) = render_parts(&chars, &spans, true, &HashMap::new());
        text + &pending
    }

    #[test]
    fn a_name_typed_in_lowercase_takes_its_case() {
        let mut m = model();
        m.add_names(
            "# names\nGitHub\t1\nGithub\t0.1\nHello\t0.4\nhello\t0.6\nkubectl\t0.2\n\
             Node.js\t0.5\nVisual Studio\t1\nSushi\t1\n",
        );
        assert_eq!(cased(&m, "githubwotsukau", None), "GitHubをつかう");
        // typed with Shift: as typed
        assert_eq!(cased(&m, "Githubwotsukau", None), "Githubをつかう");
        // an ordinary word better known than the name
        assert_eq!(m.name_case("hello", NameCase::Proper), None);
        assert_eq!(m.name_case("github", NameCase::Proper), Some("GitHub"));
        // romaji all through: Japanese
        assert!(!m.knows("sushi"));
        // a name is an English word
        assert_eq!(cased(&m, "kubectlwotsukau", None), "kubectlをつかう");
        assert_eq!(cased(&m, "node.jsnosettei", None), "Node.jsのせってい");
        // the case is shown, the input is as typed
        assert_eq!(m.cased("github", None, true, Casing::default()).0.len(), 6);
        assert!(m.knows("NODE.JS") && !m.knows("visual"));
    }

    #[test]
    fn the_name_case_level_picks_the_names() {
        let mut m = model();
        m.add_names("GitHub\t1\nGOOGLE\t0.5\tcommon\n");
        // a common name stays as typed at every level; acronyms by rule
        let at = |names: NameCase| {
            let casing = Casing {
                names,
                ..Casing::default()
            };
            let (chars, spans) = m.cased("githubtogooglewotsukau", None, true, casing);
            let (text, pending) = render_parts(&chars, &spans, true, &HashMap::new());
            text + &pending
        };
        assert_eq!(at(NameCase::Acronyms), "GitHubとgoogleをつかう");
        assert_eq!(at(NameCase::Proper), "GitHubとgoogleをつかう");
        assert_eq!(at(NameCase::Off), "githubとgoogleをつかう");
        // a name is an English word whatever the level
        assert!(m.knows("github"));
    }

    #[test]
    fn acronyms_by_their_spelling() {
        let a = |w: &str, s: bool| acronym(&w.chars().collect::<Vec<_>>(), s);
        for w in [
            "pc", "sns", "css", "pdf", "kpi", "usb", "etf", "atm", "faq", "gpt-4", "mp3",
        ] {
            assert!(a(w, false) && a(w, true), "{w}");
        }
        for w in ["ok", "api", "id", "it", "url"] {
            assert!(a(w, false) && !a(w, true), "{w}");
        }
        for w in [
            "word", "apple", "hmm", "test", "branch", "strength", "win11", "x", "google", "km",
            "kwh", "ctags",
        ] {
            assert!(!a(w, false), "{w}");
        }
        assert!(a("https", false));
        let mut m = model();
        assert_eq!(cased(&m, "pcwokatta", None), "PCをかった");
        // a name in lowercase stays
        m.add_names("npm\t1\n");
        assert_eq!(cased(&m, "npmwotsukau", None), "npmをつかう");
        let proper = Casing {
            names: NameCase::Proper,
            ..Casing::default()
        };
        let (chars, spans) = m.cased("pcwokatta", None, true, proper);
        assert_eq!(
            render_parts(&chars, &spans, true, &HashMap::new()).0,
            "pcをかった"
        );
    }

    #[test]
    fn an_english_sentence_starts_with_a_capital() {
        let mut m = model();
        m.add_names("World\t1\nnice\t1\nGitHub\t1\n");
        // a name with a capital inside only (`World` is a word here)
        assert_eq!(cased(&m, "hello world", None), "Hello world");
        assert_eq!(cased(&m, "hello github", None), "Hello GitHub");
        assert_eq!(cased(&m, "worldwotsukau", None), "Worldをつかう");
        // off: as typed, punctuation as Japanese
        let off = Casing {
            sentences: false,
            ..Casing::default()
        };
        let (chars, spans) = m.cased("hello nice.", None, true, off);
        assert_eq!(
            render_parts(&chars, &spans, true, &HashMap::new()).0,
            "hello nice。"
        );
        assert_eq!(
            cased(&m, "hello world. nice world", None),
            "Hello world. Nice world"
        );
        assert_eq!(cased(&m, "hello, world!", None), "Hello, world!");
        assert_eq!(cased(&m, "hello world", Some("Fine. ")), "Hello world");
        assert_eq!(cased(&m, "hello.world", None), "hello.world");
        // one word is no sentence, nor English inside Japanese
        assert_eq!(cased(&m, "hello", None), "hello");
        // but its punctuation is English, in a sentence with no Japanese
        assert_eq!(cased(&m, "hello.", None), "hello.");
        assert_eq!(cased(&m, "hello.", Some("ありがとう。")), "hello。");
        assert_eq!(cased(&m, "hello nice", Some("きょうは")), "hello nice");
        assert_eq!(cased(&m, "hello nicewotsukau", None), "hello niceをつかう");
        // a sentence and then Japanese
        assert_eq!(
            cased(&m, "hello world. kyouhaiitenkidesune", None),
            "Hello world. きょうはいいてんきですね"
        );
    }

    #[test]
    fn a_lone_i_before_english_is_i() {
        let m = model();
        assert_eq!(cased(&m, "i hello google", None), "I hello google");
        assert_eq!(cased(&m, "hello i google", None), "Hello I google");
        // before Japanese, or nothing, it is い
        assert_eq!(cased(&m, "i", None), "い");
        assert_eq!(cased(&m, "i kyouha", None), "い きょうは");
        assert!(ends_in_lone_i("i") && ends_in_lone_i("hello i"));
        assert!(!ends_in_lone_i("kai") && !ends_in_lone_i("i ") && !ends_in_lone_i("I"));
    }

    #[test]
    fn the_shipped_model_reads_english_after_i() {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("data/mixed");
        let m = MixedModel::load(&dir).unwrap();
        assert_eq!(cased(&m, "i think so", None), "I think so");
        assert_eq!(cased(&m, "i love you", None), "I love you");
        // romaji before a space is done: no kana is coming for its k
        assert_eq!(cased(&m, "they think so", None), "They think so");
        assert_eq!(
            cased(&m, "kyouhaiitenkidesune", None),
            "きょうはいいてんきですね"
        );
    }

    /// A letter the Japanese reading leaves, with the letters around it a
    /// Latin word the model knows, is that word (ぶg was `bug`); a slip that
    /// makes no word it knows well stays a slip; and romaji, then capitals,
    /// is Japanese, then an acronym (gaDJ).
    #[test]
    fn the_shipped_model_reads_a_left_letter_as_its_word() {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("data/mixed");
        let m = MixedModel::load(&dir).unwrap();
        assert_eq!(cased(&m, "sorehabugdesu", None), "それはbugです");
        assert_eq!(cased(&m, "kpinomokuhyou", None), "KPIのもくひょう");
        assert_eq!(cased(&m, "furqnoekusupuresu", None), "ふrqのえくすぷれす");
        assert_eq!(cased(&m, "gaDJnanode", None), "がDJなので");
        assert_eq!(cased(&m, "sateLECtakken", None), "さてLECたっけん");
        // off: the slip's letter as typed
        let mut off = MixedModel::load(&dir).unwrap();
        off.params.letter_word = 0.0;
        assert_eq!(cased(&off, "sorehabugdesu", None), "それはぶgです");
    }

    /// A word romaji reads wholly as kana is English where the lexicon
    /// finds its kana dear as Japanese (一 + 据え for いっすえ: "issue立てと
    /// いて" was 一据え立てといて), not where they are a word (まで).
    #[test]
    fn the_lexicon_reads_a_word_spelled_in_kana_as_english_where_no_word_is() {
        let table = "立て\tたて\t5.0\tverb\nと\tと\t3.0\tpart\nいて\tいて\t5.0\tverb\n\
                     ここ\tここ\t4.0\tnoun\nまで\tまで\t3.0\tpart\n来た\tきた\t4.0\tverb\n\
                     一\tいっ\t9.0\tnoun\n据え\tすえ\t11.0\tverb\n";
        let path = std::env::temp_dir().join(format!(
            "setsuna-mixed-kana-words-{}.lexicon",
            std::process::id()
        ));
        std::fs::write(&path, Lexicon::build(table, None).unwrap()).unwrap();
        let lexicon = Lexicon::open(&path).unwrap();
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("data/mixed");
        let m = MixedModel::load(&dir).unwrap();
        let with = |typed: &str| {
            let (chars, spans) = m.cased_with(typed, None, true, Casing::default(), Some(&lexicon));
            let (text, pending) = render_parts(&chars, &spans, true, &HashMap::new());
            text + &pending
        };
        assert_eq!(cased(&m, "issuetatetoite", None), "いっすえたてといて");
        assert_eq!(with("issuetatetoite"), "issueたてといて");
        assert_eq!(with("kokomadekita"), "ここまできた");
        drop(lexicon);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn punctuation_after_english_is_the_sentences() {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("data/mixed");
        let m = MixedModel::load(&dir).unwrap();
        // a 。 after it once made this all Japanese (this is あ ぺん。)
        assert_eq!(cased(&m, "this is a pen.", None), "This is a pen.");
        assert_eq!(cased(&m, "it works!", None), "It works!");
        assert_eq!(
            cased(&m, "I think so, demo ikanai.", None),
            "I think so, でも いかない。"
        );
        assert_eq!(cased(&m, "korehagoogle.", None), "これはgoogle。");
        // after the text before the caret
        assert_eq!(cased(&m, ".", Some("This is a pen")), ".");
        assert_eq!(cased(&m, ", and", Some("This is a pen")), ", and");
        assert_eq!(cased(&m, ".", Some("これはGoogle")), "。");
    }

    #[test]
    fn a_sentence_wins_over_a_rare_word_it_overlaps() {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("data/mixed");
        let mut m = MixedModel::load(&dir).unwrap();
        m.add_names(
            "DLsite	0.3
",
        );
        // dl + して over the name, but the name before a particle
        assert_eq!(cased(&m, "dlsite", None), "dlして");
        assert_eq!(cased(&m, "dlsitemita", None), "dlしてみた");
        assert_eq!(cased(&m, "dlsitede", None), "DLsiteで");
        // a common word stays, and so does the name typed as it is written
        // (as the composition is read again on Space)
        assert_eq!(cased(&m, "website", None), "website");
        assert_eq!(cased(&m, "DLsite", None), "DLsite");
        assert_eq!(cased(&m, "DLsiteで", None), "DLsiteで");
    }

    #[test]
    fn english_after_reads_the_sentence_before() {
        assert!(english_after("This is a pen"));
        assert!(english_after("OK"));
        assert!(english_after("Fine. OK"));
        assert!(english_after("ありがとう。Thank you"));
        // Japanese in the sentence, or one word after a Japanese one
        assert!(!english_after("これはGoogle"));
        assert!(!english_after("ありがとう。OK"));
        // not right after a word
        assert!(!english_after("Hello "));
        assert!(!english_after("100"));
        assert!(!english_after(""));
    }

    #[test]
    fn urls_addresses_and_paths_stay_as_typed() {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("data/mixed");
        let m = MixedModel::load(&dir).unwrap();
        assert_eq!(
            shown(&m, "shousaihahttps://example.com/a wo mite", None),
            "しょうさいはhttps://example.com/a を みて"
        );
        assert_eq!(
            shown(&m, "renrakusaki foo.bar@example.co.jp", None),
            "れんらくさき foo.bar@example.co.jp"
        );
        assert_eq!(
            shown(&m, "~/projects/setsuna niaru", None),
            "~/projects/setsuna にある"
        );
        assert_eq!(shown(&m, "@kasano san", None), "@kasano さん");
        assert_eq!(
            protected_ranges(&"kasano".chars().collect::<Vec<_>>()),
            vec![]
        );
    }

    #[test]
    fn candidates_must_keep_the_latin_words() {
        assert!(keeps_latin("きょうはgoogleで", "今日はgoogleで"));
        assert!(!keeps_latin("きょうはgoogleで", "今日はGoogleで"));
        assert!(!keeps_latin("hello", "Hhello"));
        assert!(keeps_latin("hello", "hello!"));
        assert!(keeps_latin("きょうは", "今日は"));
        assert!(keeps_latin("きょうはgoogleで", "今日はｇｏｏｇｌｅで"));
    }
}
