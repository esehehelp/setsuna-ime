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
//! The model files are counts: `kana.tsv` (`<n-gram>\t<count>`, n = 1..3,
//! `^` a reading's start, `0` any digit, `#` any other character) and
//! `latin.tsv` (`<word>\t<count in Japanese text>\t<count in English text>`),
//! with `params.txt` (`name=value` lines) overriding [`MixedParams`].

use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};
use std::path::Path;
use std::sync::{Arc, Mutex, OnceLock};

use anyhow::{Context, Result};

use crate::composing_text::{pending_kana, romaji_units, UnitKind};

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
}

impl Default for MixedParams {
    fn default() -> Self {
        Self {
            switch_in: 6.0,
            switch_out: 6.0,
            en_en: 6.0,
            start_en: 3.0,
            leftover: 8.0,
            upper: 8.0,
            known: 0.9,
            ja_text: 0.7,
            symbol: 3.0,
            digit: 1.0,
            case_break: 8.0,
        }
    }
}

impl MixedParams {
    pub const NAMES: [&'static str; 11] = [
        "switch_in",
        "switch_out",
        "en_en",
        "start_en",
        "leftover",
        "upper",
        "known",
        "ja_text",
        "symbol",
        "digit",
        "case_break",
    ];

    pub fn field(&mut self, name: &str) -> Option<&mut f64> {
        Some(match name {
            "switch_in" => &mut self.switch_in,
            "switch_out" => &mut self.switch_out,
            "en_en" => &mut self.en_en,
            "start_en" => &mut self.start_en,
            "leftover" => &mut self.leftover,
            "upper" => &mut self.upper,
            "known" => &mut self.known,
            "ja_text" => &mut self.ja_text,
            "symbol" => &mut self.symbol,
            "digit" => &mut self.digit,
            "case_break" => &mut self.case_break,
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
    pub params: MixedParams,
}

impl std::fmt::Debug for MixedModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MixedModel")
            .field("words", &self.words.len())
            .field("params", &self.params)
            .finish_non_exhaustive()
    }
}

/// The model in `dir`, loaded once per process; `None` (logged) when it
/// cannot be read.
pub fn shared(dir: &str) -> Option<Arc<MixedModel>> {
    static CACHE: OnceLock<Mutex<HashMap<String, Option<Arc<MixedModel>>>>> = OnceLock::new();
    let mut cache = CACHE
        .get_or_init(Default::default)
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    cache
        .entry(dir.to_string())
        .or_insert_with(|| match MixedModel::load(Path::new(dir)) {
            Ok(model) => Some(Arc::new(model)),
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
/// right before them, as romaji: が を で の に は と も へ から より.
/// 側 (gawa) starts with が, です with で. Not まで: `Ollamade` would
/// split before `ma` (Olla + まで).
const PARTICLES: &[&str] = &[
    "ga", "wo", "de", "no", "ni", "ha", "to", "mo", "he", "kara", "yori",
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
            params,
        })
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
        // The state after a span: Japanese, English, or Japanese that an
        // English span follows right away (its cost has the odds of `@`).
        const JA: usize = 0;
        const EN: usize = 1;
        const JA_EN: usize = 2;
        const INF: f64 = f64::INFINITY;
        let chars: Vec<char> = typed.chars().collect();
        let n = chars.len();
        let p = &self.params;
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
            let seg_fin = fin && seg_end == n;
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
                    update(start + len, EN, c0 + enter + cost, prev);
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
        let mut spans = Vec::new();
        while end > 0 {
            if !best[end][state].is_finite() {
                // Unreachable (only spaces): give up on segmenting.
                return vec![Span {
                    lang: Lang::Ja,
                    start: 0,
                    end: n,
                }];
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
        self.split_particles(&chars, spans, fin)
    }

    /// An English span the model does not know as a word that ends in a
    /// particle read as romaji (`Ollamade`, `Qwenno`) is English up to it
    /// and Japanese from it (Ollama + で): the letter model alone finds
    /// such a word as likely as the split. Of several splits, the first
    /// (Ollama + のです, not Ollamano + です); a word it knows (`code`,
    /// `piano`) stays whole.
    fn split_particles(&self, chars: &[char], spans: Vec<Span>, fin: bool) -> Vec<Span> {
        let mut out: Vec<Span> = Vec::with_capacity(spans.len() + 1);
        let mut it = spans.into_iter().peekable();
        while let Some(span) = it.next() {
            if span.lang != Lang::En {
                out.push(span);
                continue;
            }
            // Japanese right after it reads on from the particle.
            let next_ja = it
                .peek()
                .is_some_and(|n| n.lang == Lang::Ja && n.start == span.end);
            let rest_end = if next_ja {
                it.peek().map_or(span.end, |n| n.end)
            } else {
                span.end
            };
            match self.particle_split(chars, span, rest_end, fin) {
                Some(at) => {
                    if next_ja {
                        it.next();
                    }
                    out.push(Span { end: at, ..span });
                    out.push(Span {
                        lang: Lang::Ja,
                        start: at,
                        end: rest_end,
                    });
                }
                None => out.push(span),
            }
        }
        out
    }

    /// Where `span` (English) splits before a particle, Japanese running on
    /// to `rest_end`; see `split_particles`.
    fn particle_split(
        &self,
        chars: &[char],
        span: Span,
        rest_end: usize,
        fin: bool,
    ) -> Option<usize> {
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
        let rest_fin = fin && rest_end == chars.len();
        // at least two letters stay English
        (run_start + 2..span.end.saturating_sub(1)).find(|&at| {
            let rest = &chars[at..rest_end];
            let lower: String = chars[at..span.end]
                .iter()
                .map(|c| c.to_ascii_lowercase())
                .collect();
            if !PARTICLES.iter().any(|p| lower.starts_with(p)) {
                return false;
            }
            // the rest reads as kana, but for romaji still being typed
            let units = romaji_units(rest, rest_fin);
            let reads = units.iter().enumerate().all(|(i, u)| match u.kind {
                UnitKind::Kana(_) => true,
                UnitKind::Pending => !rest_fin && i + 1 == units.len() && rest_end == chars.len(),
                UnitKind::Char(_) => false,
            });
            // and what stays is English (no joiner at its end)
            reads
                && self
                    .en_costs(&chars[span.start..at], false)
                    .last()
                    .is_some_and(|(len, _)| span.start + len == at)
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
                UnitKind::Char(c) if c.is_ascii_alphabetic() => p.leftover,
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
                // Joiners stand alone, but for a URL's `://`.
                if k > 0 && is_joiner(chars[k - 1]) {
                    let url = matches!(&chars[k - 1..], [':', '/', '/', ..])
                        || chars[k - 1] == '/' && c == '/';
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
                continue;
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
        let breaks = p.case_break * piece.breaks as f64;
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
        breaks - prob.max(f64::MIN_POSITIVE).ln()
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

    /// Whether `word` (any case) is a Latin word of the model's text.
    pub fn knows(&self, word: &str) -> bool {
        self.words.contains_key(&word.to_ascii_lowercase())
    }

    /// The frequency of a lowercase word among Latin words, if seen.
    fn known(&self, word: &str) -> Option<f64> {
        let p = &self.params;
        self.words.get(word).map(|&(ja, en)| {
            p.ja_text * ja as f64 / self.ja_total + (1.0 - p.ja_text) * en as f64 / self.en_total
        })
    }
}

/// Whether `candidate` has the Latin words of `reading` as they are, in
/// order (the engine passes them through; a candidate that changed one is a
/// misconversion).
pub fn keeps_latin(reading: &str, candidate: &str) -> bool {
    // a full-width word is the same word (`symbol_width = "auto"`)
    let candidate: String = candidate.chars().map(crate::postprocess::to_half).collect();
    let mut rest = candidate.as_str();
    for word in reading
        .split(|c: char| !(c.is_ascii_alphanumeric() || is_joiner(c)))
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
    /// The letter model's cost of its chars, and its context after them.
    letters: f64,
    ctx: [char; 2],
    digits: bool,
    /// The last two chars, for case breaks: two capitals then lowercase
    /// (`SCmo`, an acronym then romaji), but for a plural `s` at the end
    /// (`PCs`), counted once the piece goes on.
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
        self.last = [self.last[1], c];
        self.digits &= c.is_ascii_digit();
        let lower = c.to_ascii_lowercase();
        self.lower.push(lower);
        let class = letter_class(lower);
        self.letters += model.letters.cost(self.ctx[0], self.ctx[1], class);
        self.ctx = [self.ctx[1], class];
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
/// `fin`, a final `n` is ん and other pending romaji is text.
pub fn render_parts(
    chars: &[char],
    spans: &[Span],
    fin: bool,
    symbols: &HashMap<char, String>,
) -> (String, String) {
    let mut text = String::new();
    let mut pending = String::new();
    let mut pos = 0;
    for span in spans {
        text.extend(&chars[pos..span.start]);
        let part = &chars[span.start..span.end];
        match span.lang {
            Lang::En => text.extend(part),
            Lang::Ja => {
                for unit in romaji_units(part, fin && span.end == chars.len()) {
                    match unit.kind {
                        UnitKind::Kana(kana) => text.push_str(kana),
                        UnitKind::Char(c) if !c.is_ascii_punctuation() => text.push(c),
                        UnitKind::Char(c) if is_decimal_point(part, unit.start) => text.push(c),
                        UnitKind::Char(c) => match symbols.get(&c) {
                            Some(s) => text.push_str(s),
                            None => text.push(match c {
                                ',' => '、',
                                '.' => '。',
                                _ => char::from_u32(c as u32 - 0x21 + 0xff01).unwrap_or(c),
                            }),
                        },
                        UnitKind::Pending if fin => text.extend(&part[unit.start..unit.end]),
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
