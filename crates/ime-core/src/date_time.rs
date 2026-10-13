//! Dynamic date/time candidates, entirely on the IME side. Only a whole
//! reading (or a focused segment's reading) is recognized. The caller owns
//! the clock snapshot, so a delayed model/rerank result cannot change the
//! date or time of an open candidate list.
//!
//! Bare weekdays mean the next occurrence, including today. Weeks start
//! on Monday; らいしゅう offers a Monday-Sunday range and its starting date.
//! Minutes/hours/seconds are elapsed time, days/weeks/months/years are
//! calendar dates (a month/year shift clamps to the target month's end).

use std::collections::HashSet;
use std::fmt::Display;

use chrono::{
    DateTime, Datelike, Days, Duration, Months, NaiveDate, Offset, TimeZone, Timelike, Weekday,
};
use unicode_normalization::UnicodeNormalization;

#[derive(Debug, Clone, Copy)]
enum Unit {
    Second,
    Minute,
    Hour,
    Day,
    Week,
    Month,
    Year,
}

#[derive(Debug, Clone, Copy)]
enum Request {
    Date(i64),
    Week(i64),
    Month(i32),
    Year(i32),
    Weekday(Weekday, Option<i64>),
    Now,
    Relative(u32, Unit, bool),
}

fn parse(reading: &str) -> Option<Request> {
    let reading: String = reading
        .nfkc()
        .map(|c| match c {
            'ヶ' | 'ヵ' => 'か',
            '\u{30A1}'..='\u{30F6}' => char::from_u32(c as u32 - 0x60).unwrap_or(c),
            c => c,
        })
        .collect();
    let s = reading.as_str();
    let request = match s {
        "きょう" | "今日" | "ほんじつ" | "本日" | "ひづけ" | "日付" => {
            Request::Date(0)
        }
        "きのう" | "昨日" | "さくじつ" => Request::Date(-1),
        "あした" | "あす" | "明日" => Request::Date(1),
        "あさって" | "明後日" => Request::Date(2),
        "おととい" | "一昨日" => Request::Date(-2),
        "こんしゅう" | "今週" => Request::Week(0),
        "らいしゅう" | "来週" => Request::Week(1),
        "さらいしゅう" | "再来週" => Request::Week(2),
        "せんしゅう" | "先週" => Request::Week(-1),
        "こんげつ" | "今月" => Request::Month(0),
        "らいげつ" | "来月" => Request::Month(1),
        "さらいげつ" | "再来月" => Request::Month(2),
        "せんげつ" | "先月" => Request::Month(-1),
        "ことし" | "今年" => Request::Year(0),
        "らいねん" | "来年" => Request::Year(1),
        "さらいねん" | "再来年" => Request::Year(2),
        "きょねん" | "去年" => Request::Year(-1),
        "いま" | "今" | "じこく" | "時刻" | "げんざいじこく" | "現在時刻" | "にちじ" | "日時" => {
            Request::Now
        }
        _ => return parse_weekday(s).or_else(|| parse_relative(s)),
    };
    Some(request)
}

fn parse_weekday(s: &str) -> Option<Request> {
    let mut name = s;
    let mut week = None;
    for (prefix, offset) in [
        ("さらいしゅう", 2),
        ("再来週", 2),
        ("らいしゅう", 1),
        ("来週", 1),
        ("こんしゅう", 0),
        ("今週", 0),
        ("せんしゅう", -1),
        ("先週", -1),
    ] {
        if let Some(rest) = s.strip_prefix(prefix) {
            name = rest.strip_prefix('の').unwrap_or(rest);
            week = Some(offset);
            break;
        }
    }
    let day = match name {
        "げつようび" | "げつよう" | "月曜日" | "月曜" => Weekday::Mon,
        "かようび" | "かよう" | "火曜日" | "火曜" => Weekday::Tue,
        "すいようび" | "すいよう" | "水曜日" | "水曜" => Weekday::Wed,
        "もくようび" | "もくよう" | "木曜日" | "木曜" => Weekday::Thu,
        "きんようび" | "きんよう" | "金曜日" | "金曜" => Weekday::Fri,
        "どようび" | "どよう" | "土曜日" | "土曜" => Weekday::Sat,
        "にちようび" | "にちよう" | "日曜日" | "日曜" => Weekday::Sun,
        _ => return None,
    };
    Some(Request::Weekday(day, week))
}

fn parse_relative(s: &str) -> Option<Request> {
    let (s, past) = if let Some(s) = s.strip_suffix('後').or_else(|| s.strip_suffix("ご")) {
        (s, false)
    } else {
        (
            s.strip_suffix('前').or_else(|| s.strip_suffix("まえ"))?,
            true,
        )
    };
    for (name, unit) in [
        ("しゅうかん", Unit::Week),
        ("週間", Unit::Week),
        ("じかん", Unit::Hour),
        ("時間", Unit::Hour),
        ("かげつ", Unit::Month),
        ("か月", Unit::Month),
        ("ヶ月", Unit::Month),
        ("箇月", Unit::Month),
        ("びょう", Unit::Second),
        ("秒", Unit::Second),
        ("ふん", Unit::Minute),
        ("ぷん", Unit::Minute),
        ("分", Unit::Minute),
        ("にち", Unit::Day),
        ("日", Unit::Day),
        ("しゅう", Unit::Week),
        ("週", Unit::Week),
        ("ねん", Unit::Year),
        ("年", Unit::Year),
    ] {
        if let Some(n) = s.strip_suffix(name).and_then(number) {
            return Some(Request::Relative(n, unit, past));
        }
    }
    None
}

fn number(s: &str) -> Option<u32> {
    if s.is_empty() {
        return None;
    }
    if s.bytes().all(|b| b.is_ascii_digit()) {
        return s.parse().ok();
    }
    // Kana quantities up to 999, including contracted minute/month readings.
    let s = s.replace("じゅっ", "じゅう").replace("じっ", "じゅう");
    let digit = |s: &str| match s {
        "" | "ぜろ" | "れい" => Some(0),
        "いち" | "いっ" => Some(1),
        "に" => Some(2),
        "さん" => Some(3),
        "よん" | "し" => Some(4),
        "ご" => Some(5),
        "ろく" | "ろっ" => Some(6),
        "なな" | "しち" => Some(7),
        "はち" | "はっ" => Some(8),
        "きゅう" | "く" => Some(9),
        _ => None,
    };
    let below_hundred = |s: &str| {
        if let Some((a, b)) = s.split_once("じゅう") {
            Some(if a.is_empty() { 1 } else { digit(a)? } * 10 + digit(b)?)
        } else {
            digit(s)
        }
    };
    for (prefix, n) in [
        ("きゅうひゃく", 900),
        ("ななひゃく", 700),
        ("ろっぴゃく", 600),
        ("はっぴゃく", 800),
        ("よんひゃく", 400),
        ("さんびゃく", 300),
        ("ごひゃく", 500),
        ("にひゃく", 200),
        ("ひゃく", 100),
    ] {
        if let Some(rest) = s.strip_prefix(prefix) {
            return Some(n + below_hundred(rest)?);
        }
    }
    if let Some(n) = below_hundred(&s) {
        return Some(n);
    }
    // Kanji quantities, up to 9999 (or a digit-only spelling like 二〇二六).
    let kanji_digit = |c| {
        "〇一二三四五六七八九"
            .chars()
            .position(|d| d == c)
            .map(|i| i as u32)
            .or_else(|| (c == '零').then_some(0))
    };
    if s.chars().all(|c| kanji_digit(c).is_some()) {
        return s
            .chars()
            .try_fold(0u32, |n, c| n.checked_mul(10)?.checked_add(kanji_digit(c)?));
    }
    let (mut total, mut digit, mut last_unit) = (0u32, None, 10000);
    for c in s.chars() {
        if let Some(n) = kanji_digit(c) {
            if digit.replace(n).is_some() {
                return None;
            }
        } else {
            let unit = match c {
                '十' => 10,
                '百' => 100,
                '千' => 1000,
                _ => return None,
            };
            if unit >= last_unit {
                return None;
            }
            total = total.checked_add(digit.take().unwrap_or(1) * unit)?;
            last_unit = unit;
        }
    }
    total.checked_add(digit.unwrap_or(0))
}

/// Whether this whole reading names a dynamic date/time. Does not read the clock.
pub fn is_reading(reading: &str) -> bool {
    parse(reading).is_some()
}

/// Generate candidates at the supplied clock snapshot (local time in production,
/// fixed-offset time in tests). Invalid/overflowing requests produce none.
pub fn candidates<Tz: TimeZone>(reading: &str, now: &DateTime<Tz>) -> Vec<String>
where
    Tz::Offset: Display,
{
    let Some(request) = parse(reading) else {
        return Vec::new();
    };
    let Some(local) = now
        .naive_utc()
        .checked_add_signed(Duration::seconds(i64::from(
            now.offset().fix().local_minus_utc(),
        )))
    else {
        return Vec::new();
    };
    let today = local.date();
    let shift = |date: NaiveDate, days| date.checked_add_signed(Duration::try_days(days)?);
    let mut out = match request {
        Request::Date(days) => shift(today, days).map(date_candidates).unwrap_or_default(),
        Request::Week(weeks) => {
            let monday = shift(
                today,
                weeks * 7 - i64::from(today.weekday().num_days_from_monday()),
            );
            monday
                .and_then(|a| Some(range_candidates(a, a.checked_add_days(Days::new(6))?)))
                .unwrap_or_default()
        }
        Request::Month(months) => {
            let first = today.with_day(1).and_then(|d| shift_month(d, months));
            first
                .and_then(|a| {
                    Some(range_candidates(
                        a,
                        a.checked_add_months(Months::new(1))?.pred_opt()?,
                    ))
                })
                .unwrap_or_default()
        }
        Request::Year(years) => today
            .year()
            .checked_add(years)
            .map(|y| vec![format!("{y}年"), y.to_string()])
            .unwrap_or_default(),
        Request::Weekday(day, week) => {
            let weekday = i64::from(today.weekday().num_days_from_monday());
            let target = i64::from(day.num_days_from_monday());
            let days = week.map_or((target - weekday).rem_euclid(7), |w| {
                w * 7 + target - weekday
            });
            shift(today, days).map(date_candidates).unwrap_or_default()
        }
        Request::Now => time_candidates(now, false),
        Request::Relative(n, unit, past) => {
            let signed = if past { -i64::from(n) } else { i64::from(n) };
            match unit {
                Unit::Second | Unit::Minute | Unit::Hour => {
                    let seconds = signed
                        * match unit {
                            Unit::Second => 1,
                            Unit::Minute => 60,
                            _ => 3600,
                        };
                    Duration::try_seconds(seconds)
                        .and_then(|d| now.clone().checked_add_signed(d))
                        .map(|d| time_candidates(&d, true))
                        .unwrap_or_default()
                }
                Unit::Day | Unit::Week => shift(
                    today,
                    signed * if matches!(unit, Unit::Week) { 7 } else { 1 },
                )
                .map(date_candidates)
                .unwrap_or_default(),
                Unit::Month | Unit::Year => {
                    let months = n.checked_mul(if matches!(unit, Unit::Year) { 12 } else { 1 });
                    months
                        .and_then(|n| {
                            if past {
                                today.checked_sub_months(Months::new(n))
                            } else {
                                today.checked_add_months(Months::new(n))
                            }
                        })
                        .map(date_candidates)
                        .unwrap_or_default()
                }
            }
        }
    };
    let mut seen = HashSet::new();
    out.retain(|c| seen.insert(c.clone()));
    out
}

fn shift_month(date: NaiveDate, months: i32) -> Option<NaiveDate> {
    if months < 0 {
        date.checked_sub_months(Months::new(months.unsigned_abs()))
    } else {
        date.checked_add_months(Months::new(months as u32))
    }
}

fn date_candidates(date: NaiveDate) -> Vec<String> {
    let weekday =
        ["月", "火", "水", "木", "金", "土", "日"][date.weekday().num_days_from_monday() as usize];
    vec![
        date.format("%Y/%m/%d").to_string(),
        date.format("%Y-%m-%d").to_string(),
        format!("{}年{}月{}日", date.year(), date.month(), date.day()),
        format!("{}月{}日", date.month(), date.day()),
        format!("{}月{}日（{weekday}）", date.month(), date.day()),
    ]
}

fn range_candidates(start: NaiveDate, end: NaiveDate) -> Vec<String> {
    let mut out = vec![
        format!("{}〜{}", start.format("%Y/%m/%d"), end.format("%Y/%m/%d")),
        format!(
            "{}月{}日〜{}月{}日",
            start.month(),
            start.day(),
            end.month(),
            end.day()
        ),
    ];
    out.extend(date_candidates(start));
    out
}

fn time_candidates<Tz: TimeZone>(time: &DateTime<Tz>, date_first: bool) -> Vec<String>
where
    Tz::Offset: Display,
{
    // A UTC value can fit while its local wall time is outside chrono's
    // range. Avoid panicking when formatting an extreme relative request.
    if time
        .naive_utc()
        .checked_add_signed(Duration::seconds(i64::from(
            time.offset().fix().local_minus_utc(),
        )))
        .is_none()
    {
        return Vec::new();
    }
    let (hour12, pm) = (
        if time.hour() % 12 == 0 {
            12
        } else {
            time.hour() % 12
        },
        time.hour() >= 12,
    );
    let mut out = vec![
        time.format("%H:%M").to_string(),
        time.format("%H:%M:%S").to_string(),
        format!("{}時{}分", time.hour(), time.minute()),
        format!(
            "{}{}時{}分",
            if pm { "午後" } else { "午前" },
            hour12,
            time.minute()
        ),
        time.format("%Y/%m/%d %H:%M").to_string(),
        time.to_rfc3339_opts(chrono::SecondsFormat::Secs, false),
    ];
    if date_first {
        let c = out.remove(4);
        out.insert(0, c);
    }
    out
}

/// Add generated candidates after the reading's kana entries (normally after
/// the first two model candidates). Keep ordinary conversions and deduplicate.
pub fn insert(reading: &str, candidates: Vec<String>, generated: Vec<String>) -> Vec<String> {
    if generated.is_empty() {
        return candidates;
    }
    let mut out: Vec<String> = candidates
        .into_iter()
        .filter(|c| !generated.contains(c))
        .collect();
    let kana = [
        reading.to_string(),
        crate::kana_candidates::to_katakana(reading),
    ];
    let at = out
        .iter()
        .rposition(|c| kana.contains(c))
        .map_or(out.len().min(2), |i| i + 1);
    out.splice(at..at, generated);
    let mut seen = HashSet::new();
    out.retain(|c| seen.insert(c.clone()));
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::FixedOffset;

    fn now() -> DateTime<FixedOffset> {
        FixedOffset::east_opt(9 * 3600)
            .unwrap()
            .with_ymd_and_hms(2026, 10, 6, 14, 5, 9)
            .unwrap()
    }

    #[test]
    fn named_dates_use_the_clock_date() {
        for (reading, date) in [
            ("きのう", "2026/10/05"),
            ("きょう", "2026/10/06"),
            ("あした", "2026/10/07"),
            ("あさって", "2026/10/08"),
            ("おととい", "2026/10/04"),
        ] {
            assert_eq!(candidates(reading, &now())[0], date);
        }
        assert!(candidates("今日", &now()).contains(&"10月6日（火）".into()));
        assert_eq!(candidates("キョウ", &now()), candidates("きょう", &now()));
    }

    #[test]
    fn weekdays_include_today_and_explicit_weeks_start_on_monday() {
        for (reading, date) in [
            ("かようび", "2026/10/06"),
            ("すいようび", "2026/10/07"),
            ("げつようび", "2026/10/12"),
            ("にちようび", "2026/10/11"),
            ("らいしゅうのすいようび", "2026/10/14"),
            ("今週月曜日", "2026/10/05"),
            ("先週の水曜日", "2026/09/30"),
        ] {
            assert_eq!(candidates(reading, &now())[0], date, "{reading}");
        }
        assert_eq!(
            candidates("らいしゅう", &now())[0],
            "2026/10/12〜2026/10/18"
        );
        let sunday = now().with_day(11).unwrap();
        assert_eq!(
            candidates("らいしゅう", &sunday)[0],
            "2026/10/12〜2026/10/18"
        );
    }

    #[test]
    fn times_have_24_hour_japanese_and_offset_forms() {
        let c = candidates("いま", &now());
        assert_eq!(&c[..4], ["14:05", "14:05:09", "14時5分", "午後2時5分"]);
        assert!(c.contains(&"2026-10-06T14:05:09+09:00".into()));
        assert!(candidates("いま", &now().with_hour(0).unwrap()).contains(&"午前12時5分".into()));
        assert!(candidates("いま", &now().with_hour(12).unwrap()).contains(&"午後12時5分".into()));
    }

    #[test]
    fn relative_times_accept_numeric_and_kana_readings() {
        for s in [
            "1時間後",
            "1じかんご",
            "１時間後",
            "いちじかんご",
            "一時間後",
        ] {
            assert_eq!(candidates(s, &now())[0], "2026/10/06 15:05", "{s}");
        }
        assert_eq!(candidates("30ぷんまえ", &now())[0], "2026/10/06 13:35");
        assert_eq!(candidates("いっぷんご", &now())[0], "2026/10/06 14:06");
        assert_eq!(candidates("にじゅっぷんご", &now())[0], "2026/10/06 14:25");
        assert_eq!(candidates("2しゅうかんご", &now())[0], "2026/10/20");
        assert_eq!(candidates("1にちまえ", &now())[0], "2026/10/05");
        assert!(candidates("1びょうご", &now()).contains(&"14:05:10".into()));
        assert_eq!(candidates("0じかんご", &now())[0], "2026/10/06 14:05");
    }

    #[test]
    fn calendar_and_midnight_boundaries() {
        let new_year = now()
            .with_month(12)
            .unwrap()
            .with_day(31)
            .unwrap()
            .with_hour(23)
            .unwrap();
        assert_eq!(candidates("1時間後", &new_year)[0], "2027/01/01 00:05");
        let leap = now()
            .with_year(2024)
            .unwrap()
            .with_month(2)
            .unwrap()
            .with_day(28)
            .unwrap();
        assert_eq!(candidates("あした", &leap)[0], "2024/02/29");
        assert_eq!(candidates("あさって", &leap)[0], "2024/03/01");
        assert_eq!(
            candidates("1ねんご", &leap.with_day(29).unwrap())[0],
            "2025/02/28"
        );
        let january = now().with_month(1).unwrap().with_day(31).unwrap();
        for s in ["1かげつご", "1か月後", "1ヶ月後", "1ヵ月後", "いっかげつご"] {
            assert_eq!(candidates(s, &january)[0], "2026/02/28", "{s}");
        }
        assert_eq!(
            candidates("らいげつ", &new_year)[0],
            "2027/01/01〜2027/01/31"
        );
        assert_eq!(candidates("らいねん", &new_year), ["2027年", "2027"]);
    }

    #[test]
    fn unknown_partial_or_overflowing_readings_are_left_alone() {
        for s in [
            "",
            "きょ",
            "きょうは",
            "いまから",
            "すいようびに",
            "1じかんごに",
            "-1時間後",
            "1.5時間後",
            "時間後",
            "999999999999999999時間後",
            "4294967295年後",
        ] {
            assert!(candidates(s, &now()).is_empty(), "{s}");
        }
        assert!(candidates(
            "あした",
            &DateTime::<FixedOffset>::from_naive_utc_and_offset(
                NaiveDate::MAX.and_hms_opt(0, 0, 0).unwrap(),
                FixedOffset::east_opt(0).unwrap()
            )
        )
        .is_empty());
    }

    #[test]
    fn local_wall_time_boundaries_do_not_panic() {
        let offset = FixedOffset::east_opt(9 * 3600).unwrap();
        let last = DateTime::<FixedOffset>::from_naive_utc_and_offset(
            NaiveDate::MAX.and_hms_opt(14, 0, 0).unwrap(),
            offset,
        );
        assert!(candidates("1じかんご", &last).is_empty());
        let outside = DateTime::<FixedOffset>::from_naive_utc_and_offset(
            NaiveDate::MAX.and_hms_opt(23, 0, 0).unwrap(),
            offset,
        );
        assert!(candidates("いま", &outside).is_empty());
        // Local dates need not be the UTC date.
        let tokyo = now().with_hour(0).unwrap();
        assert_eq!(candidates("きょう", &tokyo)[0], "2026/10/06");
    }

    #[test]
    fn quantities_and_insertion_are_bounded_and_deduplicated() {
        for (s, n) in [
            ("じゅう", 10),
            ("にじゅうさん", 23),
            ("ひゃくにじゅう", 120),
            ("さんびゃく", 300),
            ("三百二十一", 321),
            ("二〇二六", 2026),
        ] {
            assert_eq!(number(s), Some(n));
        }
        let date = candidates("きょう", &now());
        let list = vec![
            "今日".into(),
            "京".into(),
            "きょう".into(),
            "キョウ".into(),
            date[0].clone(),
        ];
        let c = insert("きょう", list, date.clone());
        assert_eq!(&c[..4], ["今日", "京", "きょう", "キョウ"]);
        assert_eq!(&c[4..], date);
        assert_eq!(HashSet::<_>::from_iter(c.iter()).len(), c.len());
    }
}
