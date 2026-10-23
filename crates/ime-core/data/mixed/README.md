# Mixed input model

The model `setsuna_core::mixed` reads (`[frontend] mixed_model` points at
this directory; the Windows installer copies it to
`%LOCALAPPDATA%\setsuna-ime\models\mixed`). It is counts only, with no text:

- `kana.tsv` — `<n-gram>\t<count>`, n = 1..3, of hiragana readings: kana and
  `、。` as they are, a digit as `0`, any other char as `#`, an English word as
  `@`, `^` and `$` around each sentence. From 6,000,000 Japanese sentences of
  FineWeb2-edu-ja without Latin letters (a sentence with two symbols in a row
  left out), read by Sudachi (SudachiDict full, split mode C), and 60,000 of
  its sentences with English words in them; n-grams seen once are left out.
- `latin.tsv` — `<word, lowercased>\t<count in Japanese text>\t<count in
  English text>`: the Latin words of four FineWeb2-edu-ja shards and of a
  FineWeb sample (English), kept from 3 / 5 occurrences. Left out: web noise
  that identifies something (a run of five digits or more, twelve hex digits
  or more, more than 30 chars: ad ids, hashes, dated file names) and slurs.
- `params.txt` — the segmenter's weights (`MixedParams`), tuned on held-apart
  mixed sentences and on Japanese ones typed with a slip of the fingers (which
  should not read as English).

## Sources

- [FineWeb2-edu-japanese](https://huggingface.co/datasets/hotchpotch/fineweb-2-edu-japanese)
  (derived from [FineWeb-2](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2)),
  ODC-By 1.0; its use is also subject to the Common Crawl Terms of Use.
- [FineWeb](https://huggingface.co/datasets/HuggingFaceFW/fineweb) (`sample/10BT`),
  ODC-By 1.0; Common Crawl Terms of Use.
- Readings by [Sudachi](https://github.com/WorksApplications/sudachi.rs) with
  [SudachiDict](https://github.com/WorksApplications/SudachiDict) (Apache-2.0).

These counts are a database derived from the datasets above and are
distributed under the [Open Data Commons Attribution License (ODC-By)
v1.0](https://opendatacommons.org/licenses/by/1-0/).
