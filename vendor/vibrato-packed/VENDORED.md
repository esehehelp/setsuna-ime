# vibrato-packed

[vibrato](https://github.com/daac-tools/vibrato) 0.5.3 (MIT OR Apache-2.0,
`LICENSE-MIT`, `LICENSE-APACHE`) as published on crates.io, renamed and
changed for setsuna-ime's analyzer (`crates/analyzer`), whose dictionary is
loaded into the engine's memory for every login: one with 570,000 words took
61 MB there with upstream's layout, 12 MB with this one, for the same
analyses.

Changes (the rest is upstream's):

- `src/dictionary/lexicon/feature.rs`: the words' features in one string, with
  a byte for each word's length and the offset of every 64th word, instead of
  a `String` per word (whose allocations were most of the memory). A word's
  features are at most 255 bytes.
- `src/dictionary/lexicon/map.rs`: the surfaces in an FST (the `fst` crate)
  instead of a double-array trie (`crawdad`, dropped) with a posting list for
  every surface; a surface's word ids that follow one another are the FST's
  value, others are in the postings. A lexicon sorted by surface has no
  postings.
- `src/dictionary.rs`: its own magic, `VibratoPacked 0.5.3 v1`: upstream's
  dictionaries are not read, nor are these by upstream.
- `Cargo.toml`: the package `vibrato-packed`, the library `vibrato_packed`.

setsuna-ime's dictionaries in this layout carry `"format":
"vibrato-packed-zstd"` in their analyzer manifest; `"vibrato-zstd"` ones are
read with upstream's vibrato.
