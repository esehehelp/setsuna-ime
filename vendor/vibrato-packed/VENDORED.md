# vibrato-packed

[vibrato](https://github.com/daac-tools/vibrato) 0.5.3 (MIT OR Apache-2.0,
`LICENSE-MIT`, `LICENSE-APACHE`) as published on crates.io, renamed and
changed for setsuna-ime's analyzer (`crates/analyzer`), whose dictionary is
loaded into the engine's memory for every login: one with 570,000 words took
61 MB there with upstream's layout, 12 MB with this one, for the same
analyses.

Changes (the rest is upstream's):

- `src/dictionary/lexicon/feature.rs`: the words' features in one string, with
  the offset (u32) of every 64th word and each word's end from its block's
  offset in two bytes (u16), instead of a `String` per word (whose
  allocations were most of the memory). The features of 64 words in a row
  are at most 64 KiB, and all of them at most 4 GiB.
- `src/dictionary/lexicon/map.rs`: the surfaces in an FST (the `fst` crate)
  instead of a double-array trie (`crawdad`, dropped) with a posting list for
  every surface; a surface's word ids that follow one another are the FST's
  value, others are in the postings. A lexicon sorted by surface has no
  postings. `common_prefix_iterator` walks the FST lazily without allocating a
  per-position `Vec`.
- `src/dictionary.rs`: its own magic, `VibratoPacked 0.5.3 v1`: upstream's
  dictionaries are not read, nor are these by upstream. A dictionary read is
  refused when a word's or an unknown word's connection ids are past its
  connector (upstream checks that only as it builds one).
- `src/common.rs`: bincode reads with a limit (`MAX_DICTIONARY_BYTES`), so
  that a length in a damaged or crafted file is an error, not an
  allocation of that size.
- `src/dictionary/connector/raw_connector/scorer.rs`: decoding refuses a base
  past 31 bits and a table past `i32::MAX` entries, which the AVX2 path's
  signed compares and gathers take for granted (upstream reads before the
  table with such a base, or panics).
- `Cargo.toml`: the package `vibrato-packed`, the library `vibrato_packed`,
  and `default = []` so the optional `train` feature (`rucrf`) is not pulled
  into inference builds by default.

setsuna-ime's dictionaries in this layout carry `"format":
"vibrato-packed-zstd"` in their analyzer manifest; `"vibrato-zstd"` ones are
read with upstream's vibrato.
