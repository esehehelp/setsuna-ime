//! Common settings in Vibrato.
use bincode::config::{self, Fixint, Limit, LittleEndian};

/// The most bytes a dictionary may claim as it is read: a length past it
/// (a damaged or crafted file) is an error, not an allocation of that size.
pub const MAX_DICTIONARY_BYTES: usize = 1 << 30;

/// Gets the common bincode configuration of serialization.
pub const fn bincode_config(
) -> config::Configuration<LittleEndian, Fixint, Limit<MAX_DICTIONARY_BYTES>> {
    config::standard()
        .with_little_endian()
        .with_fixed_int_encoding()
        .with_limit::<MAX_DICTIONARY_BYTES>()
}

/// The maximam length of an input sentence.
///
/// Note that the value must be represented with u16 so that
/// an (exclusive) end position can be represented in 16 bits.
pub const MAX_SENTENCE_LENGTH: usize = usize::MAX;

/// The fixed connection id of BOS/EOS.
pub const BOS_EOS_CONNECTION_ID: u16 = 0;
