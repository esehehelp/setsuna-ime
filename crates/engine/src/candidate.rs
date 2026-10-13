//! A conversion candidate and where it came from.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CandidateSource {
    /// A prefix-beam hypothesis.
    Beam,
    /// The per-frame argmax collapse (beam width 1, or the empty-beam fallback).
    Greedy,
    /// The reading itself (or its katakana form) when decoding produced nothing.
    Fallback,
}

impl CandidateSource {
    pub fn as_str(self) -> &'static str {
        match self {
            CandidateSource::Beam => "beam",
            CandidateSource::Greedy => "greedy",
            CandidateSource::Fallback => "fallback",
        }
    }
}

/// One candidate: its text and the CTC log-probability mass of its token
/// sequence (0 for greedy and fallback candidates, which carry no score).
#[derive(Debug, Clone, PartialEq)]
pub struct Candidate {
    pub text: String,
    pub score: f32,
    pub source: CandidateSource,
    /// For each char of `text`, the index of the reading char its token was
    /// emitted at (the CTC alignment). Empty when unknown.
    pub reading_pos: Vec<u32>,
}

/// `0..n` for a text that is the reading itself (or a char-for-char form of
/// it, such as its katakana).
pub fn identity_pos(text: &str) -> Vec<u32> {
    (0..text.chars().count() as u32).collect()
}
