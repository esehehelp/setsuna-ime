//! A conversion candidate and where it came from.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CandidateSource {
    /// A prefix-beam hypothesis.
    Beam,
    /// The per-frame argmax collapse (beam width 1, or the empty-beam fallback).
    Greedy,
    /// The reading itself (or its katakana form) when decoding produced nothing.
    Fallback,
    /// A candidate of the other of the two readings a reading corrector leaves (`pipeline::correct::merge`): the
    /// reading as typed when the corrected one was converted first, else the corrected one.
    OtherReading,
}

impl CandidateSource {
    pub fn as_str(self) -> &'static str {
        match self {
            CandidateSource::Beam => "beam",
            CandidateSource::Greedy => "greedy",
            CandidateSource::Fallback => "fallback",
            CandidateSource::OtherReading => "other_reading",
        }
    }
}

/// One candidate: its text and the CTC log-probability mass of its token
/// sequence (0 for greedy and fallback candidates, which carry no score);
/// an AR model's log-probability of its tokens and the end for an AR model,
/// or with the fusion the two models' scores combined (`parts`).
#[derive(Debug, Clone, PartialEq)]
pub struct Candidate {
    pub text: String,
    pub score: f32,
    pub source: CandidateSource,
    /// For each char of `text`, the index of the reading char its token was
    /// emitted at (the CTC alignment). Empty when unknown.
    pub reading_pos: Vec<u32>,
    /// The two models' log-probabilities `score` combines, for a candidate of
    /// an AR model scored by a CTC model too (the session's fusion); `None`
    /// otherwise. A candidate spliced together from several spans keeps the
    /// score of the span it varies, and its parts with it.
    pub parts: Option<ScoreParts>,
}

/// A fused candidate's scores: the AR model's log-probability of its tokens
/// and the end, and the CTC model's of its tokens summed over every
/// alignment (`pipeline::ctc_score`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScoreParts {
    pub ar: f32,
    pub ctc: f32,
}

/// `0..n` for a text that is the reading itself (or a char-for-char form of
/// it, such as its katakana).
pub fn identity_pos(text: &str) -> Vec<u32> {
    (0..text.chars().count() as u32).collect()
}
