//! Opt-in runtime regressions with an identity CTC graph (reading token ->
//! output token; context/special frames emit blank), a byte-fallback
//! vocabulary and max_seq_len=128. Set
//! SETSUNA_TEST_IDENTITY_MANIFEST and SETSUNA_TEST_ORT_DYLIB, then run
//! `cargo test -p setsuna-engine --test runtime_regressions -- --ignored`.

use std::path::Path;

use setsuna_artifact::{CtcNatManifest, DecodeDefaults};
use setsuna_engine::{ConversionRequest, EngineSession};

fn session() -> EngineSession {
    let ort = std::env::var("SETSUNA_TEST_ORT_DYLIB").expect("set SETSUNA_TEST_ORT_DYLIB");
    setsuna_engine::runtime::init(Some(Path::new(&ort))).unwrap();
    let path = std::env::var("SETSUNA_TEST_IDENTITY_MANIFEST")
        .expect("set SETSUNA_TEST_IDENTITY_MANIFEST");
    let manifest = CtcNatManifest::load(Path::new(&path)).unwrap();
    EngineSession::from_manifest(
        &manifest,
        &DecodeDefaults {
            beam_width: Some(1),
            symbol_split: Some(false),
            ..Default::default()
        },
    )
    .unwrap()
}

#[test]
#[ignore = "requires ONNX Runtime and an identity CTC artifact"]
fn greedy_inspection_survives_cache_mode_changes() {
    let mut session = session();
    for inspect in [0, 5, 20, 0, 5] {
        session.inspect = inspect;
        let result = session
            .convert(ConversionRequest {
                context: "",
                reading: "ゕゖ",
                continue_beam: 0,
            })
            .unwrap();
        assert_eq!(result.candidates[0].text, "ゕゖ");
        assert_eq!(result.candidates[0].score, 0.0);
        if inspect == 0 {
            assert!(result.frames.is_none());
        } else {
            let frames = result.frames.unwrap();
            assert_eq!(frames.len(), result.perf.input_len);
            assert!(frames.iter().all(|f| f.top.len() == inspect));
        }
    }
}

#[test]
#[ignore = "requires ONNX Runtime and an identity CTC artifact"]
fn byte_fallback_readings_are_never_truncated_or_misaligned() {
    let mut session = session();
    for reading in ["ゕゖ".to_string(), "ゕゖ".repeat(30)] {
        let result = session
            .convert(ConversionRequest {
                context: "",
                reading: &reading,
                continue_beam: 0,
            })
            .unwrap();
        assert_eq!(result.candidates[0].text, reading);
        assert_eq!(
            result.candidates[0].reading_pos,
            (0..reading.chars().count() as u32).collect::<Vec<_>>()
        );
    }
}
