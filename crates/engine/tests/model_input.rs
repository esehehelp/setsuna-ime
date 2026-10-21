//! The model input contract (tests/fixtures/model_input, written by the
//! training side's encoder): the engine lays the input out as the training
//! rows are laid out.

use std::path::{Path, PathBuf};

use setsuna_engine::pipeline::{proposal, symtoken};
use setsuna_engine::SharedCharTokenizer;

fn fixture() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/model_input")
}

fn check(tokenizer: &SharedCharTokenizer, key: &str) {
    let cases: serde_json::Value =
        serde_json::from_slice(&std::fs::read(fixture().join("cases.json")).unwrap()).unwrap();
    for case in cases["cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let reading = case["reading"].as_str().unwrap();
        let (reading, runs) = if case["symbol_token"].as_bool().unwrap() {
            let t = symtoken::tokenize(reading);
            (t.text, t.runs)
        } else {
            (reading.to_string(), Vec::new())
        };
        let seq_len = case["max_seq_len"].as_u64().unwrap() as usize;
        let encoded = proposal::encode(
            tokenizer,
            seq_len,
            case["max_context"].as_u64().unwrap() as usize,
            case["context"].as_str().unwrap(),
            &reading,
        );
        if name.contains("longer than max_seq_len") {
            // the session splits such a reading before encoding (chunk)
            assert!(encoded.is_err(), "{name}");
            continue;
        }
        let (ids, _, actual) = encoded.unwrap_or_else(|e| panic!("{name}: {e}"));
        let want: Vec<i64> = case[key]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_i64().unwrap())
            .collect();
        assert_eq!(&ids[..actual], &want[..], "{name} ({key})");
        let want_runs: Vec<String> = case["runs"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap().to_string())
            .collect();
        assert_eq!(runs, want_runs, "{name} runs");
    }
}

#[test]
fn input_as_training_with_the_input_vocab() {
    let tokenizer = SharedCharTokenizer::load_vocab_hex_tsv(&fixture().join("model.vocab.hex.tsv"))
        .unwrap()
        .with_input_vocab(&fixture().join("shard.vocab.hex.tsv"))
        .unwrap();
    check(&tokenizer, "input_ids");
}

#[test]
fn input_as_training_without_an_input_vocab() {
    let tokenizer =
        SharedCharTokenizer::load_vocab_hex_tsv(&fixture().join("model.vocab.hex.tsv")).unwrap();
    check(&tokenizer, "input_ids_without_shard_vocab");
}
