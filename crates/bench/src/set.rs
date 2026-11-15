//! The bench's sets: one TSV per category (`sets/<version>/<category>.tsv`),
//! a row `id, context, reading, answer, accepted...`. The context is the text
//! put before the caret first (empty for most rows); the reading is what is
//! typed, in kana as it is typed (the particle は as は), digits and symbols as
//! the keys give them; the answer and any further accepted forms are what a
//! conversion may commit. Lines starting with `#` are comments.

use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};

#[derive(Debug, Clone)]
pub struct Row {
    pub id: String,
    pub category: String,
    pub context: String,
    pub reading: String,
    /// The answer first, then the other accepted forms.
    pub answers: Vec<String>,
}

/// Every row of the sets under `dir` (its `*.tsv`, by file name), or of the
/// one file `dir` names; `only` keeps the categories named.
pub fn load(dir: &Path, only: &[String]) -> Result<Vec<Row>> {
    let files: Vec<PathBuf> = if dir.is_file() {
        vec![dir.to_path_buf()]
    } else {
        let mut f: Vec<PathBuf> = std::fs::read_dir(dir)
            .with_context(|| format!("read {}", dir.display()))?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().is_some_and(|x| x == "tsv"))
            .collect();
        f.sort();
        f
    };
    let mut rows = Vec::new();
    for file in files {
        let category = file
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_default();
        if !only.is_empty() && !only.contains(&category) {
            continue;
        }
        let text =
            std::fs::read_to_string(&file).with_context(|| format!("read {}", file.display()))?;
        for (n, line) in text.lines().enumerate() {
            if line.trim().is_empty() || line.starts_with('#') {
                continue;
            }
            let f: Vec<&str> = line.split('\t').collect();
            if f.len() < 4 || f[0].is_empty() || f[2].is_empty() || f[3].is_empty() {
                bail!(
                    "{}:{}: want id, context, reading, answer",
                    file.display(),
                    n + 1
                );
            }
            rows.push(Row {
                id: f[0].to_string(),
                category: category.clone(),
                context: f[1].to_string(),
                reading: f[2].to_string(),
                answers: f[3..]
                    .iter()
                    .filter(|s| !s.is_empty())
                    .map(|s| s.to_string())
                    .collect(),
            });
        }
    }
    Ok(rows)
}
