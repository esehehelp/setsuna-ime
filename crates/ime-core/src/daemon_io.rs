//! What the clients of the helper daemons (engine, reranker, rewriter) share
//! in reading their JSON lines and in holding their state across threads.

use std::io::{self, BufRead, Read};
use std::sync::{Mutex, MutexGuard, PoisonError};

/// The longest line read from a daemon: a longer one (a daemon gone wrong)
/// is skipped, not held whole in memory.
pub(crate) const MAX_LINE: usize = 16 << 20;

/// A line read from a daemon.
#[derive(Debug, PartialEq)]
pub(crate) enum Line {
    /// The daemon's output ended.
    Eof,
    /// A line, in the buffer given.
    Read,
    /// A line over [`MAX_LINE`] bytes, skipped to its end.
    TooLong,
}

/// Read a line into `line` (cleared first), at most `MAX_LINE` bytes of it.
pub(crate) fn read_line(reader: &mut impl BufRead, line: &mut String) -> io::Result<Line> {
    read_line_max(reader, line, MAX_LINE)
}

fn read_line_max(reader: &mut impl BufRead, line: &mut String, max: usize) -> io::Result<Line> {
    line.clear();
    let mut buf = Vec::new();
    let n = reader
        .by_ref()
        .take(max as u64 + 1)
        .read_until(b'\n', &mut buf)?;
    if n == 0 {
        return Ok(Line::Eof);
    }
    if buf.len() <= max || buf.last() == Some(&b'\n') {
        *line =
            String::from_utf8(buf).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        return Ok(Line::Read);
    }
    // the rest of the line, a chunk at a time
    loop {
        buf.clear();
        let n = reader.by_ref().take(64 << 10).read_until(b'\n', &mut buf)?;
        if n == 0 || buf.last() == Some(&b'\n') {
            return Ok(Line::TooLong);
        }
    }
}

/// Whether `line` is a daemon's ready line (`{"ready":true,...}`): anything
/// else it prints first (a library's banner) is not.
pub(crate) fn is_ready(line: &str) -> bool {
    serde_json::from_str::<serde_json::Value>(line.trim())
        .is_ok_and(|v| v.get("ready") == Some(&serde_json::Value::Bool(true)))
}

/// Lock `m`, whatever a thread that panicked holding it left: the state is
/// plain data a panic leaves whole, and the frontend's thread must not
/// panic over another's.
pub(crate) fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(PoisonError::into_inner)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_line_too_long_is_skipped_and_the_next_one_read() {
        let text = format!("{}\nnext\n{}", "x".repeat(100), "y".repeat(10));
        let mut r = io::BufReader::new(text.as_bytes());
        let mut line = String::new();
        assert_eq!(read_line_max(&mut r, &mut line, 50).unwrap(), Line::TooLong);
        assert_eq!(read_line_max(&mut r, &mut line, 50).unwrap(), Line::Read);
        assert_eq!(line, "next\n");
        assert_eq!(read_line_max(&mut r, &mut line, 50).unwrap(), Line::Read);
        assert_eq!(line, "y".repeat(10));
        assert_eq!(read_line_max(&mut r, &mut line, 50).unwrap(), Line::Eof);
        // a line of exactly the limit, its newline past it
        let text = format!("{}\n", "z".repeat(50));
        let mut r = io::BufReader::new(text.as_bytes());
        assert_eq!(read_line_max(&mut r, &mut line, 50).unwrap(), Line::Read);
    }

    #[test]
    fn only_a_ready_line_is_one() {
        assert!(is_ready("{\"ready\":true,\"version\":\"1\"}\n"));
        assert!(!is_ready("OpenMP: 16 threads\n"));
        assert!(!is_ready("{\"ready\":false}"));
        assert!(!is_ready("{\"id\":1}"));
    }
}
