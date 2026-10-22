//! Setsuna demo: the IME in a browser.
//!
//! A small local server: `GET /` serves the demo page (`web/index.html`,
//! compiled in), `GET /ws` upgrades to a WebSocket on which the page sends
//! its key presses. Each connection runs the fcitx5 frontend's `Session`
//! with its own `setsuna-engine daemon`, so the page behaves exactly as the
//! IME does; the replies carry the session's view plus the engine's
//! timings, scores and alignment for the page to show off.
//!
//!     cargo run --release -p setsuna-demo -- [--port 8765]
//!
//! The model, engine and ONNX Runtime come from `config.toml`, as for the
//! IME.

use std::io::{ErrorKind, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

use serde_json::{json, Value};
use setsuna_core::child_daemon;
use setsuna_core::config::Config;
use setsuna_core::engine_client::{EngineClient, EngineResult};
use setsuna_linux::keys::{Key, MOD_ALT, MOD_CTRL, MOD_SHIFT, MOD_SUPER};
use setsuna_linux::session::Session;
use tungstenite::{Message, WebSocket};

const INDEX: &str = include_str!("../web/index.html");

fn main() {
    let mut port = 8765u16;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--port" => port = args.next().and_then(|p| p.parse().ok()).unwrap_or(port),
            _ => {
                eprintln!("usage: setsuna-demo [--port N]");
                std::process::exit(2);
            }
        }
    }
    let listener = TcpListener::bind(("127.0.0.1", port)).unwrap_or_else(|e| {
        eprintln!("bind 127.0.0.1:{port}: {e}");
        std::process::exit(1);
    });
    println!("Setsuna demo: http://127.0.0.1:{port}/");
    for stream in listener.incoming().flatten() {
        thread::spawn(move || {
            if let Err(e) = serve(stream) {
                eprintln!("connection: {e}");
            }
        });
    }
}

/// Route by the request line, peeked so the WebSocket handshake still sees
/// the whole request.
fn serve(stream: TcpStream) -> std::io::Result<()> {
    let mut head = [0u8; 64];
    let n = stream.peek(&mut head)?;
    let line = String::from_utf8_lossy(&head[..n]);
    if line.starts_with("GET /ws") {
        match tungstenite::accept(stream) {
            Ok(ws) => run_session(ws),
            Err(e) => eprintln!("websocket handshake: {e}"),
        }
        return Ok(());
    }
    serve_page(
        stream,
        line.starts_with("GET / ") || line.starts_with("GET /index.html"),
    )
}

fn serve_page(mut stream: TcpStream, index: bool) -> std::io::Result<()> {
    // drain the request headers
    let mut req = Vec::new();
    let mut buf = [0u8; 1024];
    while !req.windows(4).any(|w| w == b"\r\n\r\n") {
        let n = stream.read(&mut buf)?;
        if n == 0 {
            break;
        }
        req.extend_from_slice(&buf[..n]);
    }
    let (status, body) = if index {
        ("200 OK", INDEX)
    } else {
        ("404 Not Found", "not found")
    };
    write!(
        stream,
        "HTTP/1.1 {status}\r\nContent-Type: text/html; charset=utf-8\r\n\
         Content-Length: {}\r\nCache-Control: no-store\r\nConnection: close\r\n\r\n{body}",
        body.len()
    )
}

/// The engine client, keeping a copy of each result for the page.
struct Tap {
    inner: Box<dyn EngineClient>,
    last: Arc<Mutex<Option<EngineResult>>>,
}

impl EngineClient for Tap {
    fn dispatch(&self, req_id: u64, context: &str, reading: &str) {
        self.inner.dispatch(req_id, context, reading);
    }
    fn dispatch_more(&self, req_id: u64, context: &str, reading: &str, continue_beam: u32) {
        self.inner
            .dispatch_more(req_id, context, reading, continue_beam);
    }
    fn take_latest(&self) -> Option<EngineResult> {
        let r = self.inner.take_latest()?;
        *self.last.lock().unwrap() = Some(r.clone());
        Some(r)
    }
    fn is_ready(&self) -> bool {
        self.inner.is_ready()
    }
}

/// Engine figures for the page: the last result and running totals.
#[derive(Default)]
struct Stats {
    seq: u64,
    engine_ms_total: f64,
    last: Option<EngineResult>,
}

impl Stats {
    fn take(&mut self, tap: &Mutex<Option<EngineResult>>) {
        if let Some(r) = tap.lock().unwrap().take() {
            self.seq += 1;
            self.engine_ms_total += r.engine_ms.unwrap_or(0.0);
            self.last = Some(r);
        }
    }

    fn json(&self) -> Value {
        let Some(r) = &self.last else {
            return Value::Null;
        };
        let n = r.candidates.len().min(5);
        // the reading span of each char of the top candidate, as the IME
        // cuts segments: the engine's emitting frames, corrected by the kana
        let top = r.candidates.first().map(String::as_str).unwrap_or("");
        let spans = setsuna_core::segments::char_starts(
            &r.reading,
            top,
            r.reading_pos.first().map(Vec::as_slice),
        )
        .map(|starts| {
            let end = r.reading.chars().count();
            (0..starts.len())
                .map(|i| {
                    [
                        starts[i],
                        starts.get(i + 1).copied().unwrap_or(end).max(starts[i] + 1),
                    ]
                })
                .collect::<Vec<_>>()
        });
        json!({
            "seq": self.seq,
            "reading": r.reading,
            "candidates": &r.candidates[..n],
            "scores": &r.scores[..r.scores.len().min(n)],
            "alignment": r.reading_pos.first(),
            "spans": spans,
            "frames": r.frames,
            "engine_ms": r.engine_ms,
            "round_trip_ms": r.round_trip_ms,
            "avg_engine_ms": self.engine_ms_total / self.seq.max(1) as f64,
        })
    }
}

fn run_session(mut ws: WebSocket<TcpStream>) {
    let mut cfg = Config::load_default();
    // the model's output frame by frame, for the lattice view
    cfg.engine.emit_frames = Some(3);
    let woke = Arc::new(AtomicBool::new(false));
    let last = Arc::new(Mutex::new(None));
    let wake = {
        let woke = woke.clone();
        Box::new(move |_req_id: u64| woke.store(true, Ordering::SeqCst))
    };
    let client = child_daemon::spawn_with_config(&cfg, None, wake).map(|inner| {
        Box::new(Tap {
            inner,
            last: last.clone(),
        }) as Box<dyn EngineClient>
    });
    let mut session = Session::new(&cfg, client);
    let mut stats = Stats::default();

    let model = cfg
        .engine
        .model_manifest
        .as_ref()
        .and_then(|p| p.file_name())
        .map(|n| {
            let n = n.to_string_lossy();
            n.trim_end_matches(".manifest.json").to_string()
        })
        .unwrap_or_default();
    let hello = json!({
        "t": "hello",
        "model": model,
        "engine": session.has_client(),
        "beam_width": cfg.engine.beam_width,
    });
    if ws.send(Message::text(hello.to_string())).is_err() {
        return;
    }
    let _ = send_view(&mut ws, &session, &stats, None);

    // Short read timeouts let the loop pick up engine results between key
    // presses.
    let _ = ws
        .get_mut()
        .set_read_timeout(Some(Duration::from_millis(8)));
    loop {
        match ws.read() {
            Ok(Message::Text(text)) => {
                let Ok(msg) = serde_json::from_str::<Value>(&text) else {
                    continue;
                };
                let taken = handle(&mut session, &msg);
                stats.take(&last);
                if send_view(&mut ws, &session, &stats, taken).is_err() {
                    break;
                }
            }
            Ok(Message::Close(_)) => break,
            Ok(_) => {}
            Err(tungstenite::Error::Io(e))
                if matches!(e.kind(), ErrorKind::WouldBlock | ErrorKind::TimedOut) => {}
            Err(_) => break,
        }
        if woke.swap(false, Ordering::SeqCst) {
            let changed = session.apply_async_result();
            stats.take(&last);
            if changed && send_view(&mut ws, &session, &stats, None).is_err() {
                break;
            }
        }
    }
}

/// Apply a page message; for a key, whether the IME took it.
fn handle(session: &mut Session, msg: &Value) -> Option<bool> {
    match msg["t"].as_str()? {
        "key" => Some(key_of(msg).is_some_and(|k| session.key(&k))),
        // a click on a candidate picks and commits it, as in the addon
        "select" => {
            let i = msg["i"].as_u64()? as usize;
            if session.select(i) {
                session.key(&Key {
                    sym: 0xff0d,
                    code: 0,
                    mods: 0,
                });
            }
            None
        }
        "toggle" => {
            session.toggle();
            None
        }
        "flush" => {
            session.flush();
            None
        }
        _ => None,
    }
}

/// A browser `KeyboardEvent` as the X key fcitx5 would report.
fn key_of(msg: &Value) -> Option<Key> {
    let key = msg["key"].as_str()?;
    let flag = |name: &str, bit: u32| {
        if msg[name].as_bool() == Some(true) {
            bit
        } else {
            0
        }
    };
    let mods = flag("shift", MOD_SHIFT)
        | flag("ctrl", MOD_CTRL)
        | flag("alt", MOD_ALT)
        | flag("meta", MOD_SUPER);
    let sym = match key {
        "Backspace" => 0xff08,
        "Tab" if mods & MOD_SHIFT != 0 => 0xfe20,
        "Tab" => 0xff09,
        "Enter" => 0xff0d,
        "Escape" => 0xff1b,
        "Delete" => 0xffff,
        "Home" => 0xff50,
        "End" => 0xff57,
        "ArrowLeft" => 0xff51,
        "ArrowUp" => 0xff52,
        "ArrowRight" => 0xff53,
        "ArrowDown" => 0xff54,
        "PageUp" => 0xff55,
        "PageDown" => 0xff56,
        "F6" => 0xffc3,
        "F7" => 0xffc4,
        "F8" => 0xffc5,
        "F9" => 0xffc6,
        "F10" => 0xffc7,
        "Zenkaku" | "Hankaku" | "ZenkakuHankaku" => 0xff2a,
        "KanjiMode" => 0xff21,
        k => {
            let mut chars = k.chars();
            match (chars.next(), chars.next()) {
                (Some(c), None) => c as u32,
                _ => return None,
            }
        }
    };
    // the JIS ろ key types `\` but means oem102_backslash
    let code = if msg["code"].as_str() == Some("IntlRo") {
        97
    } else {
        0
    };
    Some(Key { sym, code, mods })
}

fn send_view(
    ws: &mut WebSocket<TcpStream>,
    session: &Session,
    stats: &Stats,
    taken: Option<bool>,
) -> Result<(), ()> {
    let v = session.view();
    let msg = json!({
        "t": "view",
        "taken": taken,
        "commits": v.commits,
        "preedit": v.preedit,
        "focus": v.focus,
        "caret": v.caret,
        "candidates": v.candidates,
        "comments": v.comments,
        "selected": v.selected,
        "suggestions": v.suggestions,
        "suggestion_comments": v.suggestion_comments,
        "reading": v.reading,
        "form": v.form.map(|(name, key)| json!({ "name": name, "key": key })),
        "enabled": session.enabled(),
        "page_size": session.page_size(),
        "engine": stats.json(),
    });
    // the socket is gone; the caller ends the session
    ws.send(Message::text(msg.to_string())).map_err(drop)
}
