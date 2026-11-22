//! The sets typed through a Windows input method into a real text field of
//! this process: the input method activated for this process alone (its TSF
//! profile), a field of its own for each row (a Win32 EDIT, which TSF reaches
//! through CUAS, or a RichEdit with TSF on), the row's context put in first,
//! a row's keys sent with SendInput all at once (the input method takes them
//! as fast as it can), Space once the field is still, Enter after a look at
//! the conversion once the field is still again. Then some rows typed again a key at
//! a time for the latency: a key to the field's first and last change.
//! Also measured: the CPU time
//! and memory of this process (where the input method's DLL runs) and of the
//! input method's own processes that worked for the rows, and the size of its
//! files.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use anyhow::{bail, ensure, Context, Result};
use serde::Serialize;
use windows::core::{w, GUID, PCWSTR, PWSTR};
use windows::Win32::Foundation::*;
use windows::Win32::Graphics::Gdi::{GetStockObject, DEFAULT_GUI_FONT};
use windows::Win32::System::Com::*;
use windows::Win32::System::LibraryLoader::{GetModuleHandleW, LoadLibraryW};
use windows::Win32::System::ProcessStatus::*;
use windows::Win32::System::SystemInformation::GetTickCount;
use windows::Win32::System::Threading::*;
use windows::Win32::UI::Input::Ime::*;
use windows::Win32::UI::Input::KeyboardAndMouse::*;
use windows::Win32::UI::TextServices::*;
use windows::Win32::UI::WindowsAndMessaging::*;

#[derive(clap::Args)]
pub struct Args {
    /// setsuna, msime, shunti, zeroconvert, fumiori, copilot, azookey, mozkey or none (the US
    /// keyboard: the runner's own cost).
    #[arg(long)]
    ime: String,
    /// edit (a Win32 EDIT, through CUAS) or richedit (RichEdit with TSF on).
    #[arg(long, default_value = "edit")]
    field: String,
    #[arg(long, default_value = "crates/bench/sets/v1")]
    sets: PathBuf,
    #[arg(long, value_delimiter = ',')]
    only: Vec<String>,
    /// The run's directory: rows.jsonl and meta.json.
    #[arg(long)]
    out: PathBuf,
    /// A row's keys this far apart; 0: all of them at once (one SendInput),
    /// as fast as the input method takes them.
    #[arg(long, default_value_t = 0)]
    key_ms: u64,
    /// How long the field stays still after the keys before Space.
    #[arg(long, default_value_t = 100)]
    quiet_ms: u64,
    /// How long after Space Enter comes at the earliest (a look at the
    /// conversion; what lands later is not committed), and then once the
    /// field has been still for `quiet_ms`.
    #[arg(long, default_value_t = 500)]
    look_ms: u64,
    #[arg(long, default_value_t = usize::MAX)]
    limit: usize,
    /// After the rows, this many of them (spread over the set) typed again
    /// a key at a time for the latency (0: none).
    #[arg(long, default_value_t = 20)]
    latency_rows: usize,
    /// In the latency rows, how long the field stays still after a key
    /// before the next.
    #[arg(long, default_value_t = 60)]
    latency_quiet_ms: u64,
    /// A row's closing 。、！？ typed after Space, not before it: for an input
    /// method that commits what it shows at that key (Mozkey: before its
    /// delayed live conversion has shown anything, the kana typed).
    #[arg(long)]
    punct_after_space: bool,
    /// The row committed with Enter as it shows, no Space before: for an
    /// input method with live conversion whose Space goes on to the next
    /// candidate of the conversion shown (azooKey, Mozkey: the second one
    /// was committed).
    #[arg(long)]
    no_space: bool,
}

/// `keys` without the closing punctuation key `punct_after_space` types
/// after Space, and that key.
fn split_closing(keys: &str, after_space: bool) -> (&str, Option<char>) {
    match keys.chars().last() {
        Some(c @ ('.' | ',' | '!' | '?')) if after_space && keys.len() > 1 => {
            (&keys[..keys.len() - 1], Some(c))
        }
        _ => (keys, None),
    }
}

struct Profile {
    clsid: GUID,
    profile: GUID,
    /// Virtual keys that put it in Japanese input.
    on_keys: &'static [u16],
    /// Its own processes: executable names starting with these, or, for an
    /// entry with a backslash, executables under that folder (environment
    /// variables as in `storage`).
    processes: &'static [&'static str],
    /// Its files: installed, models and dictionaries, the user's data.
    storage: &'static [&'static str],
}

fn profile(name: &str) -> Result<Option<Profile>> {
    Ok(Some(match name {
        "setsuna" => Profile {
            clsid: GUID::from_u128(0xc3f4_dfa2_6e28_4aa0_9daa_407b_575e_f52d),
            profile: GUID::from_u128(0xaceb_29c0_964e_4ff1_8bdc_b507_aab7_8873),
            on_keys: &[0x16],
            processes: &["setsuna-engine", "setsuna-rerank", "setsuna-rewrite"],
            storage: &[
                "%ProgramFiles%\\setsuna-ime",
                "%LOCALAPPDATA%\\setsuna-ime",
                "%APPDATA%\\setsuna-ime",
            ],
        },
        "msime" => Profile {
            clsid: GUID::from_u128(0x03b5_835f_f03c_411b_9ce2_aa23_e117_1e36),
            profile: GUID::from_u128(0xa76c_93d9_5523_4e90_aafa_4db1_12f9_ac76),
            on_keys: &[0x16],
            processes: &[],
            storage: &[
                "%WINDIR%\\System32\\IME\\IMEJP",
                "%APPDATA%\\Microsoft\\IME\\15.0\\IMEJP",
            ],
        },
        "shunti" => Profile {
            clsid: GUID::from_u128(0x3ca2_ed9d_36d7_4e5e_92fb_5a70_2c39_a46b),
            profile: GUID::from_u128(0x80f9_dcbc_6c43_4043_b004_c627_5415_630e),
            on_keys: &[0x16],
            processes: &[],
            storage: &["%ProgramFiles%\\shunti IME", "%APPDATA%\\shunti IME"],
        },
        // ZeroConvert IME (azooKey for Windows with the ZeroCon model): its
        // conversion server, its llama-server and the other helpers run from
        // the install folder (a llama-server elsewhere is not its own).
        "zeroconvert" => Profile {
            clsid: GUID::from_u128(0xffde_fe79_2fc2_11ef_b16b_94e7_0b2c_378c),
            profile: GUID::from_u128(0xffde_fe7a_2fc2_11ef_b16b_94e7_0b2c_378c),
            on_keys: &[0x16],
            processes: &["%ProgramFiles%\\ZeroConvert IME\\"],
            storage: &[
                "%ProgramFiles%\\ZeroConvert IME",
                "%LOCALAPPDATA%\\Azookey",
                "%APPDATA%\\Azookey",
            ],
        },
        // Fumiori AI IME: Mozc and a local reranker (YamatanaAIIME.exe).
        "fumiori" => Profile {
            clsid: GUID::from_u128(0x10a6_7bc8_22fa_4a59_90dc_2546_652c_56bf),
            profile: GUID::from_u128(0x186f_700c_71cf_43fe_a00e_aacb_1d9e_6d3d),
            on_keys: &[0x16],
            processes: &["%ProgramFiles(x86)%\\Yamatana AI IME\\"],
            storage: &[
                "%ProgramFiles(x86)%\\Yamatana AI IME",
                "%LOCALAPPDATA%\\YamatanaAIIME",
                "%USERPROFILE%\\AppData\\LocalLow\\Mozc",
            ],
        },
        // azooKey for Windows (ZeroConvert's base, the same CLSID: one of the
        // two installed at a time)
        "azookey" => Profile {
            clsid: GUID::from_u128(0xffde_fe79_2fc2_11ef_b16b_94e7_0b2c_378c),
            profile: GUID::from_u128(0xffde_fe7a_2fc2_11ef_b16b_94e7_0b2c_378c),
            on_keys: &[0x16],
            processes: &["%ProgramFiles%\\Azookey\\"],
            storage: &[
                "%ProgramFiles%\\Azookey",
                "%LOCALAPPDATA%\\Azookey",
                "%APPDATA%\\Azookey",
            ],
        },
        // Mozkey: Mozc and a local Zenz (llama-server) in the OSS Mozc's
        // place and CLSID (Fumiori's too: one of the two at a time)
        "mozkey" => Profile {
            clsid: GUID::from_u128(0x10a6_7bc8_22fa_4a59_90dc_2546_652c_56bf),
            profile: GUID::from_u128(0x186f_700c_71cf_43fe_a00e_aacb_1d9e_6d3d),
            on_keys: &[0x16],
            processes: &["%ProgramFiles(x86)%\\Mozc\\"],
            storage: &[
                "%ProgramFiles(x86)%\\Mozc",
                "%USERPROFILE%\\AppData\\LocalLow\\Mozc",
            ],
        },
        "copilot" => Profile {
            clsid: GUID::from_u128(0xd4e1_a2b3_5f6c_4d7e_8a9b_0c1d_2e3f_4a5b),
            profile: GUID::from_u128(0xa1b2_c3d4_e5f6_7890_1234_5678_9abc_def0),
            on_keys: &[0x16],
            processes: &["%ProgramFiles%\\CopilotKeyboard\\"],
            storage: &["%ProgramFiles%\\CopilotKeyboard"],
        },
        "none" => return Ok(None),
        _ => bail!(
            "unknown input method {name} (setsuna, msime, shunti, zeroconvert, fumiori, copilot, azookey, mozkey, none)"
        ),
    }))
}

/// A key of a latency row (latency.jsonl): the key's place in the row's
/// keys, and the key to the field's first and last change.
#[derive(Serialize)]
struct KeyOut {
    id: String,
    key: usize,
    first_ms: f64,
    last_ms: f64,
}

#[derive(Serialize)]
struct RowOut {
    id: String,
    category: String,
    output: String,
    /// The row's keys sent to the field's last change before Space: the
    /// input method through them.
    burst_ms: f64,
    /// Space to the field's first change (the conversion shown); none when
    /// it did not change (a live conversion already in).
    space_first_ms: Option<f64>,
    /// Space to its last change before Enter.
    space_last_ms: Option<f64>,
}

#[derive(Serialize, Default, Clone, Copy)]
struct Usage {
    cpu_ms: f64,
    private_mb: f64,
    working_set_mb: f64,
}

/// This process, and each process whose executable name starts with one of
/// the prefixes, by its id.
struct Snapshot {
    me: Usage,
    others: BTreeMap<u32, Usage>,
}

fn read_usage(h: HANDLE) -> Usage {
    let mut u = Usage::default();
    unsafe {
        let (mut c, mut e, mut k, mut us) = Default::default();
        if GetProcessTimes(h, &mut c, &mut e, &mut k, &mut us).is_ok() {
            let t = |f: FILETIME| {
                ((f.dwHighDateTime as u64) << 32 | f.dwLowDateTime as u64) as f64 / 1e4
            };
            u.cpu_ms = t(k) + t(us);
        }
        let mut m = PROCESS_MEMORY_COUNTERS_EX::default();
        if GetProcessMemoryInfo(
            h,
            &mut m as *mut _ as *mut PROCESS_MEMORY_COUNTERS,
            std::mem::size_of::<PROCESS_MEMORY_COUNTERS_EX>() as u32,
        )
        .is_ok()
        {
            u.private_mb = m.PrivateUsage as f64 / 1048576.0;
            u.working_set_mb = m.WorkingSetSize as f64 / 1048576.0;
        }
    }
    u
}

fn usage(prefixes: &[&str]) -> Snapshot {
    let me = read_usage(unsafe { GetCurrentProcess() });
    let mut others = BTreeMap::new();
    if prefixes.is_empty() {
        return Snapshot { me, others };
    }
    let mut pids = vec![0u32; 8192];
    let mut got = 0u32;
    if unsafe { EnumProcesses(pids.as_mut_ptr(), (pids.len() * 4) as u32, &mut got) }.is_err() {
        return Snapshot { me, others };
    }
    for &pid in &pids[..got as usize / 4] {
        let Ok(h) = (unsafe { OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, false, pid) }) else {
            continue;
        };
        let mut buf = [0u16; 1024];
        let mut n = buf.len() as u32;
        let name = unsafe {
            QueryFullProcessImageNameW(h, PROCESS_NAME_WIN32, PWSTR(buf.as_mut_ptr()), &mut n)
        }
        .ok()
        .map(|_| String::from_utf16_lossy(&buf[..n as usize]));
        if let Some(name) = name {
            let path = name.to_lowercase();
            let exe = path.rsplit('\\').next().unwrap_or("");
            let own = |p: &&str| {
                if p.contains('\\') {
                    let dir = expand(p).to_string_lossy().to_lowercase();
                    path.starts_with(&dir)
                } else {
                    exe.starts_with(&p.to_ascii_lowercase())
                }
            };
            if prefixes.iter().any(own) {
                others.insert(pid, read_usage(h));
            }
        }
        unsafe {
            let _ = CloseHandle(h);
        }
    }
    Snapshot { me, others }
}

/// The sum over `pids` in `snap`.
fn sum(snap: &Snapshot, pids: &[u32]) -> Usage {
    let mut u = Usage::default();
    for p in pids {
        if let Some(x) = snap.others.get(p) {
            u.cpu_ms += x.cpu_ms;
            u.private_mb += x.private_mb;
            u.working_set_mb += x.working_set_mb;
        }
    }
    u
}

fn expand(path: &str) -> PathBuf {
    let mut out = path.to_string();
    for var in [
        "LOCALAPPDATA",
        "APPDATA",
        "USERPROFILE",
        "WINDIR",
        "ProgramFiles(x86)",
        "ProgramFiles",
    ] {
        if let Ok(v) = std::env::var(var) {
            out = out.replace(&format!("%{var}%"), &v);
        }
    }
    PathBuf::from(out)
}

fn dir_size(p: &std::path::Path) -> u64 {
    let Ok(meta) = std::fs::symlink_metadata(p) else {
        return 0;
    };
    if meta.is_file() {
        return meta.len();
    }
    std::fs::read_dir(p)
        .map(|rd| rd.filter_map(|e| e.ok()).map(|e| dir_size(&e.path())).sum())
        .unwrap_or(0)
}

fn pump(ms: u64) {
    let end = Instant::now() + Duration::from_millis(ms);
    unsafe {
        let mut msg = MSG::default();
        loop {
            while PeekMessageW(&mut msg, None, 0, 0, PM_REMOVE).as_bool() {
                let _ = TranslateMessage(&msg);
                DispatchMessageW(&msg);
            }
            if Instant::now() >= end {
                break;
            }
            std::thread::sleep(Duration::from_millis(1));
        }
    }
}

/// The runner's window, and whether it lost the front during a row: keys go
/// only to it, never to a window someone else brought up.
static TOP: std::sync::atomic::AtomicIsize = std::sync::atomic::AtomicIsize::new(0);
static LOST: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// How long nobody may have touched the keyboard or mouse before the runner
/// takes the front back from another window.
const IDLE_BEFORE_TAKING: Duration = Duration::from_secs(60);

/// Wait until the runner's window is in front, so that a key goes nowhere
/// else. Windows lets a process that was not given the front ask for it only
/// once its input is joined to the one in front; it does so only after the
/// keyboard and mouse have been idle a while, never while someone works in
/// another window.
fn wait_front() {
    use std::sync::atomic::Ordering;
    let top = HWND(TOP.load(Ordering::SeqCst) as _);
    if top.is_invalid() {
        return;
    }
    let mut told = false;
    loop {
        let front = unsafe { GetForegroundWindow() };
        if front == top {
            break;
        }
        LOST.store(true, Ordering::SeqCst);
        if !told {
            eprintln!("waiting for the bench window to be in front");
            told = true;
        }
        unsafe {
            if idle() >= IDLE_BEFORE_TAKING {
                let theirs = GetWindowThreadProcessId(front, None);
                let ours = GetCurrentThreadId();
                let joined = theirs != 0
                    && theirs != ours
                    && AttachThreadInput(ours, theirs, true).as_bool();
                let _ = BringWindowToTop(top);
                let _ = SetForegroundWindow(top);
                if joined {
                    let _ = AttachThreadInput(ours, theirs, false);
                }
            } else {
                let _ = SetForegroundWindow(top);
            }
        }
        pump(500);
    }
    if told {
        // the input method Windows gives the window coming to the front
        // lands meanwhile, to be seen by the caller
        pump(300);
        eprintln!("the bench window is in front again");
    }
}

/// The time since the last keyboard or mouse input (this runner's own keys
/// included).
fn idle() -> Duration {
    let mut info = LASTINPUTINFO {
        cbSize: std::mem::size_of::<LASTINPUTINFO>() as u32,
        dwTime: 0,
    };
    unsafe {
        if !GetLastInputInfo(&mut info).as_bool() {
            return Duration::ZERO;
        }
        Duration::from_millis(u64::from(GetTickCount().wrapping_sub(info.dwTime)))
    }
}

fn key(vk: u16, up: bool) {
    if !up {
        wait_front();
    }
    unsafe {
        SendInput(&[event(vk, up)], std::mem::size_of::<INPUT>() as i32);
    }
}

fn event(vk: u16, up: bool) -> INPUT {
    INPUT {
        r#type: INPUT_KEYBOARD,
        Anonymous: INPUT_0 {
            ki: KEYBDINPUT {
                wVk: VIRTUAL_KEY(vk),
                wScan: unsafe { MapVirtualKeyW(vk as u32, MAPVK_VK_TO_VSC) } as u16,
                dwFlags: if up {
                    KEYEVENTF_KEYUP
                } else {
                    KEYBD_EVENT_FLAGS(0)
                },
                time: 0,
                dwExtraInfo: 0,
            },
        },
    }
}

/// All of `keys` in one SendInput, once the runner's window is in front.
fn type_burst(keys: &str) {
    let mut events = Vec::new();
    for c in keys.chars() {
        let scan = unsafe { VkKeyScanW(c as u16) };
        if scan == -1 {
            continue;
        }
        let (vk, shift) = ((scan & 0xff) as u16, scan & 0x100 != 0);
        if shift {
            events.push(event(0x10, false));
        }
        events.push(event(vk, false));
        events.push(event(vk, true));
        if shift {
            events.push(event(0x10, true));
        }
    }
    wait_front();
    unsafe {
        SendInput(&events, std::mem::size_of::<INPUT>() as i32);
    }
}

fn press(vk: u16, shift: bool) {
    if shift {
        key(0x10, false);
    }
    key(vk, false);
    key(vk, true);
    if shift {
        key(0x10, true);
    }
}

fn type_char(c: char) {
    let scan = unsafe { VkKeyScanW(c as u16) };
    if scan == -1 {
        return;
    }
    press((scan & 0xff) as u16, scan & 0x100 != 0);
}

struct Field {
    hwnd: HWND,
}

impl Field {
    fn text(&self) -> String {
        unsafe {
            let n = GetWindowTextLengthW(self.hwnd);
            let mut buf = vec![0u16; n as usize + 1];
            let got = GetWindowTextW(self.hwnd, &mut buf);
            String::from_utf16_lossy(&buf[..got.max(0) as usize])
        }
    }

    /// The composition IMM32 holds (an EDIT under CUAS); a TSF field shows
    /// its composition in its text.
    fn composition(&self) -> String {
        unsafe {
            let himc = ImmGetContext(self.hwnd);
            if himc.is_invalid() {
                return String::new();
            }
            let n = ImmGetCompositionStringW(himc, GCS_COMPSTR, None, 0);
            let mut s = String::new();
            if n > 0 {
                // bytes, which an odd count (an IME's mistake) would leave
                // one past a buffer of n / 2 units: room for all of them,
                // and the call told the buffer's size in bytes as it is
                let mut buf = vec![0u16; (n as usize).div_ceil(2)];
                let got = ImmGetCompositionStringW(
                    himc,
                    GCS_COMPSTR,
                    Some(buf.as_mut_ptr() as *mut _),
                    (buf.len() * 2) as u32,
                );
                let units = (got.max(0) as usize / 2).min(buf.len());
                s = String::from_utf16_lossy(&buf[..units]);
            }
            let _ = ImmReleaseContext(self.hwnd, himc);
            s
        }
    }

    fn view(&self) -> String {
        format!("{}\u{1}{}", self.text(), self.composition())
    }

    /// Pump until the field has been still for `quiet` (at most `cap`); the
    /// times to its first and last change.
    fn settle(&self, quiet: Duration, cap: Duration) -> (Duration, Duration) {
        self.settle_after(Duration::ZERO, quiet, cap)
    }

    /// As `settle`, the field read after each message this thread is sent
    /// or posted (an input method changes the field in the handling of one:
    /// its key, an edit session, its own window's message, a call from its
    /// process), else every millisecond: the times to the change itself, not
    /// to a read some milliseconds on (`settle` reads every few, which put
    /// 4 to 5 ms under every key). For the keys of the latency rows.
    fn settle_close(&self, quiet: Duration, cap: Duration) -> (Duration, Duration) {
        let start = Instant::now();
        let mut last = self.view();
        let mut changed = start;
        let mut first = None;
        loop {
            unsafe {
                let _ = MsgWaitForMultipleObjectsEx(None, 1, QS_ALLINPUT, MWMO_INPUTAVAILABLE);
                let mut msg = MSG::default();
                while PeekMessageW(&mut msg, None, 0, 0, PM_REMOVE).as_bool() {
                    let _ = TranslateMessage(&msg);
                    DispatchMessageW(&msg);
                }
            }
            let now = self.view();
            if now != last {
                last = now;
                changed = Instant::now();
                first.get_or_insert(changed);
            }
            if changed.elapsed() >= quiet || start.elapsed() >= cap {
                return (first.unwrap_or(start) - start, changed - start);
            }
        }
    }

    /// As `settle`, but for `at_least` in any case.
    fn settle_after(
        &self,
        at_least: Duration,
        quiet: Duration,
        cap: Duration,
    ) -> (Duration, Duration) {
        let start = Instant::now();
        let mut last = self.view();
        let mut changed = start;
        let mut first = None;
        loop {
            // every few ms: reading the field takes this process's CPU too
            pump(4);
            let now = self.view();
            if now != last {
                last = now;
                changed = Instant::now();
                first.get_or_insert(changed);
            }
            if (changed.elapsed() >= quiet && start.elapsed() >= at_least) || start.elapsed() >= cap
            {
                return (first.unwrap_or(start) - start, changed - start);
            }
        }
    }
}

fn new_field(top: HWND, rich: bool, context: &str) -> Result<Field> {
    unsafe {
        let hinst = GetModuleHandleW(None)?;
        // ES_MULTILINE | ES_AUTOVSCROLL
        let style =
            WINDOW_STYLE(WS_CHILD.0 | WS_VISIBLE.0 | WS_BORDER.0 | WS_VSCROLL.0 | 0x0004 | 0x0040);
        let class = if rich { w!("RICHEDIT50W") } else { w!("EDIT") };
        let hwnd = CreateWindowExW(
            WINDOW_EX_STYLE(0),
            class,
            PCWSTR::null(),
            style,
            8,
            8,
            960,
            140,
            top,
            None,
            hinst,
            None,
        )?;
        if rich {
            // EM_SETEDITSTYLE: SES_USECTF (TSF on), mask the same
            SendMessageW(hwnd, 0x0400 + 204, WPARAM(0x0001_0000), LPARAM(0x0001_0000));
        }
        let font = GetStockObject(DEFAULT_GUI_FONT);
        SendMessageW(hwnd, WM_SETFONT, WPARAM(font.0 as usize), LPARAM(1));
        let text: Vec<u16> = context.encode_utf16().chain([0]).collect();
        SetWindowTextW(hwnd, PCWSTR(text.as_ptr()))?;
        let n = context.encode_utf16().count();
        SendMessageW(
            hwnd,
            0x00B1, /* EM_SETSEL */
            WPARAM(n),
            LPARAM(n as isize),
        );
        let _ = SetForegroundWindow(top);
        let _ = SetFocus(hwnd);
        Ok(Field { hwnd })
    }
}

/// Whether TSF sees the focused field as transitory (an app not TSF-aware,
/// reached through CUAS).
fn transitory() -> Option<bool> {
    unsafe {
        let tm: ITfThreadMgr =
            CoCreateInstance(&CLSID_TF_ThreadMgr, None, CLSCTX_INPROC_SERVER).ok()?;
        let _ = tm.Activate().ok()?;
        // deactivated whether the focus can be read or not
        let st = tm
            .GetFocus()
            .and_then(|d| d.GetTop())
            .and_then(|c| c.GetStatus());
        let _ = tm.Deactivate();
        Some(st.ok()?.dwStaticFlags & TS_SS_TRANSITORY != 0)
    }
}

#[derive(Serialize)]
struct Meta {
    ime: String,
    field: String,
    field_transitory: Option<bool>,
    rows: usize,
    key_ms: u64,
    quiet_ms: u64,
    look_ms: u64,
    latency_quiet_ms: u64,
    /// A row's closing punctuation typed after Space (`--punct-after-space`).
    punct_after_space: bool,
    /// Rows committed with Enter, no Space (`--no-space`).
    no_space: bool,
    minutes: f64,
    /// The input method's own processes that did work in the run.
    used_processes: usize,
    /// This process before the input method was activated.
    base: Usage,
    /// After the warm-up, idle: this process and the input method's own.
    idle: (Usage, Usage),
    /// The CPU used (ms) while idle for 5 s after the warm-up.
    idle_cpu_ms: (f64, f64),
    /// The CPU used (ms) over the rows, and the most memory seen.
    cpu_ms: (f64, f64),
    peak_private_mb: (f64, f64),
    storage_mb: f64,
    /// A row's keys sent to the field still (`RowOut::burst_ms`).
    burst_ms_p50: f64,
    burst_ms_p95: f64,
    burst_ms_p99: f64,
    /// Rows whose field changed after Space, and Space to the first and
    /// last change in them.
    space_changed: usize,
    space_first_ms_p50: f64,
    space_first_ms_p95: f64,
    space_first_ms_p99: f64,
    space_last_ms_p50: f64,
    space_last_ms_p95: f64,
    space_last_ms_p99: f64,
    /// Latency rows, a key at a time: a key to the field's first change
    /// (the key shown) and to its last (the conversion in).
    key_first_ms_p50: f64,
    key_first_ms_p95: f64,
    key_first_ms_p99: f64,
    key_last_ms_p50: f64,
    key_last_ms_p95: f64,
    key_last_ms_p99: f64,
    latency_keys: usize,
    /// The times the input method had to be made this process's again
    /// (Windows gives a window that comes to the front the input method in
    /// use elsewhere when one serves every app).
    reactivations: usize,
}

/// Make `prof` (none: the US keyboard) this process's input method.
unsafe fn activate(mgr: &ITfInputProcessorProfileMgr, prof: Option<&Profile>) -> Result<()> {
    match prof {
        Some(p) => mgr.ActivateProfile(
            TF_PROFILETYPE_INPUTPROCESSOR,
            0x0411,
            &p.clsid,
            &p.profile,
            HKL::default(),
            TF_IPPMF_FORPROCESS,
        )?,
        None => mgr.ActivateProfile(
            TF_PROFILETYPE_KEYBOARDLAYOUT,
            0x0409,
            &GUID::zeroed(),
            &GUID::zeroed(),
            HKL(0x0409_0409 as _),
            TF_IPPMF_FORPROCESS,
        )?,
    }
    Ok(())
}

/// Whether `prof` (none: a keyboard layout, no input method) is the one
/// this thread's keys go to.
unsafe fn is_active(mgr: &ITfInputProcessorProfileMgr, prof: Option<&Profile>) -> Result<bool> {
    let mut active = TF_INPUTPROCESSORPROFILE::default();
    mgr.GetActiveProfile(&GUID_TFCAT_TIP_KEYBOARD, &mut active)?;
    Ok(match prof {
        Some(p) => {
            active.dwProfileType == TF_PROFILETYPE_INPUTPROCESSOR
                && active.clsid == p.clsid
                && active.guidProfile == p.profile
        }
        None => active.dwProfileType == TF_PROFILETYPE_KEYBOARDLAYOUT,
    })
}

/// Setsuna in this process neither learns from what the runner types, nor
/// records it for prediction, nor sends it to the rewriter: the user's
/// config has an `[[apps]]` entry for the runner that turns them off. Its
/// history would change the user's conversions and the next run's, and the
/// rewriter's answers the outputs. Checked for every input method where
/// Setsuna is set up (`required` only for Setsuna's own run): Windows can
/// give the runner's window the input method in use elsewhere.
fn setsuna_records_nothing(required: bool) -> Result<()> {
    let set_up = setsuna_core::config::Config::appdata_config_path().is_some_and(|p| p.exists());
    if !set_up && !required {
        return Ok(());
    }
    let exe = std::env::current_exe()?;
    let name = exe
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default();
    let cfg = setsuna_core::config::Config::load_from_appdata().unwrap_or_default();
    let (cfg, _) = cfg.for_app(&name);
    ensure!(
        !cfg.learning.enabled && !cfg.frontend.prediction && !cfg.rewriter.enabled,
        "Setsuna would learn from, record or rewrite what this runner types: add to \
         %APPDATA%\\setsuna-ime\\config.toml\n\n[[apps]]\nprocess = \"{name}\"\n\
         learning = false\nprediction = false\nrewrite = false\n"
    );
    Ok(())
}

pub fn run(args: Args) -> Result<()> {
    let rows = crate::set::load(&args.sets, &args.only)?;
    let rows: Vec<_> = rows.into_iter().take(args.limit).collect();
    let prof = profile(&args.ime)?;
    setsuna_records_nothing(args.ime == "setsuna")?;
    let rich = match args.field.as_str() {
        "edit" => false,
        "richedit" => true,
        f => bail!("unknown field {f} (edit, richedit)"),
    };
    std::fs::create_dir_all(&args.out)?;
    let processes: Vec<&str> = prof
        .as_ref()
        .map(|p| p.processes.to_vec())
        .unwrap_or_default();
    unsafe {
        CoInitializeEx(None, COINIT_APARTMENTTHREADED).ok()?;
        if rich {
            LoadLibraryW(w!("Msftedit.dll"))?;
        }
        let hinst = GetModuleHandleW(None)?;
        let top = CreateWindowExW(
            WINDOW_EX_STYLE(WS_EX_TOPMOST.0),
            w!("STATIC"),
            w!("setsuna-bench"),
            WS_OVERLAPPEDWINDOW | WS_VISIBLE,
            80,
            80,
            1000,
            200,
            None,
            None,
            hinst,
            None,
        )?;
        TOP.store(top.0 as isize, std::sync::atomic::Ordering::SeqCst);
        pump(300);
        let base = usage(&processes).me;
        let mgr: ITfInputProcessorProfileMgr =
            CoCreateInstance(&CLSID_TF_InputProcessorProfiles, None, CLSCTX_INPROC_SERVER)?;
        activate(&mgr, prof.as_ref())?;
        pump(800);
        ensure!(
            is_active(&mgr, prof.as_ref())?,
            "{} did not become this process's input method",
            args.ime
        );
        // the times it had to be made this process's input method again
        let mut reactivations = 0;
        let mut hold = |prof: Option<&Profile>| -> Result<()> {
            wait_front();
            if !is_active(&mgr, prof)? {
                reactivations += 1;
                activate(&mgr, prof)?;
                pump(300);
                ensure!(
                    is_active(&mgr, prof)?,
                    "{} did not become this process's input method again",
                    args.ime
                );
            }
            Ok(())
        };
        let on = |p: &Option<Profile>| {
            for &k in p.as_ref().map(|p| p.on_keys).unwrap_or(&[]) {
                press(k, false);
                pump(40);
            }
        };
        let type_keys = |keys: &str| {
            if args.key_ms == 0 {
                type_burst(keys);
            } else {
                for c in keys.chars() {
                    type_char(c);
                    pump(args.key_ms);
                }
            }
        };
        let quiet = Duration::from_millis(args.quiet_ms);
        let look = Duration::from_millis(args.look_ms);
        let after_enter = Duration::from_millis(50);
        let ms = |d: Duration| d.as_secs_f64() * 1e3;
        // warm-up: models loaded, servers up
        let f = new_field(top, rich, "")?;
        pump(300);
        let field_transitory = transitory();
        hold(prof.as_ref())?;
        on(&prof);
        type_keys("korehatesutodesu");
        let _ = f.settle(quiet, Duration::from_secs(10));
        press(0x20, false);
        f.settle(Duration::from_millis(1000), Duration::from_secs(20));
        press(0x0D, false);
        f.settle(after_enter, Duration::from_secs(3));
        let _ = DestroyWindow(f.hwnd);
        pump(3000);
        let idle = usage(&processes);
        pump(5000);
        let idle_end = usage(&processes);

        // accuracy: every row's keys, Space once the field is still, Enter
        // after a look once it is still again
        let start = usage(&processes);
        let mut snaps = Vec::new();
        let mut out = std::io::BufWriter::new(std::fs::File::create(args.out.join("rows.jsonl"))?);
        let (mut burst, mut space_first, mut space_last) = (Vec::new(), Vec::new(), Vec::new());
        let t0 = Instant::now();
        for (n, row) in rows.iter().enumerate() {
            let keys = crate::romaji::keys(&row.reading)
                .map_err(|c| anyhow::anyhow!("{}: cannot type {c:?}", row.id))?;
            let mut tries = 0;
            let (f, s1, s2a, s2) = loop {
                let f = new_field(top, rich, &row.context)?;
                pump(10);
                hold(prof.as_ref())?;
                LOST.store(false, std::sync::atomic::Ordering::SeqCst);
                on(&prof);
                let (body, closing) = split_closing(&keys, args.punct_after_space);
                type_keys(body);
                let (_, s1) = f.settle(quiet, Duration::from_secs(5));
                let (s2a, s2) = if args.no_space {
                    (Duration::ZERO, Duration::ZERO)
                } else {
                    press(0x20, false);
                    f.settle_after(look, quiet, Duration::from_secs(8))
                };
                if let Some(c) = closing {
                    type_char(c);
                    f.settle(quiet, Duration::from_secs(5));
                }
                press(0x0D, false);
                f.settle(after_enter, Duration::from_secs(3));
                tries += 1;
                // typed while another window was in front, or with the
                // input method that window gave this one: again
                let kept = is_active(&mgr, prof.as_ref())?;
                if LOST.load(std::sync::atomic::Ordering::SeqCst) || !kept {
                    let _ = DestroyWindow(f.hwnd);
                    ensure!(
                        tries < 5,
                        "{}: {} did not stay this process's input method",
                        row.id,
                        args.ime
                    );
                    continue;
                }
                break (f, s1, s2a, s2);
            };
            let text = f.text().replace(['\r', '\n'], "");
            let output = text
                .strip_prefix(row.context.as_str())
                .unwrap_or(&text)
                .to_string();
            let _ = DestroyWindow(f.hwnd);
            // no change after Space (a live conversion already shown): none
            let changed = !s2a.is_zero();
            let r = RowOut {
                id: row.id.clone(),
                category: row.category.clone(),
                output,
                burst_ms: ms(s1),
                space_first_ms: changed.then(|| ms(s2a)),
                space_last_ms: changed.then(|| ms(s2)),
            };
            burst.push(r.burst_ms);
            space_first.extend(r.space_first_ms);
            space_last.extend(r.space_last_ms);
            use std::io::Write;
            writeln!(out, "{}", serde_json::to_string(&r)?)?;
            out.flush()?;
            if n % 50 == 49 {
                snaps.push(usage(&processes));
                eprintln!(
                    "{}/{} in {:.1} min",
                    n + 1,
                    rows.len(),
                    t0.elapsed().as_secs_f64() / 60.0
                );
            }
        }
        let end = usage(&processes);
        let minutes = t0.elapsed().as_secs_f64() / 60.0;
        // the input method's processes that worked for these rows (others of
        // the same names, a stale server, are left out)
        let used: Vec<u32> = end
            .others
            .iter()
            .filter(|(pid, u)| u.cpu_ms - start.others.get(pid).map_or(0.0, |s| s.cpu_ms) > 20.0)
            .map(|(pid, _)| *pid)
            .collect();
        let mut peak_me = start.me.private_mb.max(end.me.private_mb);
        let mut peak_others = sum(&start, &used)
            .private_mb
            .max(sum(&end, &used).private_mb);
        for s in &snaps {
            peak_me = peak_me.max(s.me.private_mb);
            peak_others = peak_others.max(sum(s, &used).private_mb);
        }

        // latency: some rows again, a key at a time once the field is
        // still: a key to the field's first change (the key shown) and to
        // its last (its conversion in)
        let (mut key_first, mut key_last) = (Vec::new(), Vec::new());
        let mut key_rows = Vec::new();
        if args.latency_rows > 0 {
            let step = (rows.len() / args.latency_rows).max(1);
            let pause = Duration::from_millis(args.latency_quiet_ms);
            for row in rows.iter().step_by(step).take(args.latency_rows) {
                let keys = crate::romaji::keys(&row.reading).unwrap_or_default();
                let mut tries = 0;
                let got = loop {
                    let f = new_field(top, rich, &row.context)?;
                    pump(10);
                    hold(prof.as_ref())?;
                    LOST.store(false, std::sync::atomic::Ordering::SeqCst);
                    on(&prof);
                    let mut got = Vec::new();
                    for (i, c) in keys.chars().enumerate() {
                        type_char(c);
                        let (first, last) = f.settle_close(pause, Duration::from_secs(2));
                        if !first.is_zero() {
                            got.push(KeyOut {
                                id: row.id.clone(),
                                key: i,
                                first_ms: ms(first),
                                last_ms: ms(last),
                            });
                        }
                    }
                    press(0x20, false);
                    f.settle_after(look, quiet, Duration::from_secs(8));
                    press(0x0D, false);
                    f.settle(after_enter, Duration::from_secs(3));
                    let _ = DestroyWindow(f.hwnd);
                    tries += 1;
                    let kept = is_active(&mgr, prof.as_ref())?;
                    if LOST.load(std::sync::atomic::Ordering::SeqCst) || !kept {
                        ensure!(
                            tries < 5,
                            "{}: {} did not stay this process's input method",
                            row.id,
                            args.ime
                        );
                        continue;
                    }
                    break got;
                };
                for k in got {
                    key_first.push(k.first_ms);
                    key_last.push(k.last_ms);
                    key_rows.push(k);
                }
            }
        }
        if reactivations > 0 {
            eprintln!(
                "{} made this process's input method again {reactivations} times",
                args.ime
            );
        }
        let storage: u64 = prof
            .as_ref()
            .map(|p| p.storage.iter().map(|s| dir_size(&expand(s))).sum())
            .unwrap_or(0);
        let pct = |v: &[f64], p: f64| crate::score::percentile(&mut v.to_vec(), p);
        let meta = Meta {
            ime: args.ime.clone(),
            field: args.field.clone(),
            field_transitory,
            rows: rows.len(),
            key_ms: args.key_ms,
            quiet_ms: args.quiet_ms,
            look_ms: args.look_ms,
            latency_quiet_ms: args.latency_quiet_ms,
            punct_after_space: args.punct_after_space,
            no_space: args.no_space,
            minutes,
            used_processes: used.len(),
            base,
            idle: (idle.me, sum(&idle, &used)),
            idle_cpu_ms: (
                idle_end.me.cpu_ms - idle.me.cpu_ms,
                sum(&idle_end, &used).cpu_ms - sum(&idle, &used).cpu_ms,
            ),
            cpu_ms: (
                end.me.cpu_ms - start.me.cpu_ms,
                sum(&end, &used).cpu_ms - sum(&start, &used).cpu_ms,
            ),
            peak_private_mb: (peak_me, peak_others),
            storage_mb: storage as f64 / 1048576.0,
            burst_ms_p50: pct(&burst, 0.5),
            burst_ms_p95: pct(&burst, 0.95),
            burst_ms_p99: pct(&burst, 0.99),
            space_changed: space_first.len(),
            space_first_ms_p50: pct(&space_first, 0.5),
            space_first_ms_p95: pct(&space_first, 0.95),
            space_first_ms_p99: pct(&space_first, 0.99),
            space_last_ms_p50: pct(&space_last, 0.5),
            space_last_ms_p95: pct(&space_last, 0.95),
            space_last_ms_p99: pct(&space_last, 0.99),
            key_first_ms_p50: pct(&key_first, 0.5),
            key_first_ms_p95: pct(&key_first, 0.95),
            key_first_ms_p99: pct(&key_first, 0.99),
            key_last_ms_p50: pct(&key_last, 0.5),
            key_last_ms_p95: pct(&key_last, 0.95),
            key_last_ms_p99: pct(&key_last, 0.99),
            latency_keys: key_first.len(),
            reactivations,
        };
        let mut lat = String::new();
        for k in &key_rows {
            lat.push_str(&serde_json::to_string(k)?);
            lat.push('\n');
        }
        std::fs::write(args.out.join("latency.jsonl"), lat).context("write latency.jsonl")?;
        std::fs::write(
            args.out.join("meta.json"),
            serde_json::to_string_pretty(&meta)?,
        )
        .context("write meta.json")?;
        eprintln!("{}", serde_json::to_string_pretty(&meta)?);
        let _ = DestroyWindow(top);
    }
    Ok(())
}
