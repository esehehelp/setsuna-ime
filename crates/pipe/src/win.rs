//! Both ends of the pipe over Win32: the server's instances (claimed with
//! the first, open to this user's apps at any integrity level and inside an
//! AppContainer, never to another machine) and the client's handle.
//!
//! The handles are synchronous: one thread at a time reads or writes one,
//! in turn (a request, then its response).

use std::ffi::{OsStr, OsString};
use std::io;
use std::os::windows::ffi::OsStrExt;
use std::os::windows::io::AsRawHandle;
use std::path::Path;

use windows_sys::Win32::Foundation::{
    CloseHandle, LocalFree, ERROR_ACCESS_DENIED, ERROR_BROKEN_PIPE, ERROR_NO_DATA, ERROR_PIPE_BUSY,
    ERROR_PIPE_CONNECTED, GENERIC_READ, HANDLE, INVALID_HANDLE_VALUE,
};
use windows_sys::Win32::Security::Authorization::{
    ConvertSidToStringSidW, ConvertStringSecurityDescriptorToSecurityDescriptorW, GetSecurityInfo,
    SDDL_REVISION_1, SE_KERNEL_OBJECT,
};
use windows_sys::Win32::Security::{
    GetSidSubAuthority, GetSidSubAuthorityCount, GetTokenInformation, RevertToSelf, TokenElevation,
    TokenIntegrityLevel, TokenIsAppContainer, TokenUser, OWNER_SECURITY_INFORMATION,
    PSECURITY_DESCRIPTOR, PSID, SECURITY_ATTRIBUTES, TOKEN_ELEVATION, TOKEN_MANDATORY_LABEL,
    TOKEN_QUERY, TOKEN_USER,
};
use windows_sys::Win32::Storage::FileSystem::{
    CreateFileW, ReadFile, WriteFile, FILE_FLAG_FIRST_PIPE_INSTANCE, OPEN_EXISTING,
    PIPE_ACCESS_DUPLEX,
};
use windows_sys::Win32::System::Pipes::{
    ConnectNamedPipe, CreateNamedPipeW, GetNamedPipeServerProcessId, ImpersonateNamedPipeClient,
    WaitNamedPipeW, PIPE_READMODE_BYTE, PIPE_REJECT_REMOTE_CLIENTS, PIPE_TYPE_BYTE,
    PIPE_UNLIMITED_INSTANCES, PIPE_WAIT,
};
use windows_sys::Win32::System::SystemInformation::{
    GetLogicalProcessorInformationEx, RelationCache, CACHE_RELATIONSHIP,
    SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX,
};
use windows_sys::Win32::System::Threading::{
    CreateProcessW, GetCurrentProcess, GetCurrentThread, GetExitCodeProcess,
    GetProcessAffinityMask, OpenProcess, OpenProcessToken, OpenThreadToken,
    QueryFullProcessImageNameW, SetProcessAffinityMask, CREATE_BREAKAWAY_FROM_JOB,
    CREATE_NO_WINDOW, CREATE_UNICODE_ENVIRONMENT, DETACHED_PROCESS, PROCESS_INFORMATION,
    PROCESS_QUERY_LIMITED_INFORMATION, PROCESS_SET_INFORMATION, STARTUPINFOW,
};

const BUFFER: u32 = 64 * 1024;

/// What a client may do with the pipe: read it (FILE_GENERIC_READ) and
/// write to it (FILE_WRITE_DATA, FILE_WRITE_ATTRIBUTES). Not
/// FILE_CREATE_PIPE_INSTANCE (FILE_APPEND_DATA, which GENERIC_WRITE maps
/// to, so a client opens with `CLIENT_ACCESS`), WRITE_DAC or WRITE_OWNER.
const CLIENT_RIGHTS: u32 = 0x0012_018B;
/// What a client opens the pipe for: GENERIC_READ and FILE_WRITE_DATA.
const CLIENT_ACCESS: u32 = GENERIC_READ | 0x2;
/// The server's user adds instances too: FILE_GENERIC_READ and
/// FILE_GENERIC_WRITE, as a duplex instance is made with.
const SERVER_RIGHTS: u32 = 0x0012_019F;
/// CreateFileW's SECURITY_SQOS_PRESENT | SECURITY_IDENTIFICATION: the
/// server may learn who the client is, never act as it.
const IDENTIFY_ONLY: u32 = 0x0010_0000 | 0x0001_0000;
/// SECURITY_MANDATORY_MEDIUM_RID.
const MEDIUM_INTEGRITY: u32 = 0x2000;

fn wide(s: &str) -> Vec<u16> {
    OsStr::new(s).encode_wide().chain(Some(0)).collect()
}

/// The string SID behind `sid`.
///
/// # Safety
/// `sid` is a valid SID.
unsafe fn sid_string(sid: PSID) -> io::Result<String> {
    let mut text = std::ptr::null_mut();
    if ConvertSidToStringSidW(sid, &mut text) == 0 {
        return Err(io::Error::last_os_error());
    }
    let len = (0..).take_while(|&i| *text.add(i) != 0).count();
    let s = String::from_utf16_lossy(std::slice::from_raw_parts(text, len));
    LocalFree(text.cast());
    Ok(s)
}

/// This process's user (`S-1-5-21-...`), the same inside an AppContainer.
pub fn user_sid() -> io::Result<String> {
    unsafe {
        let mut token = std::ptr::null_mut();
        if OpenProcessToken(GetCurrentProcess(), TOKEN_QUERY, &mut token) == 0 {
            return Err(io::Error::last_os_error());
        }
        let sid = token_user(token);
        CloseHandle(token);
        sid
    }
}

/// The user of `token`.
///
/// # Safety
/// `token` is a token handle open for TOKEN_QUERY.
unsafe fn token_user(token: HANDLE) -> io::Result<String> {
    let mut len = 0u32;
    GetTokenInformation(token, TokenUser, std::ptr::null_mut(), 0, &mut len);
    // u64s keep TOKEN_USER aligned
    let mut buf = vec![0u64; (len as usize).div_ceil(8).max(1)];
    if GetTokenInformation(token, TokenUser, buf.as_mut_ptr().cast(), len, &mut len) == 0 {
        return Err(io::Error::last_os_error());
    }
    sid_string((*buf.as_ptr().cast::<TOKEN_USER>()).User.Sid)
}

/// Whether this process runs elevated (an administrator's full token).
pub fn is_elevated() -> io::Result<bool> {
    unsafe {
        let mut token = std::ptr::null_mut();
        if OpenProcessToken(GetCurrentProcess(), TOKEN_QUERY, &mut token) == 0 {
            return Err(io::Error::last_os_error());
        }
        let mut elevation = TOKEN_ELEVATION { TokenIsElevated: 0 };
        let mut len = 0u32;
        let ok = GetTokenInformation(
            token,
            TokenElevation,
            (&mut elevation as *mut TOKEN_ELEVATION).cast(),
            std::mem::size_of::<TOKEN_ELEVATION>() as u32,
            &mut len,
        );
        let err = io::Error::last_os_error();
        CloseHandle(token);
        if ok == 0 {
            return Err(err);
        }
        Ok(elevation.TokenIsElevated != 0)
    }
}

/// Whether this process runs in an AppContainer (a Store app's sandbox): what
/// it starts runs there too, unseen by the login's other apps.
pub fn is_app_container() -> io::Result<bool> {
    unsafe {
        let mut token = std::ptr::null_mut();
        if OpenProcessToken(GetCurrentProcess(), TOKEN_QUERY, &mut token) == 0 {
            return Err(io::Error::last_os_error());
        }
        let mut is = 0u32;
        let mut len = 0u32;
        let ok = GetTokenInformation(
            token,
            TokenIsAppContainer,
            (&mut is as *mut u32).cast(),
            std::mem::size_of::<u32>() as u32,
            &mut len,
        );
        let err = io::Error::last_os_error();
        CloseHandle(token);
        if ok == 0 {
            return Err(err);
        }
        Ok(is != 0)
    }
}

/// The name exists already: another server owns it.
pub fn is_taken(e: &io::Error) -> bool {
    e.raw_os_error() == Some(ERROR_ACCESS_DENIED as i32)
}

/// Every instance is serving a client: wait for one with [`wait`].
pub fn is_busy(e: &io::Error) -> bool {
    e.raw_os_error() == Some(ERROR_PIPE_BUSY as i32)
}

/// Wait up to `timeout_ms` for an instance of `name` to be free; false when
/// none came (or the pipe does not exist).
pub fn wait(name: &str, timeout_ms: u32) -> bool {
    unsafe { WaitNamedPipeW(wide(name).as_ptr(), timeout_ms) != 0 }
}

/// Abort the synchronous I/O `thread` is blocked in (a read of the pipe).
pub fn cancel_io<T>(thread: &std::thread::JoinHandle<T>) {
    unsafe {
        windows_sys::Win32::System::IO::CancelSynchronousIo(thread.as_raw_handle());
    }
}

/// Start `exe args` on its own: no console, no handle of this process
/// inherited (the server outlives the app that starts it), out of this
/// process's job when the job lets it (so the app closing does not end it),
/// with the environment `env` (none: this process's). Returns the pid and
/// whether it left the job.
pub fn spawn_detached(
    exe: &Path,
    args: &[&OsStr],
    cwd: Option<&Path>,
    env: Option<&[(OsString, OsString)]>,
) -> io::Result<(u32, bool)> {
    let wide_z = |s: &OsStr| -> Vec<u16> { s.encode_wide().chain(Some(0)).collect() };
    let mut line = quote(exe.as_os_str());
    for a in args {
        line.push(u16::from(b' '));
        line.extend(quote(a));
    }
    line.push(0);
    let exe_w = wide_z(exe.as_os_str());
    let cwd_w = cwd.map(|d| wide_z(d.as_os_str()));
    let env_w = env.map(env_block);
    let run = |flags: u32| -> io::Result<u32> {
        // CreateProcessW may write to the command line
        let mut line = line.clone();
        let si = STARTUPINFOW {
            cb: std::mem::size_of::<STARTUPINFOW>() as u32,
            ..Default::default()
        };
        let mut pi = PROCESS_INFORMATION::default();
        let ok = unsafe {
            CreateProcessW(
                exe_w.as_ptr(),
                line.as_mut_ptr(),
                std::ptr::null(),
                std::ptr::null(),
                0,
                flags | CREATE_UNICODE_ENVIRONMENT,
                env_w
                    .as_ref()
                    .map_or(std::ptr::null(), |e| e.as_ptr().cast()),
                cwd_w.as_ref().map_or(std::ptr::null(), |d| d.as_ptr()),
                &si,
                &mut pi,
            )
        };
        if ok == 0 {
            return Err(io::Error::last_os_error());
        }
        unsafe {
            CloseHandle(pi.hThread);
            CloseHandle(pi.hProcess);
        }
        Ok(pi.dwProcessId)
    };
    let flags = DETACHED_PROCESS | CREATE_NO_WINDOW;
    match run(flags | CREATE_BREAKAWAY_FROM_JOB) {
        Ok(pid) => Ok((pid, true)),
        // a job that forbids breaking away refuses the flag
        Err(_) => run(flags).map(|pid| (pid, false)),
    }
}

/// The L3 caches of this machine (processor group 0): each one's size and the
/// mask of the logical processors under it. Empty when they cannot be read
/// or some processor is in another group.
fn l3_caches() -> Vec<(u32, usize)> {
    let mut len = 0u32;
    unsafe { GetLogicalProcessorInformationEx(RelationCache, std::ptr::null_mut(), &mut len) };
    if len == 0 {
        return Vec::new();
    }
    // the records hold pointer-sized masks: an 8-aligned buffer
    let mut buf = vec![0u64; (len as usize).div_ceil(8)];
    let base = buf.as_mut_ptr().cast::<u8>();
    if unsafe { GetLogicalProcessorInformationEx(RelationCache, base.cast(), &mut len) } == 0 {
        return Vec::new();
    }
    l3_of(unsafe { std::slice::from_raw_parts(base, (len as usize).min(buf.len() * 8)) })
}

/// `l3_caches` of the records `GetLogicalProcessorInformationEx` wrote: each
/// one as long as its `Size` says, which for a cache is shorter than the
/// struct (whose union holds longer relations), so that one at the end of
/// the buffer is read too.
fn l3_of(buf: &[u8]) -> Vec<(u32, usize)> {
    use std::mem::{offset_of, size_of};
    type Ex = SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX;
    let head = offset_of!(Ex, Anonymous);
    let word = |at: usize| <[u8; 4]>::try_from(&buf[at..at + 4]).unwrap();
    let mut out = Vec::new();
    let mut at = 0usize;
    while at + head <= buf.len() {
        let relationship = i32::from_ne_bytes(word(at + offset_of!(Ex, Relationship)));
        let size = u32::from_ne_bytes(word(at + offset_of!(Ex, Size))) as usize;
        if size < head || size > buf.len() - at {
            break;
        }
        if relationship == RelationCache && size >= head + size_of::<CACHE_RELATIONSHIP>() {
            // within the record, at no alignment the slice promises
            let cache: CACHE_RELATIONSHIP =
                unsafe { std::ptr::read_unaligned(buf.as_ptr().add(at + head).cast()) };
            if cache.Level == 3 {
                let mask = unsafe { cache.Anonymous.GroupMask };
                if cache.GroupCount > 1 || mask.Group != 0 {
                    return Vec::new();
                }
                out.push((cache.CacheSize, mask.Mask));
            }
        }
        at += size;
    }
    out
}

/// The logical processors under the largest L3 cache, where the L3 caches
/// differ in size (a CPU with cache stacked on one of its dies, as a Ryzen
/// X3D); `None` where they are alike or cannot be read.
pub fn largest_l3_mask() -> Option<usize> {
    largest_of(&l3_caches())
}

/// `largest_l3_mask` of these L3 caches (size, processor mask).
fn largest_of(caches: &[(u32, usize)]) -> Option<usize> {
    let largest = caches.iter().map(|c| c.0).max()?;
    if caches.iter().all(|c| c.0 == largest) {
        return None;
    }
    let mask = caches
        .iter()
        .filter(|c| c.0 == largest)
        .fold(0usize, |m, c| m | c.1);
    (mask != 0).then_some(mask)
}

/// Keep process `pid` on the logical processors under the largest L3 cache
/// (`largest_l3_mask`): the model stays in that cache between keys, and a
/// conversion typed 100 ms after the last took 0.53 ms there against 0.93
/// left to the scheduler (Ryzen 9 9950X3D). The mask set, `None` where
/// there is nothing to do (the caches alike, or the process kept to those
/// already).
pub fn place_on_largest_l3(pid: u32) -> io::Result<Option<usize>> {
    let Some(mask) = largest_l3_mask() else {
        return Ok(None);
    };
    unsafe {
        let process = OpenProcess(
            PROCESS_SET_INFORMATION | PROCESS_QUERY_LIMITED_INFORMATION,
            0,
            pid,
        );
        if process.is_null() {
            return Err(io::Error::last_os_error());
        }
        let (mut own, mut system) = (0usize, 0usize);
        let mut result = Ok(None);
        if GetProcessAffinityMask(process, &mut own, &mut system) == 0 {
            result = Err(io::Error::last_os_error());
        } else {
            let keep = own & mask;
            if keep != 0 && keep != own {
                result = if SetProcessAffinityMask(process, keep) == 0 {
                    Err(io::Error::last_os_error())
                } else {
                    Ok(Some(keep))
                };
            }
        }
        CloseHandle(process);
        result
    }
}

/// `vars` as CreateProcessW's environment block: `name=value` each, sorted
/// by name ignoring case, with a NUL after each and one more at the end.
fn env_block(vars: &[(OsString, OsString)]) -> Vec<u16> {
    let upper = |k: &OsString| k.to_string_lossy().to_uppercase();
    let mut sorted: Vec<&(OsString, OsString)> = vars.iter().collect();
    sorted.sort_by_cached_key(|(k, _)| upper(k));
    let mut block = Vec::new();
    for (k, v) in sorted {
        block.extend(k.encode_wide());
        block.push(u16::from(b'='));
        block.extend(v.encode_wide());
        block.push(0);
    }
    // an empty block is two NULs
    if block.is_empty() {
        block.push(0);
    }
    block.push(0);
    block
}

/// Whether process `pid` (one [`spawn_detached`] started) still runs.
pub fn is_running(pid: u32) -> bool {
    // GetExitCodeProcess's code while the process runs
    const STILL_ACTIVE: u32 = 259;
    unsafe {
        let handle = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, 0, pid);
        if handle.is_null() {
            return false;
        }
        let mut code = 0u32;
        let ok = GetExitCodeProcess(handle, &mut code);
        CloseHandle(handle);
        ok != 0 && code == STILL_ACTIVE
    }
}

/// `arg` as one argument of a command line, as the C runtime splits it.
fn quote(arg: &OsStr) -> Vec<u16> {
    const BACKSLASH: u16 = b'\\' as u16;
    const QUOTE: u16 = b'"' as u16;
    let wide: Vec<u16> = arg.encode_wide().collect();
    let plain = !wide.is_empty()
        && !wide
            .iter()
            .any(|&c| c == u16::from(b' ') || c == u16::from(b'\t') || c == QUOTE);
    if plain {
        return wide;
    }
    let mut out = vec![QUOTE];
    let mut backslashes = 0;
    for &c in &wide {
        if c == BACKSLASH {
            backslashes += 1;
            continue;
        }
        // backslashes before a quote are doubled, and the quote escaped
        let n = if c == QUOTE {
            backslashes * 2 + 1
        } else {
            backslashes
        };
        out.extend(std::iter::repeat_n(BACKSLASH, n));
        out.push(c);
        backslashes = 0;
    }
    // and before the closing quote
    out.extend(std::iter::repeat_n(BACKSLASH, backslashes * 2));
    out.push(QUOTE);
    out
}

/// One end of the pipe: a server instance or a client's connection.
pub struct Pipe(HANDLE);

// a kernel handle, usable from any thread
unsafe impl Send for Pipe {}
unsafe impl Sync for Pipe {}

impl Pipe {
    /// A server instance of `name`, open to `user_sid`'s processes. The
    /// `first` one claims the name: it fails ([`is_taken`]) when another
    /// server holds it.
    pub fn create(name: &str, first: bool, user_sid: &str) -> io::Result<Self> {
        // owned by this user, who may add instances (the server does); SYSTEM
        // and AppContainers may only connect; the owner's own rights (OW)
        // read the security, not change it, nor anyone's; the low label lets
        // low-integrity apps in. Any process of this user can still add an
        // instance, which a client finds out by the server's process
        // (`server_is`).
        let sddl = format!(
            "O:{user_sid}D:P(A;;0x20000;;;OW)(A;;{SERVER_RIGHTS:#x};;;{user_sid})\
             (A;;{CLIENT_RIGHTS:#x};;;SY)(A;;{CLIENT_RIGHTS:#x};;;AC)S:(ML;;NW;;;LW)"
        );
        unsafe {
            let mut sd: PSECURITY_DESCRIPTOR = std::ptr::null_mut();
            if ConvertStringSecurityDescriptorToSecurityDescriptorW(
                wide(&sddl).as_ptr(),
                SDDL_REVISION_1,
                &mut sd,
                std::ptr::null_mut(),
            ) == 0
            {
                return Err(io::Error::last_os_error());
            }
            let sa = SECURITY_ATTRIBUTES {
                nLength: std::mem::size_of::<SECURITY_ATTRIBUTES>() as u32,
                lpSecurityDescriptor: sd,
                bInheritHandle: 0,
            };
            let mut open_mode = PIPE_ACCESS_DUPLEX;
            if first {
                open_mode |= FILE_FLAG_FIRST_PIPE_INSTANCE;
            }
            let handle = CreateNamedPipeW(
                wide(name).as_ptr(),
                open_mode,
                PIPE_TYPE_BYTE | PIPE_READMODE_BYTE | PIPE_WAIT | PIPE_REJECT_REMOTE_CLIENTS,
                PIPE_UNLIMITED_INSTANCES,
                BUFFER,
                BUFFER,
                0,
                &sa,
            );
            let err = io::Error::last_os_error();
            LocalFree(sd);
            if handle == INVALID_HANDLE_VALUE {
                return Err(err);
            }
            Ok(Self(handle))
        }
    }

    /// Wait for a client on this server instance.
    pub fn accept(&self) -> io::Result<()> {
        unsafe {
            if ConnectNamedPipe(self.0, std::ptr::null_mut()) != 0 {
                return Ok(());
            }
        }
        let err = io::Error::last_os_error();
        match err.raw_os_error() {
            // it connected between create and accept; or connected and
            // left already, which its first read or write finds
            Some(e) if e == ERROR_PIPE_CONNECTED as i32 || e == ERROR_NO_DATA as i32 => Ok(()),
            _ => Err(err),
        }
    }

    /// The user of the client connected to this server instance, as its
    /// token says (taken by impersonating it at the identification level its
    /// `open` allows, after a read: the pipe impersonates the writer of what
    /// was read last). The pipe is open to every AppContainer of the machine
    /// (the `AC` entry is no one user's), so the server asks.
    pub fn client_user_sid(&self) -> io::Result<String> {
        unsafe {
            if ImpersonateNamedPipeClient(self.0) == 0 {
                return Err(io::Error::last_os_error());
            }
            let mut token: HANDLE = std::ptr::null_mut();
            // as this process (the impersonated client could not open its own thread's token for the server)
            let opened = OpenThreadToken(GetCurrentThread(), TOKEN_QUERY, 1, &mut token);
            let err = io::Error::last_os_error();
            // the thread is the server's again before anything else runs on it
            if RevertToSelf() == 0 {
                // a thread left impersonating must not go on: nothing it does is the server's
                std::process::abort();
            }
            if opened == 0 {
                return Err(err);
            }
            let sid = token_user(token);
            CloseHandle(token);
            sid
        }
    }

    /// Connect to the server of `name`: `NotFound` when there is none,
    /// [`is_busy`] when its instances are all taken.
    pub fn open(name: &str) -> io::Result<Self> {
        let handle = unsafe {
            CreateFileW(
                wide(name).as_ptr(),
                CLIENT_ACCESS,
                0,
                std::ptr::null(),
                OPEN_EXISTING,
                IDENTIFY_ONLY,
                std::ptr::null_mut(),
            )
        };
        if handle == INVALID_HANDLE_VALUE {
            return Err(io::Error::last_os_error());
        }
        Ok(Self(handle))
    }

    /// The server process at the other end of a client's connection.
    pub fn server_pid(&self) -> io::Result<u32> {
        let mut pid = 0u32;
        if unsafe { GetNamedPipeServerProcessId(self.0, &mut pid) } == 0 {
            return Err(io::Error::last_os_error());
        }
        Ok(pid)
    }

    /// Whether the server at the other end of a client's connection is one
    /// the client would have started: its image is `exe` and it runs at
    /// medium integrity or above (any process of the user can add an
    /// instance of the name, a low-integrity one included). `Ok(None)` when
    /// this process may not look at it (an AppContainer's client cannot
    /// open another process).
    pub fn server_is(&self, exe: &Path) -> io::Result<Option<bool>> {
        let pid = self.server_pid()?;
        unsafe {
            let process = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, 0, pid);
            if process.is_null() {
                let err = io::Error::last_os_error();
                if err.raw_os_error() == Some(ERROR_ACCESS_DENIED as i32) {
                    return Ok(None);
                }
                return Err(err);
            }
            let image = {
                let mut buf = vec![0u16; 32768];
                let mut len = buf.len() as u32;
                let ok = QueryFullProcessImageNameW(process, 0, buf.as_mut_ptr(), &mut len) != 0;
                ok.then(|| {
                    use std::os::windows::ffi::OsStringExt;
                    std::path::PathBuf::from(std::ffi::OsString::from_wide(&buf[..len as usize]))
                })
            };
            let mut token: HANDLE = std::ptr::null_mut();
            let integrity = if OpenProcessToken(process, TOKEN_QUERY, &mut token) != 0 {
                let rid = token_integrity(token);
                CloseHandle(token);
                rid
            } else {
                None
            };
            CloseHandle(process);
            let (Some(image), Some(integrity)) = (image, integrity) else {
                return Ok(None);
            };
            let same =
                |a: &Path, b: &Path| match (std::fs::canonicalize(a), std::fs::canonicalize(b)) {
                    (Ok(a), Ok(b)) => a.as_os_str().eq_ignore_ascii_case(b.as_os_str()),
                    _ => false,
                };
            Ok(Some(same(&image, exe) && integrity >= MEDIUM_INTEGRITY))
        }
    }

    /// The SID owning the pipe: a client checks it is its own user's before
    /// sending text to a server.
    pub fn owner_sid(&self) -> io::Result<String> {
        unsafe {
            let mut owner: PSID = std::ptr::null_mut();
            let mut sd: PSECURITY_DESCRIPTOR = std::ptr::null_mut();
            let err = GetSecurityInfo(
                self.0,
                SE_KERNEL_OBJECT,
                OWNER_SECURITY_INFORMATION,
                &mut owner,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                &mut sd,
            );
            if err != 0 {
                return Err(io::Error::from_raw_os_error(err as i32));
            }
            let s = sid_string(owner);
            LocalFree(sd);
            s
        }
    }
}

impl Drop for Pipe {
    fn drop(&mut self) {
        unsafe {
            CloseHandle(self.0);
        }
    }
}

impl io::Read for &Pipe {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        let mut n = 0u32;
        let len = buf.len().min(u32::MAX as usize) as u32;
        if unsafe { ReadFile(self.0, buf.as_mut_ptr(), len, &mut n, std::ptr::null_mut()) } != 0 {
            return Ok(n as usize);
        }
        let err = io::Error::last_os_error();
        // the other end closed: end of stream
        if err.raw_os_error() == Some(ERROR_BROKEN_PIPE as i32) {
            return Ok(0);
        }
        Err(err)
    }
}

impl io::Write for &Pipe {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        let mut n = 0u32;
        let len = buf.len().min(u32::MAX as usize) as u32;
        if unsafe { WriteFile(self.0, buf.as_ptr(), len, &mut n, std::ptr::null_mut()) } == 0 {
            return Err(io::Error::last_os_error());
        }
        Ok(n as usize)
    }

    fn flush(&mut self) -> io::Result<()> {
        // WriteFile hands the bytes to the pipe; nothing is buffered here
        Ok(())
    }
}

impl io::Read for Pipe {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        (&*self).read(buf)
    }
}

impl io::Write for Pipe {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        (&*self).write(buf)
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{BufRead, BufReader, Write};

    fn test_name() -> String {
        format!(
            r"\\.\pipe\setsuna-pipe-test-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        )
    }

    /// A cache's record as Windows writes it: as long as its relation.
    fn cache_record(level: u8, size: u32, mask: usize) -> Vec<u8> {
        use std::mem::{offset_of, size_of};
        type Ex = SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX;
        let head = offset_of!(Ex, Anonymous);
        let mut cache: CACHE_RELATIONSHIP = unsafe { std::mem::zeroed() };
        cache.Level = level;
        cache.CacheSize = size;
        cache.GroupCount = 1;
        cache.Anonymous.GroupMask.Mask = mask;
        let total = head + size_of::<CACHE_RELATIONSHIP>();
        let mut out = vec![0u8; total];
        let at = offset_of!(Ex, Relationship);
        out[at..at + 4].copy_from_slice(&RelationCache.to_ne_bytes());
        let at = offset_of!(Ex, Size);
        out[at..at + 4].copy_from_slice(&(total as u32).to_ne_bytes());
        let bytes = unsafe {
            std::slice::from_raw_parts(
                (&cache as *const CACHE_RELATIONSHIP).cast::<u8>(),
                size_of::<CACHE_RELATIONSHIP>(),
            )
        };
        out[head..].copy_from_slice(bytes);
        out
    }

    #[test]
    fn every_l3_record_is_read_the_last_one_too() {
        assert!(
            cache_record(3, 1, 1).len()
                < std::mem::size_of::<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>()
        );
        let mut buf = cache_record(2, 1 << 20, 0x3);
        buf.extend(cache_record(3, 96 << 20, 0xFFFF));
        buf.extend(cache_record(3, 32 << 20, 0xFFFF_0000));
        assert_eq!(
            l3_of(&buf),
            vec![(96 << 20, 0xFFFF), (32 << 20, 0xFFFF_0000)]
        );
        // a record cut short ends the reading
        buf.truncate(buf.len() - 1);
        assert_eq!(l3_of(&buf), vec![(96 << 20, 0xFFFF)]);
        // this machine's, read whole
        let _ = l3_caches();
    }

    #[test]
    fn a_process_goes_to_the_largest_l3_only_where_they_differ() {
        // a Ryzen X3D: 96 MB over the first die's processors, 32 over the other's
        assert_eq!(
            largest_of(&[(96 << 20, 0xFFFF), (32 << 20, 0xFFFF_0000)]),
            Some(0xFFFF)
        );
        // alike, one, none: left to the scheduler
        assert_eq!(largest_of(&[(32 << 20, 0xFF), (32 << 20, 0xFF00)]), None);
        assert_eq!(largest_of(&[(32 << 20, 0xFF)]), None);
        assert_eq!(largest_of(&[]), None);
        // this machine's, if any, is some of its processors
        let all = l3_caches().iter().fold(0usize, |m, c| m | c.1);
        if let Some(mask) = largest_l3_mask() {
            assert!(
                mask != 0 && mask & !all == 0 && mask != all,
                "{mask:#x} of {all:#x}"
            );
        }
    }

    #[test]
    fn quoting() {
        let q = |s: &str| String::from_utf16(&quote(OsStr::new(s))).unwrap();
        assert_eq!(q("serve"), "serve");
        assert_eq!(q(r"C:\models\lm.json"), r"C:\models\lm.json");
        assert_eq!(q(""), r#""""#);
        assert_eq!(
            q(r"C:\Program Files\x.json"),
            r#""C:\Program Files\x.json""#
        );
        assert_eq!(q(r"C:\dir with space\"), r#""C:\dir with space\\""#);
        assert_eq!(q(r#"a"b"#), r#""a\"b""#);
        assert_eq!(q(r#"a\"b c"#), r#""a\\\"b c""#);
    }

    #[test]
    fn user_sid_is_a_sid() {
        assert!(user_sid().unwrap().starts_with("S-1-"));
    }

    #[test]
    fn elevation_is_read() {
        // whichever way the tests run, the token tells
        is_elevated().unwrap();
        // a test is no Store app
        assert!(!is_app_container().unwrap());
    }

    #[test]
    fn a_started_process_runs_until_it_exits() {
        assert!(is_running(std::process::id()));
        let cmd = Path::new(r"C:\Windows\System32\cmd.exe");
        let (pid, _) =
            spawn_detached(cmd, &[OsStr::new("/c"), OsStr::new("exit")], None, None).unwrap();
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        while is_running(pid) && std::time::Instant::now() < deadline {
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        assert!(!is_running(pid));
    }

    #[test]
    fn first_instance_claims_the_name() {
        let name = test_name();
        let sid = user_sid().unwrap();
        let first = Pipe::create(&name, true, &sid).unwrap();
        let again = Pipe::create(&name, true, &sid).err().expect("taken");
        assert!(is_taken(&again), "{again:?}");
        // more instances of the same server
        let _second = Pipe::create(&name, false, &sid).unwrap();
        drop(first);
    }

    #[test]
    fn lines_both_ways() {
        let name = test_name();
        let sid = user_sid().unwrap();
        let server = Pipe::create(&name, true, &sid).unwrap();
        let client_name = name.clone();
        let client = std::thread::spawn(move || {
            let pipe = Pipe::open(&client_name).unwrap();
            assert_eq!(pipe.owner_sid().unwrap(), user_sid().unwrap());
            assert_eq!(pipe.server_pid().unwrap(), std::process::id());
            // the server is this test's own process, at medium integrity
            let me = std::env::current_exe().unwrap();
            assert_eq!(pipe.server_is(&me).unwrap(), Some(true));
            let other = std::path::Path::new(r"C:\Windows\System32\cmd.exe");
            assert_eq!(pipe.server_is(other).unwrap(), Some(false));
            (&pipe).write_all(b"ping\n").unwrap();
            let mut line = String::new();
            BufReader::new(&pipe).read_line(&mut line).unwrap();
            line
        });
        server.accept().unwrap();
        let mut reader = BufReader::new(&server);
        let mut line = String::new();
        reader.read_line(&mut line).unwrap();
        assert_eq!(line, "ping\n");
        (&server).write_all(b"pong\n").unwrap();
        assert_eq!(client.join().unwrap(), "pong\n");
        // the client gone: end of stream
        line.clear();
        assert_eq!(reader.read_line(&mut line).unwrap(), 0);
        let missing = Pipe::open(&format!("{name}-missing")).err().expect("none");
        assert_eq!(missing.kind(), io::ErrorKind::NotFound);
    }
}

/// Whether a client may talk to the server it connected to (after its
/// owner check): the server is `exe` at medium integrity or above
/// ([`Pipe::server_is`]), or this client, in an AppContainer, cannot look
/// and takes what its user's pipe gives.
pub fn trust_server(pipe: &Pipe, exe: &Path) -> io::Result<()> {
    let ok = match pipe.server_is(exe)? {
        Some(ok) => ok,
        None => is_app_container().unwrap_or(false),
    };
    if ok {
        return Ok(());
    }
    Err(io::Error::new(
        io::ErrorKind::PermissionDenied,
        format!(
            "the pipe's server (pid {}) is not {} at medium integrity",
            pipe.server_pid().unwrap_or(0),
            exe.display()
        ),
    ))
}

/// The integrity level (the mandatory label's RID) of `token`.
fn token_integrity(token: HANDLE) -> Option<u32> {
    unsafe {
        let mut len = 0u32;
        GetTokenInformation(
            token,
            TokenIntegrityLevel,
            std::ptr::null_mut(),
            0,
            &mut len,
        );
        if len == 0 {
            return None;
        }
        let mut buf = vec![0u64; (len as usize).div_ceil(8)];
        if GetTokenInformation(
            token,
            TokenIntegrityLevel,
            buf.as_mut_ptr().cast(),
            len,
            &mut len,
        ) == 0
        {
            return None;
        }
        let label = &*(buf.as_ptr() as *const TOKEN_MANDATORY_LABEL);
        let sid = label.Label.Sid;
        let count = *GetSidSubAuthorityCount(sid);
        if count == 0 {
            return None;
        }
        Some(*GetSidSubAuthority(sid, u32::from(count) - 1))
    }
}
