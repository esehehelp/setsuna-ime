//! Both ends of the pipe over Win32: the server's instances (claimed with
//! the first, open to this user's apps at any integrity level and inside an
//! AppContainer, never to another machine) and the client's handle.
//!
//! The handles are synchronous: one thread at a time reads or writes one,
//! in turn (a request, then its response).

use std::ffi::OsStr;
use std::io;
use std::os::windows::ffi::OsStrExt;
use std::os::windows::io::AsRawHandle;
use std::path::Path;

use windows_sys::Win32::Foundation::{
    CloseHandle, LocalFree, ERROR_ACCESS_DENIED, ERROR_BROKEN_PIPE, ERROR_NO_DATA, ERROR_PIPE_BUSY,
    ERROR_PIPE_CONNECTED, GENERIC_READ, GENERIC_WRITE, HANDLE, INVALID_HANDLE_VALUE,
};
use windows_sys::Win32::Security::Authorization::{
    ConvertSidToStringSidW, ConvertStringSecurityDescriptorToSecurityDescriptorW, GetSecurityInfo,
    SDDL_REVISION_1, SE_KERNEL_OBJECT,
};
use windows_sys::Win32::Security::{
    GetTokenInformation, TokenUser, OWNER_SECURITY_INFORMATION, PSECURITY_DESCRIPTOR, PSID,
    SECURITY_ATTRIBUTES, TOKEN_QUERY, TOKEN_USER,
};
use windows_sys::Win32::Storage::FileSystem::{
    CreateFileW, ReadFile, WriteFile, FILE_FLAG_FIRST_PIPE_INSTANCE, OPEN_EXISTING,
    PIPE_ACCESS_DUPLEX,
};
use windows_sys::Win32::System::Pipes::{
    ConnectNamedPipe, CreateNamedPipeW, GetNamedPipeServerProcessId, WaitNamedPipeW,
    PIPE_READMODE_BYTE, PIPE_REJECT_REMOTE_CLIENTS, PIPE_TYPE_BYTE, PIPE_UNLIMITED_INSTANCES,
    PIPE_WAIT,
};
use windows_sys::Win32::System::Threading::{
    CreateProcessW, GetCurrentProcess, OpenProcessToken, CREATE_BREAKAWAY_FROM_JOB,
    CREATE_NO_WINDOW, CREATE_UNICODE_ENVIRONMENT, DETACHED_PROCESS, PROCESS_INFORMATION,
    STARTUPINFOW,
};

const BUFFER: u32 = 64 * 1024;

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
        let mut len = 0u32;
        GetTokenInformation(token, TokenUser, std::ptr::null_mut(), 0, &mut len);
        // u64s keep TOKEN_USER aligned
        let mut buf = vec![0u64; (len as usize).div_ceil(8).max(1)];
        let ok = GetTokenInformation(token, TokenUser, buf.as_mut_ptr().cast(), len, &mut len);
        let err = io::Error::last_os_error();
        CloseHandle(token);
        if ok == 0 {
            return Err(err);
        }
        sid_string((*buf.as_ptr().cast::<TOKEN_USER>()).User.Sid)
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
/// process's job when the job lets it (so the app closing does not end it).
/// Returns the pid and whether it left the job.
pub fn spawn_detached(exe: &Path, args: &[&OsStr], cwd: Option<&Path>) -> io::Result<(u32, bool)> {
    let wide_z = |s: &OsStr| -> Vec<u16> { s.encode_wide().chain(Some(0)).collect() };
    let mut line = quote(exe.as_os_str());
    for a in args {
        line.push(u16::from(b' '));
        line.extend(quote(a));
    }
    line.push(0);
    let exe_w = wide_z(exe.as_os_str());
    let cwd_w = cwd.map(|d| wide_z(d.as_os_str()));
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
                std::ptr::null(),
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
        // owner and full access: this user (OW, and by SID should an
        // elevated server's default owner be Administrators), SYSTEM and
        // AppContainers; the low label lets low-integrity apps in
        let sddl = format!(
            "O:{user_sid}D:(A;;GA;;;OW)(A;;GA;;;{user_sid})(A;;GA;;;SY)(A;;GA;;;AC)S:(ML;;NW;;;LW)"
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

    /// Connect to the server of `name`: `NotFound` when there is none,
    /// [`is_busy`] when its instances are all taken.
    pub fn open(name: &str) -> io::Result<Self> {
        let handle = unsafe {
            CreateFileW(
                wide(name).as_ptr(),
                GENERIC_READ | GENERIC_WRITE,
                0,
                std::ptr::null(),
                OPEN_EXISTING,
                0,
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
            r"\\.\pipe\setsuna-rerank-pipe-test-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        )
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
