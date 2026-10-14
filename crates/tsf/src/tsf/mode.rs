//! The input mode on the taskbar: あ while the IME is on, A while it is off,
//! as fcitx5's status area shows it (`crates/linux/fcitx5`), and 直 while
//! on with the rewriter.
//!
//! Windows 8 and later show one language bar item of an IME in the input
//! indicator: the one whose `guidItem` is `GUID_LBI_INPUTMODE` (the IME is
//! registered with `GUID_TFCAT_TIPCAP_SYSTRAYSUPPORT`). Its icon is drawn
//! here at the small-icon size, a glyph in the taskbar's text color with a
//! half-opaque outline, as Microsoft's guidelines ask.
//!
//! The on / off state itself is the thread's `GUID_COMPARTMENT_KEYBOARD_
//! OPENCLOSE` compartment, so the key, a click on the icon and other
//! programs (an app opening the IME through IMM) all change the same state;
//! the service listens to it (`ITfCompartmentEventSink`) and updates the item.

use std::cell::{Cell, RefCell};
use std::rc::Rc;

use windows::core::*;
use windows::Win32::Foundation::*;
use windows::Win32::Graphics::Gdi::*;
use windows::Win32::System::Ole::{
    CONNECT_E_ADVISELIMIT, CONNECT_E_CANNOTCONNECT, CONNECT_E_NOCONNECTION,
};
use windows::Win32::System::Registry::{RegGetValueW, HKEY_CURRENT_USER, RRF_RT_REG_DWORD};
use windows::Win32::UI::TextServices::*;
use windows::Win32::UI::WindowsAndMessaging::*;

use crate::globals::CLSID_SETSUNA_TEXT_SERVICE;

/// What the item shows, shared with the service that changes it.
pub(crate) struct ModeState {
    pub on: Cell<bool>,
    /// Rewriting is on (`[rewriter] enabled`), shown while the IME is.
    pub rewrite: Cell<bool>,
    sink: RefCell<Option<ITfLangBarItemSink>>,
    /// A click on the item: the service toggles the mode.
    on_click: Box<dyn Fn()>,
}

impl ModeState {
    pub fn new(on: bool, on_click: Box<dyn Fn()>) -> Rc<Self> {
        Rc::new(Self {
            on: Cell::new(on),
            rewrite: Cell::new(false),
            sink: RefCell::new(None),
            on_click,
        })
    }

    /// What the item shows: あ on, 直 on and rewriting, A off.
    pub fn glyph(&self) -> &'static str {
        match (self.on.get(), self.rewrite.get()) {
            (false, _) => "A",
            (true, true) => "直",
            (true, false) => "あ",
        }
    }

    /// Show `on`: the language bar asks for the icon and text again.
    pub fn set(&self, on: bool) {
        self.on.set(on);
        let sink = self.sink.borrow().clone();
        if let Some(sink) = sink {
            unsafe {
                let _ = sink.OnUpdate(TF_LBI_ICON | TF_LBI_STATUS | TF_LBI_TEXT | TF_LBI_TOOLTIP);
            }
        }
    }
}

#[implement(ITfLangBarItemButton, ITfLangBarItem, ITfSource)]
pub(crate) struct ModeItem {
    state: Rc<ModeState>,
}

impl ModeItem {
    pub fn new(state: Rc<ModeState>) -> Self {
        Self { state }
    }
}

/// The only sink cookie the item hands out.
const SINK_COOKIE: u32 = 1;

/// Menu item ids (`InitMenu`).
const MENU_SETTINGS: u32 = 1;

/// The settings GUI, installed next to the DLL.
const SETTINGS_EXE: &str = "setsuna-settings.exe";

/// The right-click menu of the icon at `pt` (screen coordinates). The
/// taskbar's input indicator calls `OnClick` with `TF_LBI_CLK_RIGHT` and
/// never `InitMenu`, so the item shows the menu itself.
fn show_menu(pt: POINT) {
    unsafe {
        let Ok(menu) = CreatePopupMenu() else {
            return;
        };
        let _ = AppendMenuW(menu, MF_STRING, MENU_SETTINGS as usize, w!("設定..."));
        // A menu needs an owner window in the foreground to close on a
        // click elsewhere: an invisible one, for the menu's lifetime.
        if let Ok(owner) = CreateWindowExW(
            WS_EX_TOOLWINDOW,
            w!("STATIC"),
            PCWSTR::null(),
            WS_POPUP,
            pt.x,
            pt.y,
            0,
            0,
            None,
            None,
            None,
            None,
        ) {
            let _ = SetForegroundWindow(owner);
            let cmd = TrackPopupMenu(
                menu,
                TPM_RETURNCMD | TPM_NONOTIFY | TPM_RIGHTBUTTON | TPM_BOTTOMALIGN,
                pt.x,
                pt.y,
                0,
                owner,
                None,
            );
            let _ = PostMessageW(owner, WM_NULL, WPARAM(0), LPARAM(0));
            let _ = DestroyWindow(owner);
            if cmd.0 as u32 == MENU_SETTINGS {
                open_settings();
            }
        }
        let _ = DestroyMenu(menu);
    }
}

/// Start the settings GUI. It edits `config.toml`, which the IME reads at
/// its next activation.
fn open_settings() {
    let Some(exe) = crate::globals::dll_dir().map(|d| d.join(SETTINGS_EXE)) else {
        return;
    };
    if let Err(e) = std::process::Command::new(&exe).spawn() {
        setsuna_core::diagnostics::log(format!("settings: {} did not start: {e}", exe.display()));
    }
}

impl ITfLangBarItem_Impl for ModeItem_Impl {
    fn GetInfo(&self, pinfo: *mut TF_LANGBARITEMINFO) -> Result<()> {
        if pinfo.is_null() {
            return Err(E_INVALIDARG.into());
        }
        let mut desc = [0u16; 32];
        for (d, c) in desc.iter_mut().zip("Setsuna 入力モード".encode_utf16()) {
            *d = c;
        }
        unsafe {
            *pinfo = TF_LANGBARITEMINFO {
                clsidService: CLSID_SETSUNA_TEXT_SERVICE,
                guidItem: GUID_LBI_INPUTMODE,
                dwStyle: TF_LBI_STYLE_BTN_BUTTON | TF_LBI_STYLE_SHOWNINTRAY,
                ulSort: 0,
                szDescription: desc,
            };
        }
        Ok(())
    }

    fn GetStatus(&self) -> Result<u32> {
        Ok(0)
    }

    fn Show(&self, _fshow: BOOL) -> Result<()> {
        Ok(())
    }

    fn GetTooltipString(&self) -> Result<BSTR> {
        Ok(BSTR::from(match self.state.glyph() {
            "A" => "直接入力 (半角/全角 でひらがな)",
            "直" => "ひらがな・書き直し (半角/全角 で直接入力)",
            _ => "ひらがな (半角/全角 で直接入力)",
        }))
    }
}

impl ITfLangBarItemButton_Impl for ModeItem_Impl {
    fn OnClick(&self, click: TfLBIClick, pt: &POINT, _prcarea: *const RECT) -> Result<()> {
        setsuna_core::diagnostics::log(format!("mode item click {}", click.0));
        if click == TF_LBI_CLK_RIGHT {
            show_menu(*pt);
        } else {
            (self.state.on_click)();
        }
        Ok(())
    }

    /// The menu for the old floating language bar; the taskbar's input
    /// indicator never asks for it (`show_menu`).
    fn InitMenu(&self, pmenu: Option<&ITfMenu>) -> Result<()> {
        let Some(menu) = pmenu else {
            return Ok(());
        };
        let text: Vec<u16> = "設定...".encode_utf16().collect();
        unsafe {
            menu.AddMenuItem(
                MENU_SETTINGS,
                0,
                HBITMAP::default(),
                HBITMAP::default(),
                &text,
                std::ptr::null_mut(),
            )
        }
    }

    fn OnMenuSelect(&self, wid: u32) -> Result<()> {
        if wid == MENU_SETTINGS {
            open_settings();
        }
        Ok(())
    }

    fn GetIcon(&self) -> Result<HICON> {
        // the caller destroys it
        mode_icon(self.state.glyph()).ok_or_else(|| E_FAIL.into())
    }

    fn GetText(&self) -> Result<BSTR> {
        Ok(BSTR::from(self.state.glyph()))
    }
}

#[allow(clippy::not_unsafe_ptr_arg_deref)] // AdviseSink takes *const GUID per the interface
impl ITfSource_Impl for ModeItem_Impl {
    fn AdviseSink(&self, riid: *const GUID, punk: Option<&IUnknown>) -> Result<u32> {
        if riid.is_null() || unsafe { *riid } != ITfLangBarItemSink::IID {
            return Err(CONNECT_E_CANNOTCONNECT.into());
        }
        let Some(punk) = punk else {
            return Err(E_INVALIDARG.into());
        };
        if self.state.sink.borrow().is_some() {
            return Err(CONNECT_E_ADVISELIMIT.into());
        }
        *self.state.sink.borrow_mut() = Some(punk.cast::<ITfLangBarItemSink>()?);
        Ok(SINK_COOKIE)
    }

    fn UnadviseSink(&self, dwcookie: u32) -> Result<()> {
        if dwcookie != SINK_COOKIE || self.state.sink.borrow_mut().take().is_none() {
            return Err(CONNECT_E_NOCONNECTION.into());
        }
        Ok(())
    }
}

/// The thread's on / off compartment.
pub(crate) fn open_close(thread_mgr: &ITfThreadMgr) -> Option<ITfCompartment> {
    unsafe {
        thread_mgr
            .cast::<ITfCompartmentMgr>()
            .ok()?
            .GetCompartment(&GUID_COMPARTMENT_KEYBOARD_OPENCLOSE)
            .ok()
    }
}

/// The compartment's state; `None` while nobody has set it.
pub(crate) fn read_open(compartment: &ITfCompartment) -> Option<bool> {
    let value = unsafe { compartment.GetValue() }.ok()?;
    if value.is_empty() {
        return None;
    }
    i32::try_from(&value).ok().map(|v| v != 0)
}

pub(crate) fn write_open(compartment: &ITfCompartment, client_id: u32, on: bool) -> Result<()> {
    let value = VARIANT::from(i32::from(on));
    unsafe { compartment.SetValue(client_id, &value) }
}

/// Whether the taskbar is light (`SystemUsesLightTheme`); dark when unknown.
fn light_taskbar() -> bool {
    let mut value = 0u32;
    let mut size = std::mem::size_of::<u32>() as u32;
    let status = unsafe {
        RegGetValueW(
            HKEY_CURRENT_USER,
            w!("Software\\Microsoft\\Windows\\CurrentVersion\\Themes\\Personalize"),
            w!("SystemUsesLightTheme"),
            RRF_RT_REG_DWORD,
            None,
            Some(&mut value as *mut u32 as *mut core::ffi::c_void),
            Some(&mut size),
        )
    };
    status.is_ok() && value != 0
}

/// The mode icon at the small-icon size: `text` (あ, 直 or A) in the
/// taskbar's text color with an outline of the other color at half opacity.
pub(crate) fn mode_icon(text: &str) -> Option<HICON> {
    let size = unsafe { GetSystemMetrics(SM_CXSMICON) }.max(16);
    let bgra = icon_pixels(text, size, light_taskbar())?;
    unsafe { icon_from_bgra(&bgra, size) }
}

/// The icon's straight-alpha BGRA pixels, top row first.
fn icon_pixels(text: &str, size: i32, light: bool) -> Option<Vec<u8>> {
    let (glyph, outline) = if light {
        ([0x1f, 0x1f, 0x1f], [0xff, 0xff, 0xff])
    } else {
        ([0xff, 0xff, 0xff], [0x00, 0x00, 0x00])
    };
    let coverage = glyph_coverage(text, size)?;
    let n = size as usize;
    // the outline: the glyph grown by a pixel
    let mut grown = vec![0f32; n * n];
    for y in 0..n {
        for x in 0..n {
            let mut m = 0f32;
            for dy in -1i32..=1 {
                for dx in -1i32..=1 {
                    let (yy, xx) = (y as i32 + dy, x as i32 + dx);
                    if yy >= 0 && xx >= 0 && (yy as usize) < n && (xx as usize) < n {
                        m = m.max(coverage[yy as usize * n + xx as usize]);
                    }
                }
            }
            grown[y * n + x] = m;
        }
    }
    let mut bgra = vec![0u8; n * n * 4];
    for i in 0..n * n {
        let a_glyph = coverage[i];
        let a_line = grown[i] * 0.5 * (1.0 - a_glyph);
        let alpha = a_glyph + a_line;
        if alpha <= 0.0 {
            continue;
        }
        for (c, (g, o)) in glyph.iter().zip(outline.iter()).enumerate() {
            let v = (*g as f32 * a_glyph + *o as f32 * a_line) / alpha;
            // BGRA: the colors are given as RGB
            bgra[i * 4 + 2 - c] = v.round().clamp(0.0, 255.0) as u8;
        }
        bgra[i * 4 + 3] = (alpha * 255.0).round().clamp(0.0, 255.0) as u8;
    }
    Some(bgra)
}

/// How much of each pixel `text` covers, drawn white on black in a
/// `size`×`size` square (row-major, 0..1).
fn glyph_coverage(text: &str, size: i32) -> Option<Vec<f32>> {
    unsafe {
        let screen = GetDC(None);
        let mem = CreateCompatibleDC(screen);
        let bmp = CreateCompatibleBitmap(screen, size, size);
        let old = SelectObject(mem, bmp);
        let rect = RECT {
            left: 0,
            top: 0,
            right: size,
            bottom: size,
        };
        FillRect(mem, &rect, HBRUSH(GetStockObject(BLACK_BRUSH).0));
        let face: Vec<u16> = "Yu Gothic UI\0".encode_utf16().collect();
        let font = CreateFontW(
            -(size * 15 / 16),
            0,
            0,
            0,
            FW_BOLD.0 as i32,
            0,
            0,
            0,
            DEFAULT_CHARSET.0 as u32,
            OUT_DEFAULT_PRECIS.0 as u32,
            CLIP_DEFAULT_PRECIS.0 as u32,
            ANTIALIASED_QUALITY.0 as u32,
            FF_DONTCARE.0 as u32,
            PCWSTR(face.as_ptr()),
        );
        let old_font = SelectObject(mem, font);
        SetBkMode(mem, TRANSPARENT);
        SetTextColor(mem, COLORREF(0x00ff_ffff));
        let mut wide: Vec<u16> = text.encode_utf16().collect();
        let mut r = rect;
        DrawTextW(
            mem,
            &mut wide,
            &mut r,
            DT_CENTER | DT_VCENTER | DT_SINGLELINE | DT_NOPREFIX,
        );
        SelectObject(mem, old_font);
        let _ = DeleteObject(font);
        SelectObject(mem, old);

        let mut info = BITMAPINFO {
            bmiHeader: BITMAPINFOHEADER {
                biSize: std::mem::size_of::<BITMAPINFOHEADER>() as u32,
                biWidth: size,
                biHeight: -size,
                biPlanes: 1,
                biBitCount: 32,
                ..Default::default()
            },
            ..Default::default()
        };
        let mut pixels = vec![0u8; (size * size * 4) as usize];
        let rows = GetDIBits(
            mem,
            bmp,
            0,
            size as u32,
            Some(pixels.as_mut_ptr() as *mut core::ffi::c_void),
            &mut info,
            DIB_RGB_COLORS,
        );
        let _ = DeleteObject(bmp);
        let _ = DeleteDC(mem);
        ReleaseDC(None, screen);
        if rows == 0 {
            return None;
        }
        Some(
            pixels
                .chunks_exact(4)
                .map(|p| p[0].max(p[1]).max(p[2]) as f32 / 255.0)
                .collect(),
        )
    }
}

/// An icon from straight-alpha BGRA pixels, top row first.
unsafe fn icon_from_bgra(bgra: &[u8], size: i32) -> Option<HICON> {
    unsafe {
        let info = BITMAPINFO {
            bmiHeader: BITMAPINFOHEADER {
                biSize: std::mem::size_of::<BITMAPINFOHEADER>() as u32,
                biWidth: size,
                biHeight: -size,
                biPlanes: 1,
                biBitCount: 32,
                ..Default::default()
            },
            ..Default::default()
        };
        let screen = GetDC(None);
        let mut bits: *mut core::ffi::c_void = core::ptr::null_mut();
        let color = CreateDIBSection(screen, &info, DIB_RGB_COLORS, &mut bits, None, 0);
        ReleaseDC(None, screen);
        let color = color.ok()?;
        if bits.is_null() {
            let _ = DeleteObject(color);
            return None;
        }
        core::ptr::copy_nonoverlapping(bgra.as_ptr(), bits as *mut u8, bgra.len());
        // the alpha channel decides; the mask only has to exist
        let mask_bits = vec![0u8; ((size + 15) / 16 * 2 * size) as usize];
        let mask = CreateBitmap(
            size,
            size,
            1,
            1,
            Some(mask_bits.as_ptr() as *const core::ffi::c_void),
        );
        let icon = CreateIconIndirect(&ICONINFO {
            fIcon: TRUE,
            xHotspot: 0,
            yHotspot: 0,
            hbmMask: mask,
            hbmColor: color,
        });
        let _ = DeleteObject(color);
        let _ = DeleteObject(mask);
        icon.ok()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Writes the icons, 4x enlarged on the taskbar's color, into
    /// `local/mode-icon-<on|rewrite|off>-<light|dark>.bmp` to look at:
    /// `cargo test -p setsuna-tsf -- --ignored icon_preview`.
    #[test]
    #[ignore]
    fn icon_preview() {
        let out = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../local");
        std::fs::create_dir_all(&out).unwrap();
        let size = 24;
        let zoom = 4;
        for light in [true, false] {
            let bg: [f32; 3] = if light {
                [238.0, 238.0, 238.0]
            } else {
                [32.0, 32.0, 32.0]
            };
            for (mode, text) in [("on", "あ"), ("rewrite", "直"), ("off", "A")] {
                let px = icon_pixels(text, size, light).unwrap();
                let n = (size * zoom) as usize;
                let mut rows = Vec::with_capacity(n * n * 3);
                // BMP rows bottom-up, BGR
                for y in (0..n).rev() {
                    for x in 0..n {
                        let i = (y / zoom as usize) * size as usize + x / zoom as usize;
                        let a = px[i * 4 + 3] as f32 / 255.0;
                        for c in 0..3 {
                            let v = px[i * 4 + c] as f32 * a + bg[2 - c] * (1.0 - a);
                            rows.push(v.round() as u8);
                        }
                    }
                }
                let mut file = Vec::new();
                file.extend_from_slice(b"BM");
                file.extend_from_slice(&(54 + rows.len() as u32).to_le_bytes());
                file.extend_from_slice(&0u32.to_le_bytes());
                file.extend_from_slice(&54u32.to_le_bytes());
                file.extend_from_slice(&40u32.to_le_bytes());
                file.extend_from_slice(&(n as i32).to_le_bytes());
                file.extend_from_slice(&(n as i32).to_le_bytes());
                file.extend_from_slice(&1u16.to_le_bytes());
                file.extend_from_slice(&24u16.to_le_bytes());
                file.extend_from_slice(&[0u8; 24]);
                file.extend_from_slice(&rows);
                let name = format!(
                    "mode-icon-{mode}-{}.bmp",
                    if light { "light" } else { "dark" }
                );
                std::fs::write(out.join(name), file).unwrap();
            }
        }
    }
}
