//! The mode switch notice: "あ  ひらがな" / "A  直接入力" by the caret for a
//! moment after the IME is turned on or off, as the fcitx5 addon shows it.
//! Windows draws no such notice for a third-party IME; MS-IME draws its own.
//!
//! Drawn as the candidate window is (`[ui.candidate_window] theme`), never
//! takes the focus, and hides itself on a timer.

use setsuna_core::config::CandidateWindowCfg;
use windows::core::*;
use windows::Win32::Foundation::*;
use windows::Win32::Graphics::Gdi::*;
use windows::Win32::UI::WindowsAndMessaging::*;

use crate::tsf::candidate_window::{
    dpi_scale, draw_text, font, gradient, palette, sc, text_width, Palette, ACCENT, RADIUS,
};

const CLASS_NAME: PCWSTR = w!("SetsunaModePopup");
const HIDE_TIMER: usize = 1;
const SHOWN_MS: u32 = 1000;
const PAD_X: i32 = 14;
const PAD_Y: i32 = 8;
const GAP: i32 = 10;

struct PopupData {
    glyph: String,
    label: String,
    palette: Palette,
    font_face: Vec<u16>,
    font_px: i32,
    width: i32,
    height: i32,
}

pub struct ModePopup {
    hwnd: HWND,
}

impl ModePopup {
    pub fn new(cfg: &CandidateWindowCfg) -> Self {
        register_class();
        let hwnd = unsafe {
            CreateWindowExW(
                WS_EX_TOPMOST | WS_EX_TOOLWINDOW | WS_EX_NOACTIVATE,
                CLASS_NAME,
                w!(""),
                WS_POPUP,
                0,
                0,
                100,
                40,
                None,
                None,
                None,
                None,
            )
            .unwrap_or_default()
        };
        if hwnd.0 as usize != 0 {
            let mut face: Vec<u16> = cfg.font_face.encode_utf16().collect();
            face.push(0);
            let data = Box::new(PopupData {
                glyph: String::new(),
                label: String::new(),
                palette: palette(cfg),
                font_face: face,
                font_px: cfg.font_pt.clamp(8, 48) as i32,
                width: 0,
                height: 0,
            });
            unsafe {
                SetWindowLongPtrW(hwnd, GWLP_USERDATA, Box::into_raw(data) as isize);
            }
        }
        Self { hwnd }
    }

    /// Show the mode for a moment: below `at` (the caret, screen
    /// coordinates), else in the middle of the foreground window.
    pub fn show(&mut self, on: bool, at: Option<(i32, i32)>) {
        let hwnd = self.hwnd;
        if hwnd.0 as usize == 0 {
            return;
        }
        let ptr = unsafe { GetWindowLongPtrW(hwnd, GWLP_USERDATA) } as *mut PopupData;
        let Some(d) = (unsafe { ptr.as_mut() }) else {
            return;
        };
        let (glyph, label) = if on {
            ("あ", "ひらがな")
        } else {
            ("A", "直接入力")
        };
        d.glyph = glyph.to_string();
        d.label = label.to_string();
        let s = dpi_scale(hwnd);
        unsafe {
            let hdc = GetDC(hwnd);
            let big = font(&d.font_face, sc(d.font_px * 3 / 2, s));
            let small = font(&d.font_face, sc(d.font_px, s));
            let old = SelectObject(hdc, big);
            let gw = text_width(hdc, glyph);
            let mut tm = TEXTMETRICW::default();
            let _ = GetTextMetricsW(hdc, &mut tm);
            SelectObject(hdc, small);
            let lw = text_width(hdc, label);
            SelectObject(hdc, old);
            let _ = DeleteObject(big);
            let _ = DeleteObject(small);
            ReleaseDC(hwnd, hdc);
            let accent = if d.palette.accent.is_some() {
                sc(ACCENT, s)
            } else {
                0
            };
            d.width = sc(PAD_X, s) * 2 + gw + sc(GAP, s) + lw;
            d.height = accent + sc(PAD_Y, s) * 2 + tm.tmHeight;
        }
        let (w, h) = (d.width, d.height);
        let (x, y) = match at {
            Some((x, y)) => (x, y + sc(6, s)),
            None => unsafe {
                let mut r = RECT::default();
                let fg = GetForegroundWindow();
                if GetWindowRect(fg, &mut r).is_ok() {
                    ((r.left + r.right - w) / 2, (r.top + r.bottom - h) / 2)
                } else {
                    (
                        GetSystemMetrics(SM_CXSCREEN) / 2 - w / 2,
                        GetSystemMetrics(SM_CYSCREEN) / 2 - h / 2,
                    )
                }
            },
        };
        unsafe {
            let rgn = CreateRoundRectRgn(0, 0, w + 1, h + 1, sc(RADIUS * 2, s), sc(RADIUS * 2, s));
            SetWindowRgn(hwnd, rgn, false);
            let _ = SetWindowPos(hwnd, HWND_TOPMOST, x, y, w, h, SWP_NOACTIVATE);
            let _ = InvalidateRect(hwnd, None, false);
            let _ = ShowWindow(hwnd, SW_SHOWNOACTIVATE);
            SetTimer(hwnd, HIDE_TIMER, SHOWN_MS, None);
        }
    }

    pub fn hide(&mut self) {
        if self.hwnd.0 as usize != 0 {
            unsafe {
                let _ = KillTimer(self.hwnd, HIDE_TIMER);
                let _ = ShowWindow(self.hwnd, SW_HIDE);
            }
        }
    }
}

impl Drop for ModePopup {
    fn drop(&mut self) {
        if self.hwnd.0 as usize != 0 {
            unsafe {
                let ptr = GetWindowLongPtrW(self.hwnd, GWLP_USERDATA) as *mut PopupData;
                if !ptr.is_null() {
                    SetWindowLongPtrW(self.hwnd, GWLP_USERDATA, 0);
                    let _ = Box::from_raw(ptr);
                }
                let _ = DestroyWindow(self.hwnd);
            }
            self.hwnd = HWND::default();
        }
    }
}

fn register_class() {
    use std::sync::Once;
    static REGISTERED: Once = Once::new();
    REGISTERED.call_once(|| unsafe {
        let wc = WNDCLASSW {
            style: CS_HREDRAW | CS_VREDRAW | CS_DROPSHADOW,
            lpfnWndProc: Some(wnd_proc),
            hCursor: LoadCursorW(None, IDC_ARROW).unwrap_or_default(),
            lpszClassName: CLASS_NAME,
            ..Default::default()
        };
        RegisterClassW(&wc);
    });
}

unsafe extern "system" fn wnd_proc(
    hwnd: HWND,
    msg: u32,
    wparam: WPARAM,
    lparam: LPARAM,
) -> LRESULT {
    match msg {
        WM_PAINT => {
            unsafe { paint(hwnd) };
            LRESULT(0)
        }
        WM_ERASEBKGND => LRESULT(1),
        WM_MOUSEACTIVATE => LRESULT(MA_NOACTIVATE as isize),
        WM_TIMER if wparam.0 == HIDE_TIMER => {
            unsafe {
                let _ = KillTimer(hwnd, HIDE_TIMER);
                let _ = ShowWindow(hwnd, SW_HIDE);
            }
            LRESULT(0)
        }
        _ => unsafe { DefWindowProcW(hwnd, msg, wparam, lparam) },
    }
}

unsafe fn paint(hwnd: HWND) {
    unsafe {
        let mut ps = PAINTSTRUCT::default();
        let hdc = BeginPaint(hwnd, &mut ps);
        let ptr = GetWindowLongPtrW(hwnd, GWLP_USERDATA) as *const PopupData;
        if let Some(d) = ptr.as_ref() {
            let (w, h) = (d.width, d.height);
            let s = dpi_scale(hwnd);
            let p = d.palette;
            let mem = CreateCompatibleDC(hdc);
            let bmp = CreateCompatibleBitmap(hdc, w, h);
            let old_bmp = SelectObject(mem, bmp);

            gradient(
                mem,
                RECT {
                    left: 0,
                    top: 0,
                    right: w,
                    bottom: h,
                },
                p.panel_top,
                p.panel_bottom,
                true,
            );
            let accent = if let Some([a, b, c]) = p.accent {
                let ah = sc(ACCENT, s);
                let mid = w / 2;
                gradient(
                    mem,
                    RECT {
                        left: 0,
                        top: 0,
                        right: mid,
                        bottom: ah,
                    },
                    a,
                    b,
                    false,
                );
                gradient(
                    mem,
                    RECT {
                        left: mid,
                        top: 0,
                        right: w,
                        bottom: ah,
                    },
                    b,
                    c,
                    false,
                );
                ah
            } else {
                0
            };
            let rgn = CreateRoundRectRgn(0, 0, w + 1, h + 1, sc(RADIUS * 2, s), sc(RADIUS * 2, s));
            let border = CreateSolidBrush(p.border);
            let _ = FrameRgn(mem, rgn, border, 1, 1);
            let _ = DeleteObject(border);
            let _ = DeleteObject(rgn);

            let _ = SetBkMode(mem, TRANSPARENT);
            let big = font(&d.font_face, sc(d.font_px * 3 / 2, s));
            let small = font(&d.font_face, sc(d.font_px, s));
            let old = SelectObject(mem, big);
            let mut tm = TEXTMETRICW::default();
            let _ = GetTextMetricsW(mem, &mut tm);
            let top = accent + sc(PAD_Y, s);
            // the glyph in the highlight's color, as the selected candidate
            let glyph_color = if p.accent.is_some() {
                p.hl_bottom
            } else {
                p.text
            };
            draw_text(mem, &d.glyph, sc(PAD_X, s), top, glyph_color);
            let gw = text_width(mem, &d.glyph);
            SelectObject(mem, small);
            let mut stm = TEXTMETRICW::default();
            let _ = GetTextMetricsW(mem, &mut stm);
            draw_text(
                mem,
                &d.label,
                sc(PAD_X, s) + gw + sc(GAP, s),
                top + (tm.tmHeight - stm.tmHeight) / 2,
                p.text,
            );
            SelectObject(mem, old);
            let _ = DeleteObject(big);
            let _ = DeleteObject(small);

            let _ = BitBlt(hdc, 0, 0, w, h, mem, 0, 0, SRCCOPY);
            SelectObject(mem, old_bmp);
            let _ = DeleteObject(bmp);
            let _ = DeleteDC(mem);
        }
        let _ = EndPaint(hwnd, &ps);
    }
}
