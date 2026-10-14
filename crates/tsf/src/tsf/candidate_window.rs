//! Win32 candidate window, drawn as the fcitx5 panel (`crates/linux/fcitx5`)
//! is: only the page on screen, "[3/22]" above it, each candidate with its
//! digit label and its note ([全] カタカナ), the page padded to a fixed
//! height so the window does not jump as the user pages.
//!
//! A topmost popup that never takes focus, drawn with GDI into a memory DC:
//! a rounded panel with the system drop shadow, the theme's accent strip
//! along the top and a rounded highlight (`[ui.candidate_window] theme`). A
//! click on a candidate calls the window's `on_select`.

use setsuna_core::config::{CandidateWindowCfg, Config};
use windows::core::*;
use windows::Win32::Foundation::*;
use windows::Win32::Graphics::Gdi::*;
use windows::Win32::UI::HiDpi::GetDpiForWindow;
use windows::Win32::UI::WindowsAndMessaging::*;

const CLASS_NAME: PCWSTR = w!("SetsunaCandidateWindow");

// Layout in pixels at 96 DPI.
const PAD_X: i32 = 12;
const PAD_TOP: i32 = 8;
const PAD_BOTTOM: i32 = 10;
pub(crate) const ACCENT: i32 = 4;
pub(crate) const RADIUS: i32 = 12;
const ROW_EXTRA: i32 = 10;
const HL_RADIUS: i32 = 8;
const HL_INSET: i32 = 6;
const LABEL_GAP: i32 = 6;
const NOTE_GAP: i32 = 16;
const MIN_WIDTH: i32 = 180;

/// Candidates per page (`[ui.candidate_window] page_size`, 1-10: the digit
/// keys pick them).
pub(crate) fn page_size(cfg: &Config) -> usize {
    (cfg.ui.candidate_window.page_size as usize).clamp(1, 10)
}

/// The colors a theme draws with.
#[derive(Clone, Copy)]
pub(crate) struct Palette {
    pub panel_top: COLORREF,
    pub panel_bottom: COLORREF,
    pub border: COLORREF,
    pub text: COLORREF,
    pub muted: COLORREF,
    pub hl_top: COLORREF,
    pub hl_bottom: COLORREF,
    pub hl_border: COLORREF,
    pub hl_text: COLORREF,
    /// The strip along the top, left to right; none for `classic`.
    pub accent: Option<[COLORREF; 3]>,
}

fn rgb(r: u8, g: u8, b: u8) -> COLORREF {
    COLORREF(r as u32 | (g as u32) << 8 | (b as u32) << 16)
}

pub(crate) fn palette(cfg: &CandidateWindowCfg) -> Palette {
    // amber (the old highlight) through coral to violet, as fcitx5's themes
    let accent = Some([rgb(255, 176, 59), rgb(255, 112, 102), rgb(171, 120, 255)]);
    match cfg.theme.as_str() {
        "setsuna-dark" => Palette {
            panel_top: rgb(40, 42, 62),
            panel_bottom: rgb(26, 27, 42),
            border: rgb(72, 74, 98),
            text: rgb(0xe8, 0xe6, 0xf2),
            muted: rgb(0x9a, 0x98, 0xb4),
            hl_top: rgb(255, 166, 72),
            hl_bottom: rgb(226, 86, 40),
            hl_border: rgb(255, 196, 140),
            hl_text: rgb(255, 255, 255),
            accent,
        },
        "classic" => {
            let c = |s: &str| COLORREF(Config::parse_color(s));
            Palette {
                panel_top: c(&cfg.bg_color),
                panel_bottom: c(&cfg.bg_color),
                border: c(&cfg.border_color),
                text: rgb(0, 0, 0),
                muted: c(&cfg.footer_color),
                hl_top: c(&cfg.highlight_color),
                hl_bottom: c(&cfg.highlight_color),
                hl_border: c(&cfg.highlight_color),
                hl_text: c(&cfg.highlight_text_color),
                accent: None,
            }
        }
        _ => Palette {
            panel_top: rgb(255, 255, 255),
            panel_bottom: rgb(246, 243, 255),
            border: rgb(214, 207, 232),
            text: rgb(0x2b, 0x28, 0x38),
            muted: rgb(0x8a, 0x84, 0xa3),
            hl_top: rgb(255, 186, 82),
            hl_bottom: rgb(236, 112, 18),
            hl_border: rgb(214, 104, 10),
            hl_text: rgb(255, 255, 255),
            accent,
        },
    }
}

/// What the window draws, and how a click maps to a candidate.
struct RenderData {
    candidates: Vec<String>,
    notes: Vec<String>,
    /// A line about the list after its position (a rewrite in progress).
    status: String,
    selected: usize,
    page_size: usize,
    font_face: Vec<u16>,
    font_px: i32,
    palette: Palette,
    on_select: Option<Box<dyn Fn(usize)>>,
    /// Laid out by `show` (device pixels).
    layout: Layout,
}

#[derive(Default, Clone, Copy)]
struct Layout {
    width: i32,
    height: i32,
    rows_top: i32,
    row_h: i32,
    label_w: i32,
    note_x: i32,
    scale: f64,
}

pub struct CandidateWindow {
    hwnd: HWND,
}

impl CandidateWindow {
    pub fn new(cfg: &CandidateWindowCfg, page_size: usize) -> Self {
        register_class();
        let hwnd = create_window();
        if hwnd.0 as usize != 0 {
            let mut face: Vec<u16> = cfg.font_face.encode_utf16().collect();
            face.push(0);
            let data = Box::new(RenderData {
                candidates: Vec::new(),
                notes: Vec::new(),
                status: String::new(),
                selected: 0,
                page_size: page_size.max(1),
                font_face: face,
                font_px: cfg.font_pt.clamp(8, 48) as i32,
                palette: palette(cfg),
                on_select: None,
                layout: Layout::default(),
            });
            unsafe {
                SetWindowLongPtrW(hwnd, GWLP_USERDATA, Box::into_raw(data) as isize);
            }
        }
        Self { hwnd }
    }

    fn data(&mut self) -> Option<&mut RenderData> {
        if self.hwnd.0 as usize == 0 {
            return None;
        }
        let ptr = unsafe { GetWindowLongPtrW(self.hwnd, GWLP_USERDATA) } as *mut RenderData;
        unsafe { ptr.as_mut() }
    }

    /// Called with a candidate's index in the list when it is clicked.
    pub fn set_on_select(&mut self, f: Box<dyn Fn(usize)>) {
        if let Some(d) = self.data() {
            d.on_select = Some(f);
        }
    }

    pub fn show(&mut self, candidates: &[String], selected: usize) {
        let hwnd = self.hwnd;
        let Some(d) = self.data() else {
            return;
        };
        d.candidates = candidates.to_vec();
        d.notes = candidates
            .iter()
            .map(|c| {
                setsuna_core::char_forms::annotation(c)
                    .unwrap_or_default()
                    .to_string()
            })
            .collect();
        d.selected = selected.min(candidates.len().saturating_sub(1));
        d.layout = unsafe { layout(hwnd, d) };
        let Layout { width, height, .. } = d.layout;
        let s = d.layout.scale;
        unsafe {
            // the rounded panel: the window's region (the drop shadow follows it)
            let rgn = CreateRoundRectRgn(
                0,
                0,
                width + 1,
                height + 1,
                sc(RADIUS * 2, s),
                sc(RADIUS * 2, s),
            );
            SetWindowRgn(hwnd, rgn, false);
            let _ = SetWindowPos(
                hwnd,
                HWND_TOPMOST,
                0,
                0,
                width,
                height,
                SWP_NOMOVE | SWP_NOACTIVATE,
            );
            let _ = InvalidateRect(hwnd, None, false);
            let _ = ShowWindow(hwnd, SW_SHOWNOACTIVATE);
        }
    }

    /// Show `status` after the list's position, while the window is open;
    /// empty clears it.
    pub fn set_status(&mut self, status: &str) {
        let hwnd = self.hwnd;
        let visible = hwnd.0 as usize != 0 && unsafe { IsWindowVisible(hwnd) }.as_bool();
        let Some(d) = self.data() else {
            return;
        };
        let status = if visible { status } else { "" };
        if d.status == status {
            return;
        }
        d.status = status.to_string();
        if visible {
            let (list, selected) = (d.candidates.clone(), d.selected);
            self.show(&list, selected);
        }
    }

    pub fn hide(&mut self) {
        if self.hwnd.0 as usize == 0 {
            return;
        }
        unsafe {
            let _ = ShowWindow(self.hwnd, SW_HIDE);
        }
    }

    pub fn move_to(&mut self, x: i32, y: i32) {
        if self.hwnd.0 as usize == 0 {
            return;
        }
        unsafe {
            let _ = SetWindowPos(
                self.hwnd,
                HWND_TOPMOST,
                x,
                y,
                0,
                0,
                SWP_NOSIZE | SWP_NOACTIVATE,
            );
        }
    }
}

impl Drop for CandidateWindow {
    fn drop(&mut self) {
        if self.hwnd.0 as usize != 0 {
            unsafe {
                let ptr = GetWindowLongPtrW(self.hwnd, GWLP_USERDATA) as *mut RenderData;
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

fn create_window() -> HWND {
    unsafe {
        CreateWindowExW(
            WS_EX_TOPMOST | WS_EX_TOOLWINDOW | WS_EX_NOACTIVATE,
            CLASS_NAME,
            w!(""),
            WS_POPUP,
            0,
            0,
            200,
            100,
            None,
            None,
            None,
            None,
        )
        .unwrap_or_default()
    }
}

pub(crate) fn dpi_scale(hwnd: HWND) -> f64 {
    let dpi = unsafe { GetDpiForWindow(hwnd) };
    if dpi == 0 {
        1.0
    } else {
        dpi as f64 / 96.0
    }
}

pub(crate) fn sc(v: i32, s: f64) -> i32 {
    (v as f64 * s).round() as i32
}

pub(crate) fn font(face: &[u16], px: i32) -> HFONT {
    unsafe {
        CreateFontW(
            -px,
            0,
            0,
            0,
            FW_NORMAL.0 as i32,
            0,
            0,
            0,
            DEFAULT_CHARSET.0 as u32,
            OUT_DEFAULT_PRECIS.0 as u32,
            CLIP_DEFAULT_PRECIS.0 as u32,
            CLEARTYPE_QUALITY.0 as u32,
            FF_DONTCARE.0 as u32,
            PCWSTR(face.as_ptr()),
        )
    }
}

/// The candidates' font, and the smaller one of the heading and the notes.
fn fonts(d: &RenderData, s: f64) -> (HFONT, HFONT) {
    let px = sc(d.font_px, s);
    (
        font(&d.font_face, px),
        font(&d.font_face, (px * 3 / 4).max(1)),
    )
}

pub(crate) unsafe fn text_width(hdc: HDC, text: &str) -> i32 {
    let wide: Vec<u16> = text.encode_utf16().collect();
    let mut size = SIZE::default();
    unsafe {
        let _ = GetTextExtentPoint32W(hdc, &wide, &mut size);
    }
    size.cx
}

/// The line above the list: the selection's position, then the status.
fn heading(d: &RenderData) -> String {
    let mut h = format!("[{}/{}]", d.selected + 1, d.candidates.len());
    if !d.status.is_empty() {
        h.push_str("  ");
        h.push_str(&d.status);
    }
    h
}

/// Size the window for `d`'s list: as wide as the widest candidate of the
/// list with its note (or the line above it), always a full page high.
unsafe fn layout(hwnd: HWND, d: &RenderData) -> Layout {
    let s = dpi_scale(hwnd);
    unsafe {
        let hdc = GetDC(hwnd);
        let (big, small) = fonts(d, s);
        let old = SelectObject(hdc, big);
        let label_w = text_width(hdc, "0.") + sc(LABEL_GAP, s);
        let text_w = d
            .candidates
            .iter()
            .map(|c| text_width(hdc, c))
            .max()
            .unwrap_or(0);
        let mut tm = TEXTMETRICW::default();
        let _ = GetTextMetricsW(hdc, &mut tm);
        let row_h = tm.tmHeight + sc(ROW_EXTRA, s);
        SelectObject(hdc, small);
        let note_w = d
            .notes
            .iter()
            .map(|n| if n.is_empty() { 0 } else { text_width(hdc, n) })
            .max()
            .unwrap_or(0);
        let mut stm = TEXTMETRICW::default();
        let _ = GetTextMetricsW(hdc, &mut stm);
        let heading_w = text_width(hdc, &heading(d));
        SelectObject(hdc, old);
        let _ = DeleteObject(big);
        let _ = DeleteObject(small);
        ReleaseDC(hwnd, hdc);

        let pad_x = sc(PAD_X, s);
        let note_x = pad_x + label_w + text_w + sc(NOTE_GAP, s);
        let width = (note_x + note_w + pad_x)
            .max(pad_x + heading_w + pad_x)
            .max(sc(MIN_WIDTH, s));
        let accent = if d.palette.accent.is_some() {
            sc(ACCENT, s)
        } else {
            0
        };
        let rows_top = accent + sc(PAD_TOP, s) + stm.tmHeight + sc(4, s);
        let height = rows_top + d.page_size as i32 * row_h + sc(PAD_BOTTOM, s);
        Layout {
            width,
            height,
            rows_top,
            row_h,
            label_w,
            note_x,
            scale: s,
        }
    }
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
        // a click picks a candidate without taking the focus from the app
        WM_MOUSEACTIVATE => LRESULT(MA_NOACTIVATE as isize),
        WM_LBUTTONDOWN => {
            let y = ((lparam.0 >> 16) & 0xffff) as i16 as i32;
            let ptr = unsafe { GetWindowLongPtrW(hwnd, GWLP_USERDATA) } as *const RenderData;
            if let Some(d) = unsafe { ptr.as_ref() } {
                let l = d.layout;
                if y >= l.rows_top && l.row_h > 0 {
                    let row = ((y - l.rows_top) / l.row_h) as usize;
                    let index = d.selected / d.page_size * d.page_size + row;
                    if row < d.page_size && index < d.candidates.len() {
                        if let Some(f) = &d.on_select {
                            f(index);
                        }
                    }
                }
            }
            LRESULT(0)
        }
        _ => unsafe { DefWindowProcW(hwnd, msg, wparam, lparam) },
    }
}

fn vertex(x: i32, y: i32, c: COLORREF) -> TRIVERTEX {
    let ch = |shift: u32| (((c.0 >> shift) & 0xff) as u16) << 8;
    TRIVERTEX {
        x,
        y,
        Red: ch(0),
        Green: ch(8),
        Blue: ch(16),
        Alpha: 0,
    }
}

/// Fill `rect` with a vertical (or horizontal) gradient from `a` to `b`.
pub(crate) unsafe fn gradient(hdc: HDC, rect: RECT, a: COLORREF, b: COLORREF, vertical: bool) {
    let v = [
        vertex(rect.left, rect.top, a),
        vertex(rect.right, rect.bottom, b),
    ];
    let mesh = GRADIENT_RECT {
        UpperLeft: 0,
        LowerRight: 1,
    };
    let mode = if vertical {
        GRADIENT_FILL_RECT_V
    } else {
        GRADIENT_FILL_RECT_H
    };
    unsafe {
        let _ = GradientFill(
            hdc,
            &v,
            &mesh as *const _ as *const core::ffi::c_void,
            1,
            mode,
        );
    }
}

pub(crate) unsafe fn draw_text(hdc: HDC, text: &str, x: i32, y: i32, color: COLORREF) {
    let wide: Vec<u16> = text.encode_utf16().collect();
    unsafe {
        SetTextColor(hdc, color);
        let _ = TextOutW(hdc, x, y, &wide);
    }
}

unsafe fn paint(hwnd: HWND) {
    unsafe {
        let mut ps = PAINTSTRUCT::default();
        let hdc = BeginPaint(hwnd, &mut ps);
        let ptr = GetWindowLongPtrW(hwnd, GWLP_USERDATA) as *const RenderData;
        if let Some(d) = ptr.as_ref() {
            let l = d.layout;
            // draw into a memory DC, then copy: no flicker while paging
            let mem = CreateCompatibleDC(hdc);
            let bmp = CreateCompatibleBitmap(hdc, l.width, l.height);
            let old_bmp = SelectObject(mem, bmp);
            render(mem, d);
            let _ = BitBlt(hdc, 0, 0, l.width, l.height, mem, 0, 0, SRCCOPY);
            SelectObject(mem, old_bmp);
            let _ = DeleteObject(bmp);
            let _ = DeleteDC(mem);
        }
        let _ = EndPaint(hwnd, &ps);
    }
}

unsafe fn render(hdc: HDC, d: &RenderData) {
    let l = d.layout;
    let s = l.scale;
    let p = d.palette;
    unsafe {
        // the panel, its accent strip and its border, inside the rounded region
        let panel = CreateRoundRectRgn(
            0,
            0,
            l.width + 1,
            l.height + 1,
            sc(RADIUS * 2, s),
            sc(RADIUS * 2, s),
        );
        SelectClipRgn(hdc, panel);
        gradient(
            hdc,
            RECT {
                left: 0,
                top: 0,
                right: l.width,
                bottom: l.height,
            },
            p.panel_top,
            p.panel_bottom,
            true,
        );
        if let Some([a, b, c]) = p.accent {
            let h = sc(ACCENT, s);
            let mid = l.width / 2;
            gradient(
                hdc,
                RECT {
                    left: 0,
                    top: 0,
                    right: mid,
                    bottom: h,
                },
                a,
                b,
                false,
            );
            gradient(
                hdc,
                RECT {
                    left: mid,
                    top: 0,
                    right: l.width,
                    bottom: h,
                },
                b,
                c,
                false,
            );
        }
        SelectClipRgn(hdc, HRGN::default());
        let border = CreateSolidBrush(p.border);
        let _ = FrameRgn(hdc, panel, border, 1, 1);
        let _ = DeleteObject(border);
        let _ = DeleteObject(panel);

        let _ = SetBkMode(hdc, TRANSPARENT);
        let (big, small) = fonts(d, s);
        let pad_x = sc(PAD_X, s);
        let accent = if p.accent.is_some() { sc(ACCENT, s) } else { 0 };

        // where the selection is, as fcitx5's aux line
        let old = SelectObject(hdc, small);
        if !d.candidates.is_empty() {
            draw_text(hdc, &heading(d), pad_x, accent + sc(PAD_TOP, s), p.muted);
        }

        let mut tm = TEXTMETRICW::default();
        SelectObject(hdc, big);
        let _ = GetTextMetricsW(hdc, &mut tm);
        let mut stm = TEXTMETRICW::default();
        SelectObject(hdc, small);
        let _ = GetTextMetricsW(hdc, &mut stm);

        let first = d.selected / d.page_size * d.page_size;
        let last = (first + d.page_size).min(d.candidates.len());
        for (row, i) in (first..last).enumerate() {
            let top = l.rows_top + row as i32 * l.row_h;
            let selected = i == d.selected;
            if selected {
                let inset = sc(HL_INSET, s);
                let r = sc(HL_RADIUS * 2, s);
                let hl =
                    CreateRoundRectRgn(inset, top, l.width - inset + 1, top + l.row_h + 1, r, r);
                SelectClipRgn(hdc, hl);
                gradient(
                    hdc,
                    RECT {
                        left: inset,
                        top,
                        right: l.width - inset,
                        bottom: top + l.row_h,
                    },
                    p.hl_top,
                    p.hl_bottom,
                    true,
                );
                SelectClipRgn(hdc, HRGN::default());
                let edge = CreateSolidBrush(p.hl_border);
                let _ = FrameRgn(hdc, hl, edge, 1, 1);
                let _ = DeleteObject(edge);
                let _ = DeleteObject(hl);
            }
            let (text_color, muted) = if selected {
                (p.hl_text, p.hl_text)
            } else {
                (p.text, p.muted)
            };
            let y = top + (l.row_h - tm.tmHeight) / 2;
            SelectObject(hdc, big);
            let label = format!("{}.", (row + 1) % 10);
            draw_text(hdc, &label, pad_x, y, muted);
            draw_text(hdc, &d.candidates[i], pad_x + l.label_w, y, text_color);
            if !d.notes[i].is_empty() {
                SelectObject(hdc, small);
                let ny = top + (l.row_h - stm.tmHeight) / 2;
                draw_text(hdc, &d.notes[i], l.note_x, ny, muted);
            }
        }
        SelectObject(hdc, old);
        let _ = DeleteObject(big);
        let _ = DeleteObject(small);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Write a BMP of `d` drawn as the window draws it.
    unsafe fn save_bmp(d: &RenderData, path: &std::path::Path) {
        let l = d.layout;
        unsafe {
            let screen = GetDC(None);
            let mem = CreateCompatibleDC(screen);
            let bmp = CreateCompatibleBitmap(screen, l.width, l.height);
            let old = SelectObject(mem, bmp);
            render(mem, d);
            SelectObject(mem, old);
            let mut info = BITMAPINFO::default();
            info.bmiHeader = BITMAPINFOHEADER {
                biSize: std::mem::size_of::<BITMAPINFOHEADER>() as u32,
                biWidth: l.width,
                biHeight: -l.height,
                biPlanes: 1,
                biBitCount: 32,
                ..Default::default()
            };
            let mut pixels = vec![0u8; (l.width * l.height * 4) as usize];
            GetDIBits(
                mem,
                bmp,
                0,
                l.height as u32,
                Some(pixels.as_mut_ptr() as *mut _),
                &mut info,
                DIB_RGB_COLORS,
            );
            let _ = DeleteObject(bmp);
            let _ = DeleteDC(mem);
            ReleaseDC(None, screen);
            let mut file = Vec::new();
            let size = 14 + 40 + pixels.len() as u32;
            file.extend_from_slice(b"BM");
            file.extend_from_slice(&size.to_le_bytes());
            file.extend_from_slice(&0u32.to_le_bytes());
            file.extend_from_slice(&54u32.to_le_bytes());
            let header: [u8; 40] = std::mem::transmute(info.bmiHeader);
            file.extend_from_slice(&header);
            file.extend_from_slice(&pixels);
            std::fs::write(path, file).unwrap();
        }
    }

    /// Draws the window in each theme into `local/candidate-window-<theme>.bmp`
    /// to look at: `cargo test -p setsuna-tsf -- --ignored render_preview`.
    #[test]
    #[ignore]
    fn render_preview() {
        let out = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../local");
        std::fs::create_dir_all(&out).unwrap();
        for theme in ["setsuna", "setsuna-dark", "classic"] {
            let cfg = CandidateWindowCfg {
                theme: theme.into(),
                ..Default::default()
            };
            let mut w = CandidateWindow::new(&cfg, 10);
            let hwnd = w.hwnd;
            let d = w.data().unwrap();
            d.candidates = [
                "今日は",
                "京は",
                "教派",
                "キョウハ",
                "ｷｮｳﾊ",
                "きょうは",
                "今日ハ",
            ]
            .iter()
            .map(|s| s.to_string())
            .collect();
            d.notes = d
                .candidates
                .iter()
                .map(|c| {
                    setsuna_core::char_forms::annotation(c)
                        .unwrap_or_default()
                        .to_string()
                })
                .collect();
            d.selected = 1;
            d.layout = unsafe { layout(hwnd, d) };
            unsafe { save_bmp(d, &out.join(format!("candidate-window-{theme}.bmp"))) };
        }
    }
}
