//! Where a popup goes by the caret: below it, left-aligned with it, inside
//! the work area of the monitor it is on (the screen less the taskbar);
//! above it when below would leave the work area, as MS-IME and Mozc place
//! their windows near the bottom of the screen.

/// A rectangle in screen coordinates (`RECT`'s fields).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Rect {
    pub left: i32,
    pub top: i32,
    pub right: i32,
    pub bottom: i32,
}

/// The top-left corner of a `width` x `height` popup for `caret` in `work`,
/// `gap` pixels off the caret.
pub fn place(caret: Rect, width: i32, height: i32, work: Rect, gap: i32) -> (i32, i32) {
    // as far right as fits, never left of the work area
    let x = caret.left.min(work.right - width).max(work.left);
    let below = caret.bottom + gap;
    let above = caret.top - gap - height;
    let y = if below + height <= work.bottom {
        below
    } else if above >= work.top {
        above
    } else if work.bottom - caret.bottom >= caret.top - work.top {
        // fits on neither side: the side with more room, kept on screen
        (work.bottom - height).max(work.top)
    } else {
        work.top
    };
    (x, y)
}

/// The work area of the monitor `r` is on (the nearest one when it is on
/// none).
#[cfg(target_os = "windows")]
pub fn work_area(r: Rect) -> Option<Rect> {
    use windows::Win32::Foundation::RECT;
    use windows::Win32::Graphics::Gdi::{
        GetMonitorInfoW, MonitorFromRect, MONITORINFO, MONITOR_DEFAULTTONEAREST,
    };
    let rect = RECT {
        left: r.left,
        top: r.top,
        right: r.right,
        bottom: r.bottom,
    };
    let mut info = MONITORINFO {
        cbSize: std::mem::size_of::<MONITORINFO>() as u32,
        ..Default::default()
    };
    unsafe {
        let monitor = MonitorFromRect(&rect, MONITOR_DEFAULTTONEAREST);
        if !GetMonitorInfoW(monitor, &mut info).as_bool() {
            return None;
        }
    }
    let w = info.rcWork;
    Some(Rect {
        left: w.left,
        top: w.top,
        right: w.right,
        bottom: w.bottom,
    })
}

/// `place` in the work area of the monitor the caret is on.
#[cfg(target_os = "windows")]
pub fn place_on_screen(caret: Rect, width: i32, height: i32, gap: i32) -> (i32, i32) {
    match work_area(caret) {
        Some(work) => place(caret, width, height, work, gap),
        None => (caret.left, caret.bottom + gap),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const WORK: Rect = Rect {
        left: 0,
        top: 0,
        right: 1920,
        bottom: 1040,
    };

    fn caret(left: i32, top: i32) -> Rect {
        Rect {
            left,
            top,
            right: left + 2,
            bottom: top + 20,
        }
    }

    #[test]
    fn below_the_caret_where_it_fits() {
        assert_eq!(place(caret(100, 200), 300, 400, WORK, 0), (100, 220));
        assert_eq!(place(caret(100, 200), 300, 400, WORK, 6), (100, 226));
    }

    #[test]
    fn above_the_caret_near_the_bottom() {
        // 1000 + 20 + 400 would leave the work area (the taskbar under it)
        assert_eq!(place(caret(100, 1000), 300, 400, WORK, 0), (100, 600));
        assert_eq!(place(caret(100, 1000), 300, 400, WORK, 6), (100, 594));
    }

    #[test]
    fn kept_inside_on_the_right_and_left() {
        assert_eq!(place(caret(1800, 200), 300, 400, WORK, 0), (1620, 220));
        let second = Rect {
            left: -1280,
            top: 0,
            right: 0,
            bottom: 1024,
        };
        assert_eq!(place(caret(-1400, 200), 300, 400, second, 0), (-1280, 220));
        assert_eq!(place(caret(-100, 900), 300, 400, second, 0), (-300, 500));
    }

    #[test]
    fn too_tall_for_either_side() {
        let short = Rect {
            bottom: 600,
            ..WORK
        };
        // more room below: as low as it fits
        assert_eq!(place(caret(10, 100), 100, 550, short, 0), (10, 50));
        // more room above: at the top
        assert_eq!(place(caret(10, 400), 100, 550, short, 0), (10, 0));
        // taller than the work area: from its top
        assert_eq!(place(caret(10, 100), 100, 700, short, 0), (10, 0));
    }
}
