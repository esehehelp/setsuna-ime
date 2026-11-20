//! The vector instructions the engine's kernels use, as the running CPU has them (checked once): AVX-512 (F, BW, VNNI
//! with AVX2 and FMA), AVX-VNNI (AVX2, FMA and the 256-bit VNNI of Intel's cores from the 12th generation), AVX2
//! (with FMA: Intel from Haswell, every AMD Zen), AVX (Intel's Sandy and Ivy Bridge: 256-bit floats, 128-bit
//! integers with SSSE3 and SSE4.1, no FMA), or none of them (the plain paths, correct and slow): AVX-512 is not
//! every CPU's, so every level down to AVX has its kernels. On aarch64, ARMv8.2's dot products (`dotprod`).
//!
//! SETSUNA_CPU_LEVEL caps the level for research and tests (`avx512`, `avx-vnni`, `avx2`, `avx`, `plain`): the
//! integer kernels give the same values on every level (their tests compare the paths' bits), so a capped run shows
//! how fast a CPU of that level would be.

use std::sync::OnceLock;

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub enum Level {
    Plain,
    Avx,
    Avx2,
    AvxVnni,
    Avx512,
}

/// The kernels' level: the CPU's, capped by SETSUNA_CPU_LEVEL.
pub fn level() -> Level {
    static LEVEL: OnceLock<Level> = OnceLock::new();
    *LEVEL.get_or_init(|| {
        let cap = std::env::var("SETSUNA_CPU_LEVEL").ok();
        capped(detected(), avx_vnni_present(), cap.as_deref())
    })
}

/// AVX-512 F, BW, VNNI (with AVX2 and FMA).
pub fn avx512() -> bool {
    level() == Level::Avx512
}

/// The 256-bit VNNI (AVX-VNNI) as the best path: a CPU without AVX-512 that has it, or one capped to it.
pub fn avx_vnni() -> bool {
    level() == Level::AvxVnni
}

/// AVX2 and FMA (every level from `Avx2` up).
pub fn avx2() -> bool {
    level() >= Level::Avx2
}

/// AVX with SSSE3 and SSE4.1 (every level from `Avx` up).
pub fn avx() -> bool {
    level() >= Level::Avx
}

/// ARMv8.2's dot products (SDOT / UDOT: aarch64 phones since about 2018), unless SETSUNA_CPU_LEVEL is `plain`.
pub fn dotprod() -> bool {
    static DOT: OnceLock<bool> = OnceLock::new();
    *DOT.get_or_init(|| {
        let plain = std::env::var("SETSUNA_CPU_LEVEL").is_ok_and(|v| v.trim() == "plain");
        #[cfg(target_arch = "aarch64")]
        {
            !plain && std::arch::is_aarch64_feature_detected!("dotprod")
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            let _ = plain;
            false
        }
    })
}

/// The CPU's own level (no cap): what a kernel of a level may run on.
pub fn hardware() -> Level {
    static HARDWARE: OnceLock<Level> = OnceLock::new();
    *HARDWARE.get_or_init(detected)
}

/// The CPU has AVX-VNNI (with AVX2 and FMA), whatever level it is at.
pub fn has_avx_vnni() -> bool {
    static VNNI: OnceLock<bool> = OnceLock::new();
    *VNNI.get_or_init(avx_vnni_present)
}

/// The CPU has ARMv8.2's dot products (no cap).
pub fn has_dotprod() -> bool {
    #[cfg(target_arch = "aarch64")]
    {
        std::arch::is_aarch64_feature_detected!("dotprod")
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        false
    }
}

fn detected() -> Level {
    #[cfg(target_arch = "x86_64")]
    {
        let f = |name: &str| match name {
            "avx512f" => is_x86_feature_detected!("avx512f"),
            "avx512bw" => is_x86_feature_detected!("avx512bw"),
            "avx512vnni" => is_x86_feature_detected!("avx512vnni"),
            "avx2" => is_x86_feature_detected!("avx2"),
            "fma" => is_x86_feature_detected!("fma"),
            "avx" => is_x86_feature_detected!("avx"),
            "ssse3" => is_x86_feature_detected!("ssse3"),
            "sse4.1" => is_x86_feature_detected!("sse4.1"),
            _ => false,
        };
        let avx = f("avx") && f("ssse3") && f("sse4.1");
        let avx2 = avx && f("avx2") && f("fma");
        if avx2 && f("avx512f") && f("avx512bw") && f("avx512vnni") {
            return Level::Avx512;
        }
        if avx2 && avx_vnni_present() {
            return Level::AvxVnni;
        }
        if avx2 {
            return Level::Avx2;
        }
        if avx {
            return Level::Avx;
        }
    }
    Level::Plain
}

fn avx_vnni_present() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        is_x86_feature_detected!("avxvnni")
            && is_x86_feature_detected!("avx2")
            && is_x86_feature_detected!("fma")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

/// `detected` capped by `cap` (no cap, or one not known: the CPU's): a cap to AVX-VNNI on a CPU without it is AVX2.
fn capped(detected: Level, vnni: bool, cap: Option<&str>) -> Level {
    let cap = match cap.map(str::trim) {
        Some("plain") => Level::Plain,
        Some("avx") => Level::Avx,
        Some("avx2") => Level::Avx2,
        Some("avx-vnni") => Level::AvxVnni,
        _ => Level::Avx512,
    };
    let l = detected.min(cap);
    if l == Level::AvxVnni && !vnni {
        Level::Avx2
    } else {
        l
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn caps_lower_the_level_and_never_raise_it() {
        assert_eq!(capped(Level::Avx512, true, None), Level::Avx512);
        assert_eq!(capped(Level::Avx512, true, Some("avx2")), Level::Avx2);
        assert_eq!(capped(Level::Avx512, true, Some("avx")), Level::Avx);
        assert_eq!(
            capped(Level::Avx512, true, Some("avx-vnni")),
            Level::AvxVnni
        );
        // an AVX-512 CPU without the 256-bit VNNI (Zen 4): AVX2
        assert_eq!(capped(Level::Avx512, false, Some("avx-vnni")), Level::Avx2);
        assert_eq!(capped(Level::Avx2, false, Some("avx512")), Level::Avx2);
        assert_eq!(capped(Level::Avx, false, Some("avx2")), Level::Avx);
        assert_eq!(capped(Level::AvxVnni, true, Some("plain")), Level::Plain);
        assert_eq!(capped(Level::Avx2, false, Some("something")), Level::Avx2);
    }
}
