//! The engine's threads on one last-level cache. On an AMD Zen CPU with more than one L3 (the CCDs of a Ryzen 9,
//! the CCXs of Zen 2), ONNX Runtime's intra-op threads and the MoE kernel's pool spread over two of them, and a
//! conversion's activations cross the I/O die between its ops. `pin_to_one_l3` pins the process to one logical CPU
//! of each physical core of one L3 domain: the one with the largest L3 (an X3D part's V-cache CCD), else the first.
//! On a Ryzen 9 9950X3D at 4 intra-op threads, Shiran-v2 int4 over AJIMEE's whole readings, p50 / p90: unpinned
//! 6.0 / 11.0 ms, pinned so 5.0 / 10.6 (the readings live conversion sends 6.0 -> 4.7); with other work on the
//! machine, unpinned 8.9 / 17.6, the V-cache CCD's 8 cores 5.4 / 10.5, its 16 logical CPUs 5.9 / 17.6, the other
//! CCD 10.6 / 21.4. (The reranker's 16 llama.cpp threads want every core: pinned to 8 they took 48 -> 89 ms.)
//! Off unless `SETSUNA_CPU_AFFINITY=1`: an IME's conversion runs on one thread, which the OS places.

/// The logical CPUs that share one L3, and its size.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Domain {
    pub l3_bytes: u64,
    pub cpus: Vec<usize>,
}

/// The CPUs to pin to: of the domain with the largest L3 (the first of equal ones, `domains` in the order of their
/// lowest CPU), the lowest allowed logical CPU of each physical core (`cores`, each core's logical CPUs). None with
/// fewer than two domains, or no allowed CPU in the chosen one.
pub fn choose(domains: &[Domain], cores: &[Vec<usize>], allowed: &[usize]) -> Option<Vec<usize>> {
    let (first, rest) = domains.split_first()?;
    if rest.is_empty() {
        return None;
    }
    let best = rest
        .iter()
        .fold(first, |b, d| if d.l3_bytes > b.l3_bytes { d } else { b });
    let mut cpus: Vec<usize> = cores
        .iter()
        .filter_map(|core| {
            core.iter()
                .copied()
                .filter(|c| best.cpus.contains(c) && allowed.contains(c))
                .min()
        })
        .collect();
    cpus.sort_unstable();
    (!cpus.is_empty()).then_some(cpus)
}

/// An AMD CPU of family 17h or later (Zen and its successors).
#[cfg(target_arch = "x86_64")]
fn is_zen() -> bool {
    use std::arch::x86_64::__cpuid;
    // cpuid is always there on x86-64 (`__cpuid` is unsafe on older toolchains)
    #[allow(unused_unsafe)]
    let (v, f) = unsafe { (__cpuid(0), __cpuid(1).eax) };
    let vendor = [v.ebx, v.edx, v.ecx].map(u32::to_le_bytes).concat();
    let base = (f >> 8) & 0xF;
    let family = if base == 0xF {
        base + ((f >> 20) & 0xFF)
    } else {
        base
    };
    vendor == b"AuthenticAMD" && family >= 0x17
}

#[cfg(not(target_arch = "x86_64"))]
fn is_zen() -> bool {
    false
}

/// Pin this process to one L3 domain (`choose`) on a Zen CPU with several; returns the CPUs pinned to, None where
/// it did nothing. Call it before any thread is started: on Linux later threads inherit the affinity, earlier ones
/// keep theirs.
pub fn pin_to_one_l3() -> Option<Vec<usize>> {
    if !std::env::var("SETSUNA_CPU_AFFINITY").is_ok_and(|v| v == "1") || !is_zen() {
        return None;
    }
    let (domains, cores, allowed) = sys::topology()?;
    let cpus = choose(&domains, &cores, &allowed)?;
    sys::pin(&cpus).then_some(cpus)
}

/// "0-3,8,10-11" as its CPUs.
#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
fn parse_cpu_list(s: &str) -> Option<Vec<usize>> {
    let mut out = Vec::new();
    for part in s.trim().split(',').filter(|p| !p.is_empty()) {
        match part.split_once('-') {
            Some((a, b)) => out.extend(a.parse::<usize>().ok()?..=b.parse().ok()?),
            None => out.push(part.parse().ok()?),
        }
    }
    Some(out)
}

/// A sysfs cache size ("32768K", "96M") in bytes.
#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
fn parse_size(s: &str) -> Option<u64> {
    let s = s.trim();
    let (n, unit) = match s.char_indices().last()? {
        (i, 'K') => (&s[..i], 1 << 10),
        (i, 'M') => (&s[..i], 1 << 20),
        (i, 'G') => (&s[..i], 1 << 30),
        _ => (s, 1),
    };
    Some(n.parse::<u64>().ok()? * unit)
}

#[cfg(windows)]
mod sys {
    use super::Domain;
    use windows_sys::Win32::System::SystemInformation::{
        GetLogicalProcessorInformationEx, RelationAll, RelationCache, RelationProcessorCore,
        GROUP_AFFINITY, SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX,
    };
    use windows_sys::Win32::System::Threading::{
        GetCurrentProcess, GetProcessAffinityMask, SetProcessAffinityMask,
    };

    fn bits(mask: usize) -> impl Iterator<Item = usize> {
        (0..usize::BITS as usize).filter(move |i| mask >> i & 1 == 1)
    }

    /// The CPUs of `n` group masks starting at `first`; None past processor group 0 (a process's affinity mask
    /// reaches group 0 only, and a CPU of 64 logical processors or fewer has no other).
    ///
    /// # Safety
    /// `first` must point at `n` (or, when `n` is 0, one) group masks.
    unsafe fn group0(first: *const GROUP_AFFINITY, n: u16) -> Option<Vec<usize>> {
        let masks = unsafe { std::slice::from_raw_parts(first, n.max(1) as usize) };
        let mut cpus = Vec::new();
        for m in masks {
            if m.Group != 0 {
                return None;
            }
            cpus.extend(bits(m.Mask));
        }
        Some(cpus)
    }

    pub fn topology() -> Option<(Vec<Domain>, Vec<Vec<usize>>, Vec<usize>)> {
        let mut len = 0u32;
        unsafe { GetLogicalProcessorInformationEx(RelationAll, std::ptr::null_mut(), &mut len) };
        // u64s: the records' alignment
        let mut buf = vec![0u64; (len as usize).div_ceil(8)];
        let base = buf.as_mut_ptr() as *mut u8;
        if unsafe { GetLogicalProcessorInformationEx(RelationAll, base.cast(), &mut len) } == 0 {
            return None;
        }
        let (mut domains, mut cores) = (Vec::new(), Vec::new());
        let mut off = 0usize;
        while off + std::mem::size_of::<u32>() * 2 <= len as usize {
            let rec =
                unsafe { &*(base.add(off) as *const SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX) };
            if rec.Size == 0 {
                break;
            }
            if rec.Relationship == RelationCache {
                let c = unsafe { &rec.Anonymous.Cache };
                if c.Level == 3 {
                    let cpus = unsafe { group0(c.Anonymous.GroupMasks.as_ptr(), c.GroupCount) }?;
                    domains.push(Domain {
                        l3_bytes: c.CacheSize as u64,
                        cpus,
                    });
                }
            } else if rec.Relationship == RelationProcessorCore {
                let p = unsafe { &rec.Anonymous.Processor };
                cores.push(unsafe { group0(p.GroupMask.as_ptr(), p.GroupCount) }?);
            }
            off += rec.Size as usize;
        }
        domains.sort_by_key(|d| d.cpus.first().copied());
        let (mut process, mut system) = (0usize, 0usize);
        if unsafe { GetProcessAffinityMask(GetCurrentProcess(), &mut process, &mut system) } == 0 {
            return None;
        }
        Some((domains, cores, bits(process).collect()))
    }

    pub fn pin(cpus: &[usize]) -> bool {
        let mask = cpus.iter().fold(0usize, |m, &c| m | 1 << c);
        unsafe { SetProcessAffinityMask(GetCurrentProcess(), mask) != 0 }
    }
}

#[cfg(target_os = "linux")]
mod sys {
    use super::{parse_cpu_list, parse_size, Domain};
    use std::collections::BTreeSet;
    use std::fs::read_to_string;

    const CPU: &str = "/sys/devices/system/cpu";

    pub fn topology() -> Option<(Vec<Domain>, Vec<Vec<usize>>, Vec<usize>)> {
        let online = parse_cpu_list(&read_to_string(format!("{CPU}/online")).ok()?)?;
        let (mut l3, mut cores) = (BTreeSet::new(), BTreeSet::new());
        for cpu in &online {
            let dir = format!("{CPU}/cpu{cpu}");
            cores.insert(parse_cpu_list(
                &read_to_string(format!("{dir}/topology/thread_siblings_list")).ok()?,
            )?);
            for index in 0.. {
                let cache = format!("{dir}/cache/index{index}");
                let Ok(level) = read_to_string(format!("{cache}/level")) else {
                    break;
                };
                if level.trim() == "3" {
                    let cpus =
                        parse_cpu_list(&read_to_string(format!("{cache}/shared_cpu_list")).ok()?)?;
                    let size = parse_size(&read_to_string(format!("{cache}/size")).ok()?)?;
                    l3.insert((cpus, size));
                }
            }
        }
        let mut domains: Vec<Domain> = l3
            .into_iter()
            .map(|(cpus, l3_bytes)| Domain { l3_bytes, cpus })
            .collect();
        domains.sort_by_key(|d| d.cpus.first().copied());
        let allowed = unsafe {
            let mut set: libc::cpu_set_t = std::mem::zeroed();
            if libc::sched_getaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &mut set) != 0 {
                return None;
            }
            online
                .iter()
                .copied()
                .filter(|&c| libc::CPU_ISSET(c, &set))
                .collect()
        };
        Some((domains, cores.into_iter().collect(), allowed))
    }

    pub fn pin(cpus: &[usize]) -> bool {
        unsafe {
            let mut set: libc::cpu_set_t = std::mem::zeroed();
            for &c in cpus {
                libc::CPU_SET(c, &mut set);
            }
            libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &set) == 0
        }
    }
}

#[cfg(not(any(windows, target_os = "linux")))]
mod sys {
    use super::Domain;

    pub fn topology() -> Option<(Vec<Domain>, Vec<Vec<usize>>, Vec<usize>)> {
        None
    }

    pub fn pin(_: &[usize]) -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn domain(l3_mb: u64, cpus: std::ops::Range<usize>) -> Domain {
        Domain {
            l3_bytes: l3_mb << 20,
            cpus: cpus.collect(),
        }
    }

    /// Logical CPUs 2c and 2c + 1 on core c, as Windows numbers them.
    fn smt_cores(n: usize) -> Vec<Vec<usize>> {
        (0..n).map(|c| vec![2 * c, 2 * c + 1]).collect()
    }

    #[test]
    fn the_v_cache_ccd_one_cpu_a_core() {
        // a 9950X3D: CCD0 with 96 MB of L3 on CPUs 0-15, CCD1 with 32 MB on 16-31
        let domains = [domain(96, 0..16), domain(32, 16..32)];
        let all: Vec<usize> = (0..32).collect();
        assert_eq!(
            choose(&domains, &smt_cores(16), &all),
            Some(vec![0, 2, 4, 6, 8, 10, 12, 14])
        );
        // the V-cache on the second CCD (a 7950X3D's numbering can differ)
        let domains = [domain(32, 0..16), domain(96, 16..32)];
        assert_eq!(
            choose(&domains, &smt_cores(16), &all),
            Some(vec![16, 18, 20, 22, 24, 26, 28, 30])
        );
    }

    #[test]
    fn equal_l3s_take_the_first_and_one_l3_is_left_alone() {
        let all: Vec<usize> = (0..32).collect();
        let domains = [domain(32, 0..16), domain(32, 16..32)];
        assert_eq!(choose(&domains, &smt_cores(16), &all).unwrap()[0], 0);
        assert_eq!(choose(&[domain(32, 0..16)], &smt_cores(8), &all), None);
        assert_eq!(choose(&[], &[], &all), None);
    }

    #[test]
    fn only_allowed_cpus() {
        let domains = [domain(96, 0..16), domain(32, 16..32)];
        // a core's sibling stands in for its first CPU; none allowed in the domain, nothing to pin
        assert_eq!(
            choose(&domains, &smt_cores(16), &[1, 3, 20]),
            Some(vec![1, 3])
        );
        assert_eq!(choose(&domains, &smt_cores(16), &[20, 21]), None);
        // without SMT
        let cores: Vec<Vec<usize>> = (0..32).map(|c| vec![c]).collect();
        assert_eq!(choose(&domains, &cores, &[0, 1, 2]), Some(vec![0, 1, 2]));
    }

    #[test]
    fn sysfs_lists_and_sizes() {
        assert_eq!(
            parse_cpu_list("0-3,8,10-11\n"),
            Some(vec![0, 1, 2, 3, 8, 10, 11])
        );
        assert_eq!(parse_cpu_list("5"), Some(vec![5]));
        assert_eq!(parse_cpu_list("x"), None);
        assert_eq!(parse_size("32768K\n"), Some(32 << 20));
        assert_eq!(parse_size("96M"), Some(96 << 20));
        assert_eq!(parse_size("512"), Some(512));
    }

    #[test]
    #[ignore]
    /// This machine's L3 domains and what the engine would pin to:
    /// cargo test -p setsuna-engine --lib cpu::tests::this_machine -- --ignored --nocapture
    fn this_machine() {
        let t = sys::topology();
        eprintln!("zen {} topology {t:?}", is_zen());
        if let Some((d, c, a)) = t {
            eprintln!("choose {:?}", choose(&d, &c, &a));
        }
    }
}
