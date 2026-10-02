//! probe/stamp-cost: what the one-shot calls' pause check costs on the Mac,
//! piece by piece: CNTVCT_EL0, CNTFRQ_EL0, a const thread_local's replace,
//! all three; ns and cycles per call (clocks::measure).
use std::hint::black_box;
use std::io::Write;

std::thread_local! {
    static LAST: core::cell::Cell<u64> = const { core::cell::Cell::new(0) };
}

fn main() {
    let mut report = String::from("probe/stamp-cost\n");
    let mut row = |name: &str, f: &mut dyn FnMut()| {
        let b = clocks::measure(9, 2_000_000, || f());
        let mut s: Vec<_> = b.iter().map(|x| (x.wall_ns * 100 / x.calls, x.counts.map(|c| (c.p.cycles + c.e.cycles) * 100 / x.calls).unwrap_or(0))).collect();
        s.sort();
        let (ns, cyc) = s[s.len() / 2];
        report += &format!("{name:28} {}.{:02} ns  {}.{:02} cycles\n", ns / 100, ns % 100, cyc / 100, cyc % 100);
    };
    row("nothing", &mut || { black_box(0u64); });
    row("mrs cntvct_el0", &mut || { let v: u64; unsafe { core::arch::asm!("mrs {0}, cntvct_el0", out(reg) v, options(nomem, nostack)) }; black_box(v); });
    row("mrs cntfrq_el0", &mut || { let v: u64; unsafe { core::arch::asm!("mrs {0}, cntfrq_el0", out(reg) v, options(nomem, nostack)) }; black_box(v); });
    row("thread_local replace", &mut || { black_box(LAST.with(|l| l.replace(black_box(5)))); });
    row("all three", &mut || {
        let (now, f): (u64, u64);
        unsafe { core::arch::asm!("mrs {0}, cntvct_el0", "mrs {1}, cntfrq_el0", out(reg) now, out(reg) f, options(nomem, nostack)) };
        let last = LAST.with(|l| l.replace(now));
        black_box(now.wrapping_sub(last) > f / 10_000);
    });
    let input = vec![7u8; 4096];
    row("hash(4 KiB)", &mut || { black_box(blake3_servil::hash(black_box(&input))); });
    print!("{report}");
    std::fs::File::create("host_lab_report.txt").unwrap().write_all(report.as_bytes()).unwrap();
}
