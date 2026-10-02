//! Diagnostic workload outside bench-hashes' frozen 64-byte batch axis.
//! Raw samples use its v4 format and shared clocks/speed rules.
use std::fmt::Write;
use std::hint::black_box;

fn main() {
    blake3_servil::initialize();
    let source = std::env::args().nth(1).expect("supply the measured source commit/fingerprint");
    let mut rows = String::new();
    let mut trace = String::from("point,sample,calls,wall_ns,p_cycles,p_instructions,p_ns,e_cycles,e_instructions,e_ns\n");
    let mut point = 0usize;
    for len in [64usize, 256, 1024, 2048, 2047, 2049] {
        for count in [1usize, 3, 4, 6, 8, 12, 16, 24, 64, 128, 129, 1024] {
            // Different bytes per message: byte i = i % 251. Slots are
            // padded with zeros, as the public API requires.
            let stride = len.next_multiple_of(64);
            let mut allocation = vec![0u8; stride * count + 63];
            // Exercise unaligned addresses in the measured workload.
            let offset = point % 64;
            let input = &mut allocation[offset..offset + stride * count];
            for (m, slot) in input.chunks_exact_mut(stride).enumerate() {
                for (i, byte) in slot[..len].iter_mut().enumerate() { *byte = ((m * len + i) % 251) as u8; }
            }
            let mut output = vec![[0u8; 32]; count];
            let batches = clocks::measure(32, 2_000_000, || {
                blake3_servil::hash_many(black_box(input), len, black_box(&mut output));
                black_box(output.as_slice());
            });
            write!(rows, "blake3-servil-st\tsolo\tTwoChunkBatchProbe\t{count} x {len} B\tmsg\t").unwrap();
            for (i, batch) in batches.iter().enumerate() {
                if i > 0 { rows.push(','); }
                write!(rows, "{}/{}", batch.wall_ns, batch.calls * count as u64).unwrap();
                write!(trace, "{count} x {len} B,{i},{},{}", batch.calls, batch.wall_ns).unwrap();
                if let Some(c) = batch.counts {
                    writeln!(trace, ",{},{},{},{},{},{}", c.p.cycles, c.p.instructions, c.p.time_ns, c.e.cycles, c.e.instructions, c.e.time_ns).unwrap();
                } else { trace.push_str(",,,,,,\n"); }
            }
            rows.push('\t');
            for (i, batch) in batches.iter().enumerate() {
                if i > 0 { rows.push(','); }
                write!(rows, "{}", batch.started_ns / 1_000_000).unwrap();
            }
            rows.push('\n');
            point += 1;
        }
    }
    let windows = clocks::load::windows();
    println!("# bench-hashes samples v4\n# fork source: {source}\n# probe: x86_batch_probe, 32 samples of 2 ms, producer outside timing\n# clocks source: 8825450\n# target: generic\n# power: not reported by this OS\n# cycles: unavailable on Linux\n# load: {}", clocks::load::describe(&windows));
    print!("# load windows (start ms-end ms:other milli-CPUs:steal milli-CPUs): ");
    for (i, window) in windows.iter().enumerate() {
        if i > 0 { print!(","); }
        print!("{}-{}:{}:{}", window.start_ns / 1_000_000, window.end_ns / 1_000_000, window.other_milli_cpus, window.steal_milli_cpus);
    }
    println!("\ncontender\tscenario\tuse_case\tpoint\tunit\tns/units\tstart ms\n{rows}");
    std::fs::write("clocks.csv", trace).unwrap();
}
