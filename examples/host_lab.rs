//! probe/gate-busy: other programs' load on the Mac, window by window, for
//! 90 seconds of this process sleeping (clocks::load's one-second windows).
fn main() {
    let began = clocks::now();
    while clocks::since_ns(began) < 90_000_000_000 {
        clocks::load::tick();
        std::thread::sleep(std::time::Duration::from_millis(50));
    }
    for w in clocks::load::windows() {
        println!("{:6} ms  other {:>5} milli-CPUs{}", w.start_ns / 1_000_000, w.other_milli_cpus, if w.busy() { "  BUSY" } else { "" });
    }
}
