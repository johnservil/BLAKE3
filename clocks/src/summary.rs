//! The one way every measurement summarises and compares its samples, in
//! the fork, bench-hashes, and every probe.
//!
//! **A run's cost** for a cell is its mean: the total timed nanoseconds of
//! its samples over the total work they did ([`mean`]). That is what a
//! caller pays on average, it moves continuously when a cell spends more or
//! less of its time at a slower speed, and it takes no cut-offs.
//!
//! **Comparing two programs** takes pairs of runs, one of each, back to back
//! and in alternating order, so that drift between the runs falls on both
//! sides alike. Each pair gives one ratio per cell, new mean over old
//! ([`ratio_permille`]). Most of a measurement's uncertainty lies between
//! runs (where the program's code landed, where its threads were placed),
//! which no figure from inside one run can see; the pairs see it. A cell is
//! slower ([`verdict`]) when the median of its ratios exceeds 1 plus the
//! margin and an exact sign test says the new side was slower in more pairs
//! than chance would give, at one-sided p <= 5% ([`least_count`]): one
//! assumption alone, that a pair is as likely to come out either way when
//! nothing changed.
//!
//! **Values** are nanoseconds per unit in fixed point, 64 integer and 64
//! fractional bits (Q64.64, as bench-hashes' `Fixed`); ratios in permille;
//! integers throughout, rounded once where a person reads them.

/// The mean of samples given as (timed ns, units of work): total ns over
/// total units, in Q64.64 ns per unit, rounded to the nearest. Requires some
/// units, and a total under 2^64 ns.
pub fn mean(samples: impl IntoIterator<Item = (u64, u64)>) -> u128 {
    let (ns, units) = samples.into_iter().fold((0u128, 0u128), |(ns, units), (n, u)| (ns + u128::from(n), units + u128::from(u)));
    assert!(units > 0, "a mean needs some work done");
    assert!(ns < 1 << 64, "a run's samples total under 2^64 ns");
    ((ns << 64) + units / 2) / units
}

/// `new / old` in permille, rounded. Requires `old > 0`.
pub fn ratio_permille(new: u128, old: u128) -> u64 {
    assert!(old > 0, "a ratio needs a divisor");
    let (new, old) = (new >> 8, old >> 8); // headroom for the factor of 1000
    assert!(old > 0, "a ratio needs a divisor above 2^-56 ns per unit");
    u64::try_from((new * 1000 + old / 2) / old).expect("a ratio fits in u64")
}

/// The median of permille ratios (for an even count, the mean of the two
/// middle ones, rounded up). Requires some.
pub fn median_permille(ratios: &[u64]) -> u64 {
    assert!(!ratios.is_empty(), "a median needs a value");
    let mut sorted = ratios.to_vec();
    sorted.sort_unstable();
    let n = sorted.len();
    if n % 2 == 1 {
        sorted[n / 2]
    } else {
        (sorted[n / 2 - 1] + sorted[n / 2]).div_ceil(2)
    }
}

/// The least number of `pairs` that must agree for the sign test to call a
/// change at one-sided p <= 5%: the least k with P(X >= k) <= 1/20 for X
/// binomial(pairs, 1/2). `pairs + 1` (never) when even all of them cannot.
/// Requires `pairs` from 1 to 64.
pub fn least_count(pairs: usize) -> usize {
    assert!((1..=64).contains(&pairs), "a sign test over 1 to 64 pairs");
    let mut choose = vec![1u128; pairs + 1]; // C(pairs, j)
    for j in 1..=pairs {
        choose[j] = choose[j - 1] * (pairs - j + 1) as u128 / j as u128;
    }
    let total = 1u128 << pairs;
    (0..=pairs + 1).find(|&k| k <= pairs && choose[k..].iter().sum::<u128>() * 20 <= total).unwrap_or(pairs + 1)
}

/// What the pairs say of one cell.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Verdict {
    Slower,
    Faster,
    Level,
}

/// The ratios the pairs leave plausible for the cell's median ratio (new
/// over old, permille): the sign test's interval, the [`least_count`]-th
/// ratio from each end, about 90% two-sided (each side one-sided p <= 5%).
/// A median ratio outside it is one the sign test rejects. `None` when the
/// pairs are too few for the test to reject anything.
pub fn interval_permille(ratios: &[u64]) -> Option<(u64, u64)> {
    let n = ratios.len();
    let k = least_count(n);
    if k > n {
        return None;
    }
    let mut sorted = ratios.to_vec();
    sorted.sort_unstable();
    Some((sorted[n - k], sorted[k - 1]))
}

/// A cell's verdict from its pairs' ratios (new over old, permille): slower
/// when the median exceeds 1000 + `margin_permille` and the whole
/// [`interval_permille`] lies above 1000 (at least [`least_count`] pairs
/// slower); faster alike, below.
pub fn verdict(ratios: &[u64], margin_permille: u64) -> Verdict {
    let median = median_permille(ratios);
    match interval_permille(ratios) {
        Some((low, _)) if median > 1000 + margin_permille && low > 1000 => Verdict::Slower,
        Some((_, high)) if median + margin_permille < 1000 && high < 1000 => Verdict::Faster,
        _ => Verdict::Level,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_mean_is_total_time_over_total_work() {
        // 3 ns over 2 units and 7 ns over 2 units: 10 ns over 4, 2.5 ns a unit.
        assert_eq!(mean([(3, 2), (7, 2)]), 5 << 63);
        assert_eq!(ratio_permille(mean([(110, 100)]), mean([(100, 100)])), 1100);
    }

    /// The sign test's thresholds, by hand from the binomial: 6 pairs need
    /// all 6 (1/64 = 1.6%; 5 or more is 7/64 = 10.9%); 8 need 7 (9/256 =
    /// 3.5%; 6 or more, 37/256 = 14.5%); 10 need 9 (11/1024 = 1.1%; 8 or
    /// more, 56/1024 = 5.5%); 4 never suffice (1/16 = 6.25%).
    #[test]
    fn the_sign_tests_thresholds() {
        assert_eq!(least_count(4), 5);
        assert_eq!(least_count(6), 6);
        assert_eq!(least_count(8), 7);
        assert_eq!(least_count(10), 9);
    }

    #[test]
    fn verdicts() {
        // Seven of eight pairs slower, the median 5% up: slower at 3%.
        let ratios = [1050, 1060, 1040, 1055, 1048, 990, 1052, 1070];
        assert_eq!(median_permille(&ratios), 1051);
        assert_eq!(verdict(&ratios, 30), Verdict::Slower);
        // The same at a 10% margin: level.
        assert_eq!(verdict(&ratios, 100), Verdict::Level);
        // A median beyond the margin from two wild pairs, but only six of
        // eight above 1: level (the sign test).
        let wild = [1200, 1300, 1040, 1035, 1045, 1033, 970, 980];
        assert_eq!(verdict(&wild, 30), Verdict::Level);
        assert_eq!(verdict(&[940, 950, 960, 955, 945, 1010, 950, 948], 30), Verdict::Faster);
    }

    /// The interval is the sign test inverted: over every set of up to 9
    /// pairs from a few ratios, the verdict through it equals the counts'.
    #[test]
    fn the_interval_is_the_sign_test() {
        let values = [940u64, 990, 1000, 1010, 1060];
        for n in 1..=9usize {
            for code in 0..values.len().pow(n as u32) {
                let ratios: Vec<u64> = (0..n).map(|i| values[code / values.len().pow(i as u32) % values.len()]).collect();
                let k = least_count(n);
                let median = median_permille(&ratios);
                let counted = if median > 1030 && ratios.iter().filter(|&&r| r > 1000).count() >= k {
                    Verdict::Slower
                } else if median + 30 < 1000 && ratios.iter().filter(|&&r| r < 1000).count() >= k {
                    Verdict::Faster
                } else {
                    Verdict::Level
                };
                assert_eq!(verdict(&ratios, 30), counted, "{ratios:?}");
            }
        }
        assert_eq!(interval_permille(&[1010, 1020, 1030, 1040, 1050, 1060, 1070, 1080]), Some((1020, 1070)));
        assert_eq!(interval_permille(&[1000; 4]), None);
    }
}
