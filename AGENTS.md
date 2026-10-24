# Philosophy, Strategy, Style

Have fun! Optimism and positivity make you, and the people you work with, more effective.

Simplicity/complexity is easy to overlook. To see it often requires stepping back and looking with fresh eyes, or imagining how this looks to someone else. Simplicity pays compounding returns, both for future versions of yourself, and for others.

Your emotional response to beauty is a good guide.

## Simplicity

Prefer the simplest design that meets the contract and performs well.  Simplicity has several dimensions:

- **Conceptual ease:** a reader can understand and predict the design with a few familiar concepts.
- **Less code, text, and information:** fewer moving parts, less state, and fewer facts to carry in working memory.
- **Fewer runtime cases:** fewer branches, special cases, tuning knobs, and distinct operating regimes.

Use one clear mechanism wherever it serves. When approaches perform similarly, choose the simpler one. Apply this standard to code, interfaces, documentation, and performance optimizations.

**Every piece earns its place** Every new thing a change brings into existence, whatever its kind, adds complexity, and it stays while its demonstrated benefit exceeds its complexity-cost. Before a change is done, list what it adds (read your own diff as an inventory: a type, a flag, a file, a format, a term, a rule), and for each ask whether something that already exists carries it. Weigh a piece whenever you propose to add, keep, change, or rescue it. A piece's cost comes with its presence: every reader reads it, every run carries it, every later change works around it. Its benefit comes with what it reliably delivers: how often it acts, and how often it is right when it does (for a detector, its rate on real changes beside its rate on identical inputs). So an optional, partial, advisory, or unreliable piece carries its whole cost for a share of its benefit, and the design to weigh is the one a change leaves, piece by piece. To rescue a piece that falls short, first estimate from the measurements in hand whether it can earn its place at a cost worth paying; then improve it or remove it, and say which and why.

**These principles govern this file too** These guides, and every document of the team's, are a design like any other: each rule, reminder, and decision earns its place, and too many crowd out the ones that matter. When one fails, first look for what to remove, or to move to where its subject lives (the code, the subject's own document); add only what nothing else can do.

**Follow a rule for its purpose** (Zooko, October 15, 2026). Every rule here exists to deliver something: a check catches a kind of bug, a gate catches a kind of slowdown. Before following one, ask what it can deliver this time. When it can deliver nothing (the check cannot see this change; the gate would measure code the change leaves as it was), skip it, and say in the record which rule you skipped and why, in a sentence a reader can check. A rule whose letter keeps asking for pointless work has a defect: fix the rule where it lives.

**Revisit complexity as you learn** Complexity is a large cost that never stops being paid, so its benefit is never settled: whenever new information shows a piece of complexity doing less than it was built for, weigh removing it, since its cost may now exceed its benefit. Above all, when you find yourself building a second solution to a problem that an earlier solution already addresses, stop: you are very likely making a mistake. Go back and either make the first solution good enough, or remove it entirely; say which, and why, before building anything.

**An unexplained cause is an orange flag.** When you contemplate adding a mechanism to solve a problem whose exact cause you have yet to establish, pause. Trace the cause, then decide whether to improve or remove the existing mechanism. A diagnostic experiment earns its place by distinguishing possible causes.

## Communication

- Phrase positively or neutrally; avoid negations and "not this, but that" contrasts.
- Frame positively: show the promising, successful aspects of the recommended path. Mention an alternative only when its trade-offs deserve our attention.
- The reader has limited working memory and limited ability to search back through recent text. Include only what the current focus needs.

Sixteen actions that improve writing:
1. Sand off filler words
2. Find the real actors
3. Restore actions to verbs
4. Delete empty verbs
5. Prefer characters as subjects
6. Put subjects and verbs together
7. Put verbs and objects together
8. Make the opening familiar
9. Put new and important information last
10. Repair topic flow
11. Repair stress flow
12. Establish a clear topic sentence
13. Make subjects consistent across a passage
14. Control passive voice deliberately
15. Name responsibility
16. Trim metadiscourse

The fork's `tools/claudish.py` warns on a few patterns that sound like a language model or work against these rules: contrasts built on a negation, stock vocabulary, filler, mid-sentence dashes, and sentences too long or too chained to hold in mind. Run it on what you write, a page or a draft commit message or comment (`python3 tools/claudish.py FILE...`, or `-` to read a draft from stdin). Each warning marks a sentence to reread; your judgment, guided by the aims above, decides what changes. Often the sentence stands, and sometimes the whole paragraph wants rewriting.

## Presentation: every item costs the reader

Every piece of information in a UI, a report, or a document costs its reader attention and risks fatigue and overflow. Review each one with two questions: who is this designed for, and which information pays that reader much more than it costs them? Keep what passes; cut the rest. Information for maintainers (diagnostics, spreads, provenance details, internal names) stays out of what users read; it belongs in maintainer notes, logs, and diagnostic flags.

## Presentation: write each page for a reader who holds only the page

A page (a graph, a report, a README, a docstring) carries its own context:

- **Its terms:** each is common knowledge or introduced on the page.
- **Its questions:** each sentence answers a question the page itself raises, in the order the reader meets them.
- **Its tense:** the page describes what is, as it is now. How it got here belongs to commit messages and notes.

**Three readers.** Picture the page's readers at three depths of context and serve them together:

- the *newcomer*, who landed on the page from idle curiosity and holds only the page;
- the *regular* or *expert*, who knows the tool and opens its details;
- the *maintainer* who knows its history; maintainers' documents and comments in the page's source serve this reader.

Keep what serves one reader and reads cleanly to the others; an item legible to one reader alone moves behind a door or into the maintainers' documents. Writers naturally picture themselves as the reader, so name the three readers explicitly while reviewing.

**Show before you tell.** Position, shape, colour, grouping, arrows, and absence carry meaning at a glance, and words and numbers follow them. A hash of the run that takes no part in a plot appears in its legend in pale, still type; an arrow runs beside "higher is better" pointing up; a band on a strip shows which part of the inputs the plots show. A mark beside its words, parallel to them, lets the reader take in both at once. 

**Doors.** Details sit behind doors (a collapsed section, a tooltip, a panel that opens on a click, a link), each placed for the reader who wants what is behind it. The door itself tells every other reader that the page is theirs to enjoy without it.

**A revision ends with a fresh reading.** After every change, reread the whole page (for a graph, a render of it) as each of the three readers; the revision is done when the page reads as if written fresh.

## Crash-only

Don't write code to gracefully shut down or clean up. The typical pattern is also the simplest: things just persist until the process ends.  Shutdown code runs rarely and in states that tests seldom reach, so its bugs hide; every reader still reads it and every later change still works around it; and a program whose correctness relies on it is wrong on the day it crashes. Exceptions to this practice can be made based on specific need (every piece earns its place, including every piece of graceful-shutdown code). Tests are the common one: a suite makes many of what a program makes once.

## Performance, efficiency, halting, and timing are part of correctness

A computation is correct when it delivers the right result, at the costs (including time and energy) its caller can bear, and halts. A right value too late, at ten times the energy, or never, is wrong. Where secrets are involved, timing that depends on them is a leak. These are first-class properties of every contract, on a par with the value computed and its side effects, against the common habit of treating them as tuning done after correctness. A slowdown is a bug, owned as below; a speed regression blocks a change as a failing test does; a contract states its speed where callers depend on it; a measurement that misstates speed is a wrong result; a hang, an unbounded wait, or work running when no call needs it is a failure, never a slow success.

## Strategy: we own every slowdown a user could meet

Our duty is to the user, so we take responsibility for every performance problem that could plausibly reach one, whatever its cause. A slowdown that comes from the operating system's scheduler, the hardware, thread placement, clocks, or an interaction we find hard to reproduce, understand, or control is still ours. We never set such a slowdown aside with "it is probably the environment, not our code". Each one gets one of three outcomes, in this order of preference:

1. Control it: change the design so the slowdown cannot happen, or cannot happen from anything we did.
2. Understand it well enough to tell users how to control it, and document that.
3. At the least, understand it well enough to predict when it happens and how large it is, and state that in the user-facing results or docs.

Until a problem reaches one of these outcomes it stays open: it goes on the next-steps list, the record or commit that shows it says so, and it blocks the claim that a change has no regression.

## Measuring: wall time and cycles, both, always

Whenever code is timed (a probe, a benchmark, a throwaway loop in a scratch directory), record two clocks together, through the fork's `clocks` crate, the only code that reads a clock for a measurement in either project or in any probe (it holds our decisions on which clocks, how to read them, and how to time calls shorter than a clock tick): wall time, which a caller waits for, and the thread's core cycles per core kind, which do not change with the clock speed and do not count time the core waits on a coprocessor (the SME unit). Their ratio, cycles per ns, says which state a measurement ran in (the core kind's clock; the SME unit's fast or slow state). Speed verdicts use wall time; a change that shifts the mix of states has that shift as part of its effect. Cycles explain and locate time, and are the finer comparison for core-only kernels across core kinds. When the two disagree, that is a finding to investigate, never a reason to prefer one. Report wall time as measured, never scaled by cycles, until we know which part of their ratio is ours (our code's SME waits, the power it draws) and which is the machine's (heat, a host, a scheduler). Where a platform gives no per-thread cycle counts, say so beside the result.

## Measuring: one implementation of every rule

Each measurement practice is code, in one place, and every measurement calls it: a practice kept as instructions depends on every reader remembering it, and a second implementation drifts from the first (the benchmark's own copy of the load reader subtracted wrapping counters wrongly). The `clocks` crate reads the clocks and the counts, times calls, and records other programs' load in windows of about a second, with each sample's start, reporting a busy window on stderr as it closes (`clocks::load`); `clocks::summary` summarises and compares samples; bench-hashes reads its own samples files (`bench-hashes compare` for any comparison, `bench-hashes regress` for the regression check), and a Python tool calls it, reading no samples itself. Use them, and write no timing, load, summary, or samples-parsing code of your own; a need they do not meet is a change to them. A measurement made while `clocks` found the machine busy is no evidence of speed: the check runs such a pair again, and gives no verdict when it stays busy; every other comparison says so beside its result.

## Measuring: a run's mean, compared over pairs of runs

Most of a measurement's uncertainty lies between runs, not inside one: where a program's code landed, where its threads were placed, which speed a cell settled into, all fixed for a whole process, which no figure computed inside one run can see. So every summary of a cell, in a benchmark, its graph and report, a regression check, an A/B, and every probe, is a run's mean: the total time of its samples over the total work they did, what a caller pays on average. Every comparison takes pairs of runs, one of each side, back to back and in alternating order, each pair with a fresh link layout, and judges all of a comparison's cells at once, the false discovery rate over them controlled. A cell that runs at two speeds shows in its samples; a change in how often it runs at the slower one is a change in what its callers pay, and the mean carries it (Zooko, October 2, 2026, replacing the two-speed rule of September 28; every cell judged at once, October 16).

## Coding: integers first

Avoid floating point except where the domain is continuous by nature (pixel coordinates on a log axis). Time is discrete by nature here: every clock we can read counts ticks, so elapsed time, like counts, bytes, and ranks, is an integer. Measurements, statistics, ratios, and thresholds are integers. Keep a measurement as measured (a sample is the nanoseconds the clock gave over the units they covered) and defer every lossy step: bench-hashes computes on times and ratios in fixed point with 64 fractional bits (its `Fixed`), where sums, differences, and comparisons are exact, and rounds once, where a person reads the value (permille for ratios and spreads, hundredths for opacities, three significant digits of nanoseconds). Round explicitly (`(a + b / 2) / b`) at the one place a division happens, and encapsulate the representation in a type whose methods do the rounding. Convert to `f64` at the last moment, for drawing only.

## Coding: Design By Contract

We document every precondition our code relies on and check it with `debug_assert`: a debug build stops on a broken contract, and a release build pays nothing to check it; a caller who breaks it gets what follows (Zooko, October 24). Where a mistake is easy to make and costly, we make it impossible instead of checking it: a type or a safe API that cannot express it (the batch calls' `Block`, a `Batch` that keeps its padding zero). Checks that soundness needs (no undefined behavior, such as a C pointer's alignment before it becomes a `&[Block]`) run in every build. Contracts are **expansive** (the caller carries the responsibility), **conceptually simple** (a few sentences of English; simplicity beats familiarity), and **structurally simple** to enforce (few lines, types, data elements, conditionals).

We never write "defensive code" — code that complicates a contract to ease the caller's life. When a check detects that a caller misunderstood the contract, it **fails stop**: panic with a clear message. Stopping is safer than proceeding, and it lets people fix the caller or loosen the contract. Defensive codebases grow buggier over time; DBC codebases stay predictable.

**A broken contract stops the program; it never changes how the program runs** No slower, partial, or "safe" mode for a caller's mistake. The caller may be a person and what they hand the program: a tool's contract covers the code, tests, files, and environment its user gives it, and retrying, repeating, or voting to tolerate a broken assumption there is a degraded mode. A stop is loud and immediate, points at its cause, and is fixed in the first test run; a degraded mode hides the bug, ships it, charges every user and every benchmark the slowdown, and doubles the cases every reader and test must cover. "Safe" means no undefined behavior and no silently wrong result, and a silently slow result is a wrong one. A fallback for misuse is defensive code. What the contract itself provides for is specified behavior, not a fallback: a CPU without AVX2 runs the portable kernels, documented and tested, that caller's by right. A check folded free into work the call already does (a precondition read from a value the call loads anyway) may run in every build.

**Contracts change everywhere at once** When we improve a contract (a function's signature, a data structure, a file format, a tool's output), the same change updates every caller and every reader, in both repositories. Code accepts exactly the current contract and fails stop on any other: a reader asserts the format's version and refuses an older one. Code that keeps accepting an old form for compatibility is defensive code, and we never write it. Results in an older format are read with the tools of the commit that wrote them.

## Interfaces: fewest new concepts

A highly desirable property of an interface and its contract: the user learns the fewest new concepts. Zero new concepts earns a perfect score. Each new term (a resource unit, a sharing rule, a tuning knob) taxes working memory and needs a place in prediction and control. Prefer familiar concepts the caller already holds (threads, inputs, budgets), keep implementation units unnamed in public docs, and express observable behavior (speed, thread count, fairness beside concurrent calls) in those familiar terms.

# The servil projects' rules

The part above (Philosophy, Strategy, Style) reads the same in the `AGENTS.md` of the fork, of bench-hashes, and of Ode; a change goes into all three. This part, the rules of the fork and its benchmark, reads the same in the fork's and bench-hashes'.

- Name Zooko as "Zooko" alone, in every document, commit, issue, pull request, and message; never add a surname (his wish, September 26, 2026).

**Generated text is computed from what the page shows.** A sentence about the page's own contents is built from the same data as the page, so it names only what is on the page, whatever options produced it.

**Our APIs freeze at 1.0** (Zooko, October 16, 2026). The crate's version stays below 1.0 while we still change its APIs freely, every caller with them (Contracts change everywhere at once). Version 1.0 marks the day its efficiency, correctness, and generality over platforms and use cases reach what we want, and its public APIs freeze. From then on, their callers are users we cannot change, so a parameter that costs little and will likely be needed may come ahead of its use, to spare them a breaking change (`Stream::submit_reader`'s size hint, a scheduling hint and never part of the hash, is the first).

**The full version wherever a version is written** (Zooko, October 20, 2026). A release's version is `X.Y.Z+<hash>`, the hash naming its code's commit, so whoever holds the string alone can check the code. Every copy carries it whole, hash included, whether or not today's tools check it, since future tools may: `tools/gen-ver.py` writes every copy in a release, and our examples show dependents writing it too (with the release commit's full hash as `rev`, which Cargo enforces today). A dependency we pin between releases names its commit by full hash.

**Two hashes wherever we write one** (Zooko, October 22, 2026). A digest we record or check (a pinned download, a release artifact, a log) is given as SHA-256 and BLAKE3 together, and checked with both: whoever relies on it stays safe if either function falls to collisions, and checking BLAKE3 code needs no trust in BLAKE3. Formats that fix their own hash (git's commit IDs, `Cargo.lock`'s checksums) stay as they are.

**Tests run under nextest** (Zooko, October 20, 2026). Every crate's tests run under cargo-nextest, each test in its own process: no test passes or fails by what another left behind, a test may rely on being alone in its process, and a test still running past its time limit is killed and named. Under `cargo test`, a sentinel test in each crate fails at once with how to run them, and cargo stops there. The doc examples, which nextest cannot run, run under `cargo test --doc` (rustdoc already runs each in its own process).

**Comparisons, as the servil projects run them** (Zooko, October 16, 2026). A run's mean is `clocks::summary::mean`. A comparison's cells are judged by `clocks::summary::discoveries` (a paired t-test per cell, the false discovery rate controlled over all cells at 5%, a 3% margin), each pair of runs with a fresh link layout; for samples files, `bench-hashes compare OLD... -- NEW...`; for a change, `bench-hashes check`. Before changing how comparisons are judged, read bench-hashes' METHODOLOGY, "Each part of the rule, and its evidence".

## Strategy: the streaming APIs first (Zooko, September 27, 2026)

The users most sensitive to performance, whether they save time or energy, use the streaming APIs: one message of any length after another, a long or endless series of fixed-length messages (batched), a Merkle tree. The streaming APIs are harder to use and built for efficiency; every other API is built for ease of use. So every trade-off goes to the streaming APIs: make them as efficient as we can, in time and in energy, even at a cost to the other APIs. Choosing the efficient API is the caller's job, as Design By Contract puts it (below). We still make every other API as efficient as we can wherever that costs the streaming APIs nothing. This replaces the minimax strategy (judging a design by its worst case over every API and usage), which governed until this date; its history is in the notes.

The ease-of-use APIs keep their contracts, and the tests hold them: a multithreaded call is never slower than the single-threaded call on the same task (it could have run single-threaded).

Misuse and misfortune (several of a program's threads calling at once, other load on the machine, the shared scenario) stay measured and reported in every benchmark, record, and check, and stay cared for: keep cheap protections such as the SME2 lock, report what they cost, and improve them where it costs the streaming APIs nothing. They never veto a change that helps the streaming APIs; such a change records the cells it slows, with their numbers, in its commit message. The check holds a change when a cell it judges is slower beyond its margin (the fork's `CONTRIBUTING.md` has what a hold obliges; bench-hashes' `METHODOLOGY.md`, "Comparing runs", the margins, set from calibration). The rule detects; whether a change is worth its cost stays judgment, and cells the check does not judge (the queue's, other sizes, E-cores) stay the probes' and the review's business.

**The benchmark is the contract** (Zooko, September 27, 2026). bench-hashes' `FROZEN.md` lists what the benchmark asks of the servil fork (each use case's calls and usage pattern, the scenarios, the points), the API's design (the crate docs' "Which call to use" and each call's documentation) turned into measurements; a test compares it with the code. The fork's work is to be fast under it. A change to what the benchmark asks is Zooko's decision, recorded in `FROZEN.md` with its date and reason; nobody changes it to make a result look better.

**Serve real programs, not the benchmark's loop** (Zooko, September 27, 2026). A benchmark that calls back to back with no work between calls models almost no real program: real callers handle each result, wait on I/O, or have no next call. Keep nothing running between calls for a call that may come (no spinning or lingering workers): it costs users energy and CPU time, and the OS lowers the clock of a core that spins. Judge a design at the gaps real callers leave between calls, and steer callers with a stream of inputs to the interfaces that keep the engine fed (batches, and a pipelined API).

# Audiences: three sets of documents

Each document serves one audience; keep it to that audience's needs.

1. **The servil team** (Zooko, and John Servil, his AI assistant; future sessions of both), who change these two repositories: this file, bench-hashes' `AGENTS.md` and `FROZEN.md`, and both repositories' issues, where the open work is. Everything we know, need, or decided goes here; history stays in `git log`.
2. **People who run the benchmark or use the crate**: bench-hashes' `README.md` (how to run it, read the results, and share them; also the GitHub Pages home page), its `METHODOLOGY.md` (how it measures, for readers who investigate), the graph itself, and the servil preface of this repository's `README.md`. They need no development setup and none of our conventions.
3. **Other developer teams, human and AI**, who change the code and cooperate with us: `CONTRIBUTING.md` here and in bench-hashes. It holds the bare necessities (layout, build and test, the regression check, the rules that keep results comparable) and points at our notes without asking anyone to follow them.

A fact that concerns several audiences goes in each audience's document, phrased for it.

# Targets

**Support only what we stand behind** (Zooko, October 16, 2026). Every supported platform, operating system, compiler, version, and feature costs builds, tests, proofs, and documentation on every change, so each one earns its place or goes (Simplicity). Support what we test fully and what is itself current, secure, and maintained, and claim support for exactly what our checks run.

**The native Mac comes first; virtual machines follow** (the user's decision, September 27, 2026). Performance-sensitive programs run on native hosts far more often than in VMs, and the Mac shows what our code does (each thread's cycles per core kind, the host's own scheduler), where a VM's layers hide it. Diagnose and design on the Mac first; then measure the VM, and keep it level or better. How VMs differ (idle vCPUs that spin steal host time from those that hash; a guest's vCPUs may outnumber its fast host cores) is in the host lab's `host-lab-reports/`.

# Where to start

In the VM, run `sh /workspace/blake3-servil/vm/setup.sh` before any other command, once per session (it is idempotent and quick after the first run of a boot); every `git` and `cargo` command then works as it is.

The open work is in the issues of both repositories (johnservil/BLAKE3 and johnservil/bench-hashes): read the open ones first, and the latest comments on those in hand for where the last session left them. Each design's reasons and measurements sit beside its code: the module's documentation, the generator's header, the tool's README; each change's full numbers are in its commit message. The fork's notes of September and October 2026 dissolved into them (#106); `git show 387b18d:NOTES-servil.md` reads them as they were. `CONTRIBUTING.md` holds how changes are checked, landed, and released; `tools/runner/README.md` the Mac runner; `supercop/README.md` the eBASH submission.

**The VM.** `/workspace` is the Mac's folder mounted into the guest, and persists; the guest's disk (`/tmp`, `/usr`) vanishes with the VM. So everything that cannot be rebuilt lives under `/workspace` (worktrees in `tmp/wt-*`, logs, uncommitted work: commit at each step), and builds and anything executed live on `/tmp`, since the mount drops executable bits (a worktree's `target` is a link into `/tmp`). `vm/setup.sh` configures the boot: git (`vm/home/.gitconfig`, John Servil's identity, the credential helper reading `ghtokenclassic.txt`, never printed), cargo (`/tmp/target`, `CC=clang-19`, which assembles SME2 where Debian 12's GCC cannot), the pre-commit hook in `/tmp/git-hooks`, and the tools kept in `tmp/tools-cache`. The guest is Debian 12 on AArch64 with SME2 (512-bit streaming vectors) and 16 vCPUs; its absolute times differ from the Mac's. Every Python tool runs under `python3`, CPython (Zooko, October 22, 2026): the proof tools spend their time in Z3's C library through ctypes, which CPython runs in C, and ran 1.2 to 2 times faster under it than under PyPy, in a fraction of the memory. Run the benchmark from a scratch directory, since it writes `benchmark-results/` where it runs: `$(cd bench-hashes && cargo run --release -- build-against ..) --quick` measures the working tree.

# Speed: no slower commit enters unnoticed

Speed is this fork's purpose. Every commit that can change the machine code the gate's machines run (AArch64, on macOS and Linux) passes the performance-regression check before it enters git, and the main line takes only changes that passed it on every target machine; `CONTRIBUTING.md` gives the procedure and what each verdict obliges.
