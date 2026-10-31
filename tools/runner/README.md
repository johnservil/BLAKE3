# Mac benchmark runner

`runner.py` runs benchmark jobs on the Mac as the hidden standard account
`benchrunner`, with code from GitHub at the commits each job names.
Jobs go in the checkout's `runner/jobs/`; results come back in
`runner/results/<job>.<time>/`. Both folders stay out of git
(`.git/info/exclude`). The runner keeps its clones of the fork and of
bench-hashes, side by side, in `~benchrunner/checkouts/`, and moves
them to each job's commits, so Cargo rebuilds only what changed: a
benchmark job builds bench-hashes against the fork as it is, through
`bench-hashes build-against` (in a directory with its own `Cargo.lock`;
the committed lock is never written), and check jobs build their two
sides the same way under the fork's `target/check/`. Before each job it
removes anything untracked from both clones, so each reads as the commit
the job names. The job format is the docstring at the top of
`runner.py`.

## The account (once)

    pw="$(openssl rand -base64 32)"
    sudo sysadminctl -addUser benchrunner -fullName "Bench Runner" -password "$pw"
    sudo dscl . create /Users/benchrunner IsHidden 1
    sudo dscl . -create /Users/benchrunner UserShell /usr/bin/false
    sudo dscl . -passwd /Users/benchrunner "$(openssl rand -base64 32)"

Nobody can log in to it; `sudo -u benchrunner` runs commands as it.

## Setting up and starting

    sh ~/piplayground/blake3-servil/blake3-servil/tools/runner/setup-mac.sh

Run it as yourself; it asks for sudo when it needs it. It makes the
account's home private, installs Python 3 when absent, and gives the account
the same rustc as yours (matched by commit hash). It then makes the
exchange folders (`jobs/` yours and readable, `results/` the runner's and
readable) and lets the account pass through the folders above them. Last,
it copies `runner.py` to `/Users/Shared/bench-runner/` (yours, read-only to
the runner) and starts it. Steps already done are skipped, so the same
command starts the runner every time, including after a change to
`runner.py`. Ctrl-C stops the runner after the current job.

## Running a job

Write `runner/jobs/NNN-name.json` naming pushed commits (the runner
fetches branches alone: push a pull request's head as `candidate/*`
first); wait with `python3 tools/runner/wait_for.py NNN-name`; results land
in `runner/results/`. A job runs once per file name, so a changed job
takes a new number.

While a job that times runs (`benchmark`, `b3sum`, `example`, `check`),
nothing else should run on the Mac, the VM included, and Activity
Monitor stays closed (it keeps `sysmond` near a whole CPU busy: with it
open, 170 of 219 load windows of a run read busy, closed 5 of 214). The
Mac stays on mains power: on battery it runs more calls after a pause on
its efficiency cores (233 of 400 against 32), and every job's
`verdict.json` records the power state at its start and end; read it
before trusting a job. A `test` job (the fork's suites, natively) needs
no quiet machine.

A direct A/B is four `benchmark` jobs, old new new old, back to back,
read with `bench-hashes compare`; a calibration is one job with
`"repeat": N`. Mac-only measurements (cycles by core kind, QoS) go in a
`probe/<topic>` branch with its own `examples/host_lab.rs`; the branch
becomes a tag of the same name once its finding is written down.

## Tested

Every job type runs on the Mac. A thorough `--all` benchmark run under the
runner matches the same run from the user's Terminal (704 cells, median
ratio 1.001).

## The chip's power (`"powermetrics": true`)

Energy is a lab instrument (Zooko): measured in experiments here, and kept
out of the benchmark and the gate, where a number would become a goal.
Heat confounds it: back-to-back full-load runs throttle within minutes, so
each run starts after a minute's rest, in an order that cancels drift (0,
1, 2, 3, 3, 2, 1, 0). macOS's per-process estimate (`ri_energy_nj`) is a
model, -16% to +30% against the chip's power, unfit to compare builds.

A job may record the chip's power beside it. The runner's user runs one
fixed command line as root, which only prints and stops by itself after
two hours. Allow it once, with `sudo visudo -f
/etc/sudoers.d/benchrunner-powermetrics`, holding exactly:

    benchrunner ALL=(root) NOPASSWD: /usr/bin/powermetrics --samplers cpu_power -i 500 -n 14400

Check it with `sudo -u benchrunner sudo -n /usr/bin/powermetrics --samplers
cpu_power -i 500 -n 14400` (Ctrl-C after a sample). Any other arguments
stay refused, so the runner cannot use `powermetrics -o` to write files as
root.
