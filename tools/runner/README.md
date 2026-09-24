# Mac benchmark runner

`runner.py` runs benchmark jobs on the Mac as the hidden standard account
`benchrunner`, with code cloned from GitHub at the commits each job names.
Jobs go in the checkout's `runner/jobs/`; results come back in
`runner/results/<job>.<time>/`. Both folders stay out of git
(`.git/info/exclude`). The job format is the docstring at the top of
`runner.py`.

## The account (once)

    pw="$(openssl rand -base64 32)"
    sudo sysadminctl -addUser benchrunner -fullName "Bench Runner" -password "$pw"
    sudo dscl . create /Users/benchrunner IsHidden 1
    sudo dscl . -create /Users/benchrunner UserShell /usr/bin/false
    sudo dscl . -passwd /Users/benchrunner "$(openssl rand -base64 32)"

Nobody can log in to it; `sudo -u benchrunner` runs commands as it.

## Setting up and starting

    sh ~/piplayground/blake3-servil/tools/runner/setup-mac.sh

Run it as yourself; it asks for sudo when it needs it. It makes the
account's home private, installs PyPy when absent, and gives the account
the same rustc as yours (matched by commit hash). It then makes the
exchange folders (`jobs/` yours and readable, `results/` the runner's and
readable) and lets the account pass through the folders above them. Last,
it copies `runner.py` to `/Users/Shared/bench-runner/` (yours, read-only to
the runner) and starts it. Steps already done are skipped, so the same
command starts the runner every time, including after a change to
`runner.py`. Ctrl-C stops the runner after the current job.

While a job runs, nothing else should run on the Mac, the VM included.

## Tested

September 24, 2026: all three job types (benchmark, example, perf_regress)
and rejected jobs in the VM under PyPy; on the Mac, a thorough `--all` run
under the runner matched the same run from the user's Terminal (704
cells, median ratio 1.001; bench-hashes NEXT-STEPS.md has the numbers).
