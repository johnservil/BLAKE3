#!/bin/sh
# Sets up the benchmark runner for the hidden `benchrunner` account, then
# starts it. Run as yourself on the Mac (it asks for sudo when it needs it):
#
#     sh ~/piplayground/blake3-servil/tools/runner/setup-mac.sh
#
# Safe to run again: steps already done are skipped, and runner.py is
# copied afresh each time, so this is also how to start the runner after a
# reviewed change to runner.py.
set -eu

code=$(cd "$(dirname "$0")" && pwd)
checkout=$(cd "$code/../.." && pwd)
# The exchange folder, kept out of git by .git/info/exclude.
runner=$checkout/runner
grep -qx "/runner/" "$checkout/.git/info/exclude" || echo "/runner/" >> "$checkout/.git/info/exclude"
installed=/Users/Shared/bench-runner

step() { printf '\n== %s\n' "$*"; }

step "the benchrunner account exists"
dscl . -read /Users/benchrunner UserShell

step "PyPy"
command -v pypy3 >/dev/null || brew install pypy3
pypy=$(command -v pypy3)
"$pypy" --version

step "benchrunner's home is private"
sudo chmod 700 /Users/benchrunner

step "benchrunner's Rust toolchain: the very compiler you use"
# Matched by rustc's commit hash. A channel name such as `nightly` would
# install whatever is newest today. A nightly is dated a day or so after
# its rustc commit, so try the dates around it.
br_rustup="sudo -u benchrunner -H /Users/benchrunner/.cargo/bin/rustup"
br_rustc="sudo -u benchrunner -H /Users/benchrunner/.cargo/bin/rustc"
want_hash=$(cd ~ && rustc -vV | sed -n 's/^commit-hash: //p')
want_date=$(cd ~ && rustc -vV | sed -n 's/^commit-date: //p')
echo "yours: $(cd ~ && rustc --version)"
if ! sudo -u benchrunner -H test -x /Users/benchrunner/.cargo/bin/cargo; then
    sudo -u benchrunner -H sh -c "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --no-modify-path --default-toolchain none"
fi
if [ "$($br_rustc -vV 2>/dev/null | sed -n 's/^commit-hash: //p')" != "$want_hash" ]; then
    found=
    for shift in +1d +0d +2d +3d; do
        toolchain=nightly-$(date -j -v"$shift" -f %Y-%m-%d "$want_date" +%Y-%m-%d)
        $br_rustup toolchain install --profile minimal "$toolchain"
        got=$($br_rustup run "$toolchain" rustc -vV | sed -n 's/^commit-hash: //p')
        if [ "$got" = "$want_hash" ]; then found=$toolchain; break; fi
        $br_rustup toolchain uninstall "$toolchain"
    done
    [ -n "$found" ] || { echo "no nightly near $want_date has rustc $want_hash"; exit 1; }
    $br_rustup default "$found"
    $br_rustup toolchain list | grep -v "^$found" | cut -d' ' -f1 | while read -r other; do $br_rustup toolchain uninstall "$other"; done
fi
echo "benchrunner's: $($br_rustc --version)"

step "the exchange folder: jobs/ yours, results/ the runner's"
chmod 600 "$checkout/ghtokenclassic.txt"
mkdir -p "$runner/jobs" "$runner/results"
sudo chown benchrunner "$runner/results"
for dir in ~ ~/piplayground "$checkout"; do
    ls -led "$dir" | grep -q "benchrunner allow search" || chmod +a "benchrunner allow search" "$dir"
done
ls -le "$checkout/ghtokenclassic.txt"
ls -led "$runner/jobs" "$runner/results"

step "runner.py and perf_regress.py installed read-only for benchrunner"
sudo mkdir -p "$installed"
sudo chown "$(whoami)" "$installed"
cp "$code/runner.py" "$installed/runner.py"
cp "$code/../perf_regress.py" "$installed/perf_regress.py"
chmod 755 "$installed"
chmod 644 "$installed/runner.py" "$installed/perf_regress.py"

step "benchrunner can run PyPy"
sudo -u benchrunner -H "$pypy" --version

step "starting the runner (Ctrl-C stops it after the current job)"
exec sudo -u benchrunner -H "$pypy" "$installed/runner.py" "$runner"
