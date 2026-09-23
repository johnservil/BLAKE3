#!/bin/sh
# Install the pre-commit hook in this checkout: a dispatcher in the git
# directory's hooks folder that runs tools/git-hooks/pre-commit, so the
# hook itself stays under version control. Run once per checkout.
set -e
hooks=$(git rev-parse --git-path hooks)
mkdir -p "$hooks"
printf '#!/bin/sh\nexec sh "$(git rev-parse --show-toplevel)/tools/git-hooks/pre-commit" "$@"\n' > "$hooks/pre-commit"
chmod +x "$hooks/pre-commit"
echo "installed $hooks/pre-commit"
