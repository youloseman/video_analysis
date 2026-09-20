#!/usr/bin/env bash
# Send the current main to the phone: tag it and push the tag. Codemagic
# builds the iOS app on every mobile/* tag (see codemagic.yaml) and drops it
# into the Internal TestFlight group; the TestFlight app then offers Update.
#
#   scripts/mobile-release.sh            tag = mobile/2026-09-20T18-42Z
#   scripts/mobile-release.sh 1.1-beta   tag = mobile/1.1-beta
#
# Refuses on a dirty tree or off main: the build takes what is committed, and
# a tag on a half-committed state is the build you cannot reproduce.
set -euo pipefail
cd "$(dirname "$0")/.."

branch=$(git rev-parse --abbrev-ref HEAD)
if [ "$branch" != "main" ]; then
  echo "on '$branch' -- the app is built from main. Switch first." >&2; exit 1
fi
if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
  echo "tracked files are modified -- commit or stash first (untracked files are fine: the build takes what is committed)." >&2; exit 1
fi
git fetch -q origin main
if [ "$(git rev-parse HEAD)" != "$(git rev-parse origin/main)" ]; then
  echo "local main differs from origin/main -- push (or pull) first." >&2; exit 1
fi

name=${1:-$(date -u +%Y-%m-%dT%H-%MZ)}
tag="mobile/$name"
if git rev-parse -q --verify "refs/tags/$tag" >/dev/null; then
  echo "tag $tag already exists." >&2; exit 1
fi

last=$(git describe --tags --match 'mobile/*' --abbrev=0 2>/dev/null || true)
echo "tagging $(git rev-parse --short HEAD) as $tag"
if [ -n "$last" ]; then
  echo "since $last: $(git rev-list --count "$last"..HEAD) commits, of which touching the app:"
  git log --oneline "$last"..HEAD -- backend/app/static mobile | sed 's/^/  /'
fi
git tag -a "$tag" -m "iOS build from $(git rev-parse --short HEAD)"
git push -q origin "$tag"
echo "pushed. Codemagic: https://codemagic.io/apps -> video_analysis (build ~12 min, then TestFlight -> Update)."
