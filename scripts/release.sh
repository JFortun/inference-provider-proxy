#!/usr/bin/env bash
#
# Cut a release: bump the version, push it to main, tag it. CI does the rest:
# the tag push runs the version check, creates the release with its notes,
# builds and attaches the wheel and the sdist, then deploys (see
# .github/workflows/release.yml).
#
# Usage: scripts/release.sh X.Y.Z

set -Eeuo pipefail

usage() {
  echo "usage: $0 X.Y.Z   (for example: $0 0.4.0)" >&2
  exit 2
}

[ $# -eq 1 ] || usage
VERSION=$1
[[ "$VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] || usage
TAG="v${VERSION}"

cd "$(git rev-parse --show-toplevel)"

current="$(uv version --short)"
if [ "$current" = "$VERSION" ]; then
  echo "pyproject.toml already declares $VERSION, nothing to bump" >&2
  exit 1
fi

if [ "$(git rev-parse --abbrev-ref HEAD)" != "main" ]; then
  echo "not on main ($(git rev-parse --abbrev-ref HEAD)), checkout main first" >&2
  exit 1
fi

if [ -n "$(git status --porcelain)" ]; then
  echo "the worktree is dirty, commit or stash first:" >&2
  git status --short >&2
  exit 1
fi

git fetch origin main
behind=$(git rev-list --count main..origin/main)
if [ "$behind" -ne 0 ]; then
  echo "local main is $behind commit(s) behind origin/main, pull first" >&2
  exit 1
fi

if git rev-parse -q --verify "refs/tags/${TAG}" >/dev/null ||
  git ls-remote --exit-code --tags origin "refs/tags/${TAG}" >/dev/null 2>&1; then
  echo "tag ${TAG} already exists locally or on origin, delete it first" >&2
  exit 1
fi

uv version "$VERSION"
git add pyproject.toml uv.lock
git commit -m "chore(release): bump version to ${VERSION}"
git push origin main

git tag -s "${TAG}" -m "${TAG}"
git push origin "${TAG}"

echo
echo "pushed ${TAG}: CI runs check-tag, then creates the release, attaches"
echo "the wheel and the sdist, and deploys. Watch it with:"
echo "  gh run list --workflow Release --limit 1"
echo "  gh run watch <run-id>"
