#!/usr/bin/env bash
# Move this repository's forensics_core submodule to a release, and refuse if the suite fails.
#
#   scripts/bump_core.sh core-vX.Y.Z
#
# Both project repositories must be bumped to the same commit: CI's submodule-parity job fails
# otherwise, because two halves of the programme running different method code would produce
# results that are silently not comparable.
set -euo pipefail

TAG="${1:?usage: bump_core.sh core-vX.Y.Z}"

if [ -n "$(git status --porcelain)" ]; then
  echo "working tree is dirty; commit or stash first" >&2
  exit 1
fi

git submodule update --init --recursive
cd packages/forensics_core
git fetch --tags origin
if ! git rev-parse --verify "$TAG" >/dev/null 2>&1; then
  echo "forensics_core has no tag ${TAG}" >&2
  exit 1
fi
git checkout -q "$TAG"
cd ../..

echo "submodule moved to ${TAG}; running the suite before committing"
if ! uv sync --all-packages; then
  echo "uv sync failed against ${TAG}; NOT committing the bump" >&2
  git submodule update --init --recursive
  exit 1
fi
if ! uv run pytest -q; then
  echo "the suite fails against ${TAG}; NOT committing the bump" >&2
  git submodule update --init --recursive
  exit 1
fi

git add packages/forensics_core
git commit -m "Bump forensics_core to ${TAG}

The suite passes against it. The sibling project repository must be bumped to the same tag,
or CI's submodule-parity job will fail."
echo "bumped. Now bump the sibling repository to ${TAG} as well."
