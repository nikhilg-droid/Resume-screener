#!/usr/bin/env bash
# Creates one GitHub issue per numbered line in docs/issues.md. Run from the repo root after `gh auth login`.
set -euo pipefail
grep -E '^[0-9]+\. \*\*' docs/issues.md | while IFS= read -r line; do
  title=$(sed -E 's/^[0-9]+\. \*\*([^*]+)\*\*.*/\1/; s/\.$//' <<<"$line")
  body=$(sed -E 's/^[0-9]+\. \*\*[^*]+\*\* ?//' <<<"$line")
  gh issue create --title "$title" --body "$body"$'\n\nSee docs/prd.md.'
done
