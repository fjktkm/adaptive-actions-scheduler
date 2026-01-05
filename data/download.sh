#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"

echo "cron,startedAt" > data.csv
gh run list --json headSha,startedAt --limit 100 | \
  jq -r '.[] | .startedAt + " " + .headSha' | \
  while read time sha; do
    cron=$(gh api "repos/{owner}/{repo}/contents/.github/workflows/schedule.yml?ref=$sha" \
      --jq '.content | @base64d | match("cron: \"(.*)\"") | .captures[0].string' 2>/dev/null)
    echo "$cron,$time"
  done >> data.csv
