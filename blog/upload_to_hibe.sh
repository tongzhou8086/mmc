#!/usr/bin/env bash
# Render blog/post1.md to a self-contained HTML page and (re)upload it to
# hibe.dev. Re-running updates the same project (same name), so the public share
# URL stays stable.
#
#   ./blog/upload_to_hibe.sh              # name "gemm-blackwell-post"
#   HIBE_NAME=my-draft ./blog/upload_to_hibe.sh
#
# Prereq: a hibe token at ~/.config/hibe/token. To mint one (device flow):
#   curl -s -X POST https://hibe.dev/api/auth/device           # note user_code + device_code
#   # open https://hibe.dev/device, enter the user_code, approve
#   curl -s -X POST https://hibe.dev/api/auth/device/poll \
#     -H 'content-type: application/json' -d '{"device_code":"<device_code>"}'
#   mkdir -p ~/.config/hibe && printf %s '<access_token>' > ~/.config/hibe/token && chmod 600 ~/.config/hibe/token

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NAME="${HIBE_NAME:-gemm-blackwell-post}"
HTML="$(mktemp -d)/index.html"
TOKEN_FILE="${HIBE_TOKEN_FILE:-$HOME/.config/hibe/token}"

TOK="$(tr -d '[:space:]' < "$TOKEN_FILE" 2>/dev/null || true)"
[ -n "$TOK" ] || { echo "ERROR: empty/missing token at $TOKEN_FILE (see device-flow notes at top)." >&2; exit 1; }

# 1. Render the post, inlining every local figure as a data: URI.
python3 "$HERE/render_html.py" "$HTML"

# 2. Create the project, or (if the name already exists) update it in place via a
#    tarball PUT -- which preserves the share URL. Look up any existing id by name:
existing_id="$(curl -sS https://hibe.dev/api/projects -H "Authorization: Bearer $TOK" \
    | python3 -c "import sys,json
d=json.load(sys.stdin)
ps=d if isinstance(d,list) else d.get('projects',d.get('items',[]))
print(next((p['id'] for p in ps if p.get('name')==sys.argv[1]),''))" "$NAME" 2>/dev/null || true)"

if [ -n "$existing_id" ]; then
    tar -C "$(dirname "$HTML")" -czf "$(dirname "$HTML")/update.tgz" index.html
    resp="$(curl -sS -X PUT "https://hibe.dev/api/projects/$existing_id" \
        -H "Authorization: Bearer $TOK" -F "tar=@$(dirname "$HTML")/update.tgz")"
    echo "updated: $resp"
    id="$existing_id"
else
    resp="$(curl -sS -X POST https://hibe.dev/api/projects/single-html \
        -H "Authorization: Bearer $TOK" -F "name=$NAME" -F "html=@$HTML")"
    echo "created: $resp"
    id="$(printf '%s' "$resp" | python3 -c "import sys,json;print(json.load(sys.stdin).get('id',''))" 2>/dev/null || true)"
fi
rm -rf "$(dirname "$HTML")"
[ -n "$id" ] || { echo "ERROR: upload failed (no project id in response)." >&2; exit 1; }

# 3. Ensure public sharing is on, then print the stable share URL.
share="$(curl -sS -X PATCH "https://hibe.dev/api/projects/$id" \
    -H "Authorization: Bearer $TOK" -H 'content-type: application/json' \
    -d '{"share_enabled":true,"share_public":true}')"
sid="$(printf '%s' "$share" | python3 -c "import sys,json;print(json.load(sys.stdin).get('share_id',''))" 2>/dev/null || true)"
[ -n "$sid" ] && echo "PUBLIC URL: https://hibe.dev/s/$sid/" || echo "share: $share"
