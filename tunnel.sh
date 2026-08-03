#!/bin/bash
#
# Cloudflare Tunnel Helper Script
# ================================
# Exposes a local HTTP server publicly via Cloudflare Tunnel.
#
# Usage:
#   ./tunnel.sh
#
# What it does:
#   - Runs cloudflared to create a tunnel to http://localhost:8787
#   - Prints the public tunnel URL to stdout in a parseable format
#   - Runs until interrupted (Ctrl+C)
#
# Requirements:
#   - cloudflared must be installed (https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/)
#   - No authentication needed for quick tunnels
#
# Example output:
#   TUNNEL_URL=https://random-name.trycloudflare.com
#

set -euo pipefail

LOCAL_PORT="${LOCAL_PORT:-8780}"
LOCAL_URL="http://localhost:${LOCAL_PORT}"

echo "Starting Cloudflare tunnel to ${LOCAL_URL} ..." >&2

# Run cloudflared, capture the tunnel URL from stderr (cloudflared prints it there),
# and echo it to stdout in a parseable format.
cloudflared tunnel --url "${LOCAL_URL}" --no-autoupdate 2>&1 |
  while IFS= read -r line; do
    echo "$line" >&2
    # Parse the tunnel URL from lines like:
    # 2024/01/01 12:00:00 INF +[https://random-name.trycloudflare.com] +https://random-name.trycloudflare.com
    if [[ "$line" =~ \+(https?://[a-zA-Z0-9.-]+\.trycloudflare\.com) ]]; then
      tunnel_url="${BASH_REMATCH[1]}"
      echo "TUNNEL_URL=${tunnel_url}"
    fi
  done
