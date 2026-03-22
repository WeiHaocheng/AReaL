---
name: ssh_debug_trail
description: SSH into a trail debug server. Updates ~/.ssh/config and connects. Use when the user wants to SSH to a trail server.
argument-hint: <trial-id> [ip] [port]
disable-model-invocation: true
allowed-tools: Bash, Read, Edit, Write
---

Connect to a trail debug server via SSH.

## Arguments
- `$0`: Trial ID (e.g. `trail-123`) — used as the SSH Host alias
- `$1`: Trial IP (default: `fdbd:dccd:cdc2:12d1:0:260::`) — IPv4 or IPv6 address
- `$2`: Trial Port (default: `10741`)

## Steps

1. Parse the arguments:
   - TRIAL_ID = $0
   - TRIAL_IP = $1 (use `fdbd:dccd:cdc2:12d1:0:260::` if not provided)
   - TRIAL_PORT = $2 (use `10741` if not provided)

2. Update `~/.ssh/config` to add or replace the Host block for this Trial ID.

   The target block looks like:
   ```
   Host {TRIAL_ID}
       HostName {TRIAL_IP}
       Port {TRIAL_PORT}
       User tiger
       ProxyJump jump-proxy-eu.tiktok-row.org
       StrictHostKeyChecking=no
       UserKnownHostsFile=/dev/null
   ```

   Use the following Bash logic to update `~/.ssh/config`:
   - If a `Host {TRIAL_ID}` block already exists, replace it in-place (remove the old block and insert the new one).
   - If it does not exist, append it to the end of the file.

   Here is a reliable Bash snippet to do this:
   ```bash
   TRIAL_ID="$0"
   TRIAL_IP="${1:-fdbd:dccd:cdc2:12d1:0:260::}"
   TRIAL_PORT="${2:-10741}"
   SSH_CONFIG="$HOME/.ssh/config"

   NEW_BLOCK="Host $TRIAL_ID
       HostName $TRIAL_IP
       Port $TRIAL_PORT
       User tiger
       ProxyJump jump-proxy-eu.tiktok-row.org
       StrictHostKeyChecking=no
       UserKnownHostsFile=/dev/null"

   # Remove existing block for this Host (if any), then append new block
   python3 - <<'PYEOF'
   import re, os, sys

   config_path = os.path.expanduser("~/.ssh/config")
   trial_id = os.environ["TRIAL_ID"]
   new_block = os.environ["NEW_BLOCK"]

   if os.path.exists(config_path):
       with open(config_path, "r") as f:
           content = f.read()
   else:
       content = ""

   # Remove existing block matching this Host
   pattern = r"(?m)^Host\s+" + re.escape(trial_id) + r"\s*$.*?(?=^Host\s|\Z)"
   content = re.sub(pattern, "", content, flags=re.DOTALL | re.MULTILINE).rstrip()

   content = content + "\n\n" + new_block + "\n"
   with open(config_path, "w") as f:
       f.write(content)

   print(f"Updated ~/.ssh/config for Host: {trial_id}")
   PYEOF
   ```

   Export `TRIAL_ID` and `NEW_BLOCK` before running the Python snippet.

3. Print the SSH command that will be used:
   ```
   ssh {TRIAL_ID}
   ```

4. Run: `ssh {TRIAL_ID}` to connect to the server.

## Example invocations
- `/ssh_debug_trail trail-42` — uses default IP and port
- `/ssh_debug_trail trail-42 fdbd:dccd:cdc2:12d1:0:260:: 10741` — explicit IP and port
- `/ssh_debug_trail trail-99 10.0.0.1 22` — IPv4 example
