---
name: install-areal
description: SSH into a remote server and install AReaL. Clones the repo if not present (using the project's fork and branch), installs via uv, and validates the installation.
argument-hint: [ssh-host] [work-dir] [branch]
user-invocable: true
disable-model-invocation: true
allowed-tools: Bash
---

Install AReaL on a remote server via SSH.

## Arguments

- `$0`: SSH host alias (default: `trail-debug`)
- `$1`: Work directory on remote (default: `/opt/tiger/AReaL`)
- `$2`: Branch to checkout (default: `scaffolding_rollout_workspace`)

## Constants

- `REPO_URL` = `https://github.com/WeiHaocheng/AReaL.git`

## Steps

1. Parse arguments:
   - SSH_HOST = $0 (default: `trail-debug`)
   - WORK_DIR = $1 (default: `/opt/tiger/AReaL`)
   - BRANCH = $2 (default: `scaffolding_rollout_workspace`)
   - REPO_URL = `https://github.com/WeiHaocheng/AReaL.git`

2. Check if `$WORK_DIR` already exists on the remote:
   - If it exists: run `git fetch && git checkout $BRANCH && git pull` inside it
   - If not: run `git clone -b $BRANCH $REPO_URL $WORK_DIR`

   ```bash
   ssh $SSH_HOST "
     if [ -d '$WORK_DIR/.git' ]; then
       echo 'Repo exists, updating...'
       cd '$WORK_DIR' && git fetch && git checkout '$BRANCH' && git pull
     else
       echo 'Cloning repo...'
       git clone -b '$BRANCH' '$REPO_URL' '$WORK_DIR'
     fi
   "
   ```

3. Check which install method to use (Docker or uv):

   First check if `uv` is available on the remote:
   ```bash
   ssh $SSH_HOST "which uv 2>/dev/null && echo HAS_UV || echo NO_UV"
   ```

   **If uv is available** — install via `uv sync`:
   ```bash
   ssh $SSH_HOST "cd '$WORK_DIR' && uv sync --extra cuda 2>&1 | tail -20"
   ```

   **If uv is NOT available** — install via pip in editable mode (Docker environment):
   ```bash
   ssh $SSH_HOST "cd '$WORK_DIR' && uv pip install -e . --no-deps 2>&1 | tail -20"
   ```
   If uv is also not available as `uv pip`, fall back to:
   ```bash
   ssh $SSH_HOST "cd '$WORK_DIR' && pip install -e . --no-deps 2>&1 | tail -20"
   ```

4. Validate the installation:
   ```bash
   ssh $SSH_HOST "cd '$WORK_DIR' && uv run python3 areal/tools/validate_installation.py 2>&1 | tail -30"
   ```
   If `uv` is not available, try:
   ```bash
   ssh $SSH_HOST "cd '$WORK_DIR' && python3 areal/tools/validate_installation.py 2>&1 | tail -30"
   ```

5. Print a summary: installed branch, repo commit (`git log -1 --oneline`), and whether validation passed.

## Example invocations

- `/install-areal` — use defaults (trail-debug, /opt/tiger/AReaL, scaffolding_rollout_workspace)
- `/install-areal trail-debug` — explicit SSH host, default dir and branch
- `/install-areal trail-debug /opt/tiger/AReaL` — explicit host and dir, default branch
- `/install-areal trail-debug /opt/tiger/AReaL scaffolding_rollout_workspace` — all explicit
