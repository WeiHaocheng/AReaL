---
name: install_sglang
description: SSH into a trail server and install sglang from search-sglang repo. Clones the repo if not present, then runs pip installs.
argument-hint: [ssh-host] [work-dir] [branch]
disable-model-invocation: true
allowed-tools: Bash
---

Install sglang on a remote trail server via SSH.

## Arguments
- `$0`: SSH host alias (default: `trail-debug`)
- `$1`: Work directory on remote (default: `/opt/tiger/search-llm-inference`)
- `$2`: Branch to checkout (default: `main`)

## Steps

1. Parse arguments:
   - SSH_HOST = $0 (default: `trail-debug`)
   - WORK_DIR = $1 (default: `/opt/tiger/search-llm-inference`)
   - BRANCH = $2 (default: `main`)
   - REPO_URL = `git@code.byted.org:data/search-sglang.git`
   - REPO_DIR = `$WORK_DIR/search-sglang`

2. Check if `$REPO_DIR` already exists on the remote:
   - If it exists: run `git fetch && git checkout $BRANCH && git pull` inside it
   - If not: run `git clone -b $BRANCH $REPO_URL $REPO_DIR`

   ```bash
   ssh $SSH_HOST "
     if [ -d '$REPO_DIR' ]; then
       echo 'Repo exists, updating...'
       cd '$REPO_DIR' && git fetch && git checkout '$BRANCH' && git pull
     else
       echo 'Cloning repo...'
       git clone -b '$BRANCH' git@code.byted.org:data/search-sglang.git '$REPO_DIR'
     fi
   "
   ```

3. Install sglang in editable mode:
   ```bash
   ssh $SSH_HOST "cd '$REPO_DIR' && pip install -e 'python' 2>&1 | tail -5"
   ```

4. Install nvidia-cudnn:
   ```bash
   ssh $SSH_HOST "pip install nvidia-cudnn-cu12==9.16.0.29 2>&1 | tail -5"
   ```

5. Verify installation:
   ```bash
   ssh $SSH_HOST "python3 -c 'import sglang; print(\"sglang version:\", sglang.__version__)'"
   ```

6. Print a summary of what was installed (sglang version, cudnn version, repo commit).

## Example invocations
- `/install_sglang` — use defaults (trail-debug, /opt/tiger/search-llm-inference, main)
- `/install_sglang trail-debug` — explicit SSH host
- `/install_sglang trail-debug /opt/tiger/search-llm-inference main` — all explicit
