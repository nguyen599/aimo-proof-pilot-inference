---
title: NII Relay Nguyen
emoji: 🛰️
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 6.20.0
app_file: app.py
pinned: false
---

# NII Relay

A private message relay between your browser and shell daemons running inside
no-SSH containers (e.g. on the NII cluster). The Space holds a queue of
commands; daemons poll it, run the commands in persistent bash sessions, and
post the output back.

## Deploy

1. Create a **private** Space (SDK: Gradio). Upload `app.py` and
   `requirements.txt`. Keeping it **private** is the auth boundary — reaching it
   requires an HF token with access; there is no separate app-level secret.
2. Wait for the Space to build. The URL (`https://huggingface.co/spaces/<you>/<name>`)
   and its repo id (`<you>/<name>`) are what the daemon connects to.

## Use

Open the Space, select a team member or one online container, name a bash
session, and enter a command. The UI supports three scopes:

- **Run on selected node** queues the command on one exact client ID. The
  container dropdown is pinned to the selected member's assigned pair.
- **Run on member nodes** queues it on the selected member's two online nodes.
- **Broadcast to all online** queues it on every connected client.

The NII nodes share their main filesystem. Run package installs, repository
updates, downloads, and other shared-file mutations on exactly one selected
node. Use member or all-node broadcast only for node-local work such as
`nvidia-smi`, process inspection, or starting one rank-specific service per
node.

Reuse a session name to keep its `cd`, environment, and virtual environment;
use a new name to open a fresh shell.

The container list is updated on page load, team-member changes, and the
**Refresh** button. The periodic UI refresh updates command histories and
downloads only, so it cannot move a selection to another member's node. Open
tabs automatically reload after a Space rebuild so stale Gradio component IDs
cannot route API responses into the UI dropdowns.

## Team routing

The UI's human-facing node numbers map to the current zero-based daemon labels:

| Member | Human nodes | Default client ranks |
| --- | --- | --- |
| `vu` | 0-1 | `node0`, `node1` |
| `bogo` | 2-3 | `node2`, `node3` |
| `yi` | 4-5 | `node4`, `node5` |
| `nguyen` | 6-7 | `node6`, `node7` |

The server first uses optional `RELAY_MEMBER` metadata registered by the daemon,
then falls back to parsing the rank from client IDs such as
`node2-hnode070`. Member routing is a convenience selector, not an authorization
boundary: anyone who can access the private Space can select any member or node.

Call the two-node API with the member name:

```python
from gradio_client import Client
from huggingface_hub import get_token

client = Client(
    "imo2026-challenge/control-panel-nguyen",
    token=get_token(),
)
result = client.predict(
    member="vu",
    session="main",
    command="nvidia-smi",
    timeout=120,
    api_name="/ui_team_broadcast",
)
print(result)
```

Use `/ui_team_clients` to inspect the resolved clients for a member and
`/ui_team_history` to read the combined history. Existing `/ui_send` and
`/ui_broadcast` calls remain unchanged. `/ui_send` accepts one exact registered
client ID or one unambiguous rank shorthand such as `node2`; it does not accept
a list or a comma-separated subset.

## Daemon

The bundled client accepts optional routing metadata:

```bash
export RELAY_SPACE=imo2026-challenge/control-panel-nguyen
export CLIENT_ID=node0-$(hostname)
export RELAY_MEMBER=vu
python daemon/client.py
```

`RELAY_MEMBER` is optional when `CLIENT_ID` contains a zero-based `node0` through
`node7` rank. The launcher in `aimo-proof-pilot` derives the member automatically
from `GLOBAL_RANK`, `NODE_RANK`, or `SLURM_NODEID`.

The daemon lives in `daemon/`.
