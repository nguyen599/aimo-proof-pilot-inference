"""
NII Relay — a tiny HF Space that acts as a message relay between you (in the
browser) and shell daemons running inside no-SSH containers on the cluster.

Flow
----
1. A daemon inside a container registers with a CLIENT_ID and polls this Space
   every few seconds (via the `gradio_client` API endpoints below).
2. You pick a container + bash session in the web UI and submit a shell command.
3. The next poll hands the command to the daemon, which marks it ACKNOWLEDGED,
   runs it in a persistent bash session, and posts stdout + exit code back.
4. The UI shows the output.

State is in-memory only. If the Space restarts, daemons simply re-register on
their next poll — but any not-yet-delivered commands are lost. That is fine for
interactive debugging.

Auth: the Space is PRIVATE, so reaching it at all requires an HF token with
access — that token is the auth boundary. There is no separate app-level secret.
"""

import json
import os
import re
import time
import uuid
import shutil
import tempfile
import pathlib
import threading
from collections import defaultdict

import gradio as gr

CLIENT_STALE_AFTER = 30                                # seconds w/o a poll => "offline"
MAX_OUTPUT_CHARS = 200_000                             # cap stored output per command
MAX_HISTORY_PER_CLIENT = 200                           # ring-buffer of commands
MAX_FILE_BYTES = 100 * 1024 * 1024                     # 100 MB cap on transfers
FILE_TTL = 3600                                        # delete stored files older than 1h

FILES_DIR = pathlib.Path(tempfile.gettempdir()) / "relay_files"
FILES_DIR.mkdir(exist_ok=True)

_HEX32 = re.compile(r"\A[0-9a-f]{32}\Z")
_NODE_RANK = re.compile(r"(?:^|[^a-z0-9])node[-_ ]?(\d+)(?=[^0-9]|$)", re.IGNORECASE)

# Human-facing nodes use the same zero-based numbering as daemon labels
# node0-node7. Explicit member metadata from a daemon takes precedence.
TEAM_LAYOUT = {
    "vu": {"display": "Vu", "nodes": "0-1", "ranks": frozenset((0, 1))},
    "bogo": {"display": "Bogo", "nodes": "2-3", "ranks": frozenset((2, 3))},
    "yi": {"display": "Yi", "nodes": "4-5", "ranks": frozenset((4, 5))},
    "nguyen": {"display": "Nguyen", "nodes": "6-7", "ranks": frozenset((6, 7))},
}
TEAM_CHOICES = [
    (f"{config['display']} - nodes {config['nodes']}", member)
    for member, config in TEAM_LAYOUT.items()
]


def _valid_id(s) -> bool:
    return isinstance(s, str) and bool(_HEX32.match(s))

LOCK = threading.Lock()
CLIENTS: dict[str, dict] = {}                          # cid -> {last_seen, meta}
COMMANDS: dict[str, dict] = {}                         # command_id -> record
BY_CLIENT: dict[str, list[str]] = defaultdict(list)    # cid -> [command_id, ...]
PUT_FILES: dict[str, dict] = {}                        # file_id -> {path, filename} (server->client payloads)


def _now() -> float:
    return time.time()


def _human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f}{unit}" if unit == "B" else f"{n:.1f}{unit}"
        n /= 1024


def _normalize_member(member: str) -> str:
    key = (member or "").strip().lower()
    return key if key in TEAM_LAYOUT else ""


def _canonical_client_id(client_id: str) -> str:
    """Return the daemon ID, stripping any UI-only online-status prefixes."""
    client_id = str(client_id or "").strip()
    while client_id and client_id[:1] in "🟢🔴":
        client_id = client_id[1:].lstrip()
    return client_id


def _client_rank(client_id: str) -> int | None:
    match = _NODE_RANK.search(client_id or "")
    return int(match.group(1)) if match else None


def _parse_client_meta(meta: str) -> dict:
    if not meta:
        return {}
    try:
        parsed = json.loads(meta)
    except (TypeError, ValueError):
        parsed = None
    if isinstance(parsed, dict):
        return parsed

    fields = {}
    for key in ("member", "team_member", "relay_member", "node_rank"):
        match = re.search(rf"(?:^|\s){key}=([^\s]+)", str(meta), re.IGNORECASE)
        if match:
            fields[key] = match.group(1)
    return fields


def _member_for_rank(rank: int | None) -> str:
    if rank is None:
        return ""
    for member, config in TEAM_LAYOUT.items():
        if rank in config["ranks"]:
            return member
    return ""


def _client_member(client_id: str, client: dict) -> str:
    explicit = _normalize_member(client.get("member", ""))
    if explicit:
        return explicit
    rank = client.get("node_rank")
    if not isinstance(rank, int):
        rank = _client_rank(client_id)
    return _member_for_rank(rank)


def _client_sort_key(item: tuple[str, dict]) -> tuple:
    client_id, client = item
    rank = client.get("node_rank")
    if not isinstance(rank, int):
        rank = _client_rank(client_id)
    return (rank is None, rank if rank is not None else 0, client_id)


def _touch_client(cid: str, meta: str = ""):
    c = CLIENTS.setdefault(cid, {"first_seen": _now(), "meta": meta})
    c["last_seen"] = _now()
    if meta:
        c["meta"] = meta
        fields = _parse_client_meta(meta)
        member = _normalize_member(
            fields.get("member")
            or fields.get("team_member")
            or fields.get("relay_member")
            or ""
        )
        if member:
            c["member"] = member
        try:
            c["node_rank"] = int(fields["node_rank"])
        except (KeyError, TypeError, ValueError):
            pass
    if "node_rank" not in c:
        rank = _client_rank(cid)
        if rank is not None:
            c["node_rank"] = rank


def _unlink(path: str):
    try:
        os.unlink(path)
    except OSError:
        pass


def _drop_command(cmd_id: str):
    """Remove a command record and any file it owns on disk. Call under LOCK."""
    rec = COMMANDS.pop(cmd_id, None)
    if not rec:
        return
    if rec.get("stored_path"):
        _unlink(rec["stored_path"])
    fid = rec.get("file_id")
    if fid:
        info = PUT_FILES.pop(fid, None)
        if info:
            _unlink(info["path"])


def _append_and_trim(cid: str, cmd_id: str):
    """Append to a client's history and ring-buffer it, deleting evicted files.
    Call under LOCK. Used by every append site so growth is bounded uniformly."""
    hist = BY_CLIENT[cid]
    hist.append(cmd_id)
    while len(hist) > MAX_HISTORY_PER_CLIENT:
        _drop_command(hist.pop(0))


def _file_reaper():
    """Backstop against disk growth on the ephemeral Space: delete stored files
    older than FILE_TTL and prune PUT_FILES whose file is gone."""
    while True:
        time.sleep(600)
        cutoff = _now() - FILE_TTL
        try:
            for p in FILES_DIR.iterdir():
                try:
                    if p.is_file() and p.stat().st_mtime < cutoff:
                        p.unlink()
                except OSError:
                    pass
        except OSError:
            pass
        with LOCK:
            for fid in list(PUT_FILES):
                if not os.path.isfile(PUT_FILES[fid]["path"]):
                    PUT_FILES.pop(fid, None)


# --------------------------------------------------------------------------- #
# Machine API (called by the daemon via gradio_client)                         #
# --------------------------------------------------------------------------- #
def api_register(client_id: str, meta: str):
    client_id = _canonical_client_id(client_id)
    if not client_id:
        raise gr.Error("empty client_id")
    with LOCK:
        _touch_client(client_id, meta or "")
    return {"ok": True, "ts": _now()}


def api_poll(client_id: str):
    """Return all pending commands for this client and flip them to acknowledged."""
    client_id = _canonical_client_id(client_id)
    out = []
    with LOCK:
        _touch_client(client_id)
        for cmd_id in list(BY_CLIENT.get(client_id, [])):
            rec = COMMANDS.get(cmd_id)
            if rec is None:
                continue
            if rec["status"] == "pending":
                rec["status"] = "acknowledged"
                rec["acknowledged_at"] = _now()
                out.append({
                    "id": rec["id"],
                    "type": rec.get("type", "exec"),
                    "session": rec.get("session", "main"),
                    "command": rec.get("command", ""),
                    "timeout": rec.get("timeout", 120),
                    # file-transfer fields (None for plain exec commands)
                    "file_id": rec.get("file_id"),
                    "dest": rec.get("dest"),
                    "remote_path": rec.get("remote_path"),
                    "filename": rec.get("filename"),
                })
    return {"commands": out, "ts": _now()}


def api_result(command_id: str, client_id: str,
               exit_code: float, stdout: str, session: str):
    if not _valid_id(command_id):
        raise gr.Error("bad command id")
    client_id = _canonical_client_id(client_id)
    with LOCK:
        rec = COMMANDS.get(command_id)
        if rec is None:
            # Space probably restarted; record it anyway so output isn't lost.
            rec = COMMANDS[command_id] = {
                "id": command_id, "client_id": client_id, "session": session,
                "command": "(unknown — relay restarted)", "created": _now(),
                "status": "pending",
            }
            _append_and_trim(client_id, command_id)
        rec["status"] = "done"
        rec["finished_at"] = _now()
        rec["exit_code"] = int(exit_code)
        rec["stdout"] = (stdout or "")[:MAX_OUTPUT_CHARS]
        _touch_client(client_id)
    return {"ok": True}


def api_fetch_file(file_id: str):
    """Daemon calls this to download a server->client (push) payload."""
    if not _valid_id(file_id):
        raise gr.Error("bad file id")
    info = PUT_FILES.get(file_id)
    if not info or not os.path.isfile(info["path"]):
        raise gr.Error("unknown or expired file_id")
    return info["path"]


def api_upload_file(command_id: str, client_id: str, fileobj, error: str):
    """Daemon calls this to post the result of a client->server (pull) request.
    `fileobj` is the uploaded file path on success, or None when `error` is set."""
    if not _valid_id(command_id):
        raise gr.Error("bad command id")
    client_id = _canonical_client_id(client_id)

    # Do filesystem work BEFORE taking the lock (a 100 MB copy must not stall
    # every concurrent poll/UI handler). Enforce the size cap server-side too —
    # don't trust the client.
    saved = None
    fname = sz = None
    too_big = False
    if fileobj:
        sz = os.path.getsize(fileobj)
        if sz > MAX_FILE_BYTES:
            too_big = True
        else:
            fname = os.path.basename(fileobj)
            saved = FILES_DIR / f"{command_id}__{fname}"
            shutil.copy(fileobj, saved)

    with LOCK:
        rec = COMMANDS.get(command_id)
        if rec is None:  # relay restarted — keep the file anyway
            rec = COMMANDS[command_id] = {
                "id": command_id, "client_id": client_id, "session": "file",
                "type": "get", "command": "(file fetch — relay restarted)",
                "created": _now(), "status": "pending",
            }
            _append_and_trim(client_id, command_id)
        rec["status"] = "done"
        rec["finished_at"] = _now()
        _touch_client(client_id)
        if too_big:
            rec.update(exit_code=1,
                       stdout=f"file exceeds the {_human(MAX_FILE_BYTES)} limit")
        elif saved is not None:
            rec.update(stored_path=str(saved), filename=fname, size=sz,
                       exit_code=0, stdout=f"fetched {fname} ({_human(sz)})")
        else:
            rec.update(exit_code=1, stdout=(error or "fetch failed"))
    return {"ok": True}


# --------------------------------------------------------------------------- #
# Human UI helpers                                                             #
# --------------------------------------------------------------------------- #
def _client_choices(member: str = "") -> list[tuple[str, str]]:
    member = _normalize_member(member)
    with LOCK:
        # Dropdown labels include online status, but values remain stable daemon
        # IDs. Canonicalizing here also collapses clients accidentally registered
        # with a copied UI label such as "🟢 node6-host".
        clients_by_id = {}
        for raw_id, client in CLIENTS.items():
            client_id = _canonical_client_id(raw_id)
            if not client_id:
                continue
            if member and _client_member(client_id, client) != member:
                continue
            previous = clients_by_id.get(client_id)
            if (
                previous is None
                or client.get("last_seen", 0) > previous.get("last_seen", 0)
            ):
                clients_by_id[client_id] = client
        items = list(clients_by_id.items())
        items.sort(key=_client_sort_key)
        choices = []
        for cid, c in items:
            age = _now() - c.get("last_seen", 0)
            dot = "🟢" if age < CLIENT_STALE_AFTER else "🔴"
            choices.append((f"{dot} {cid}", cid))
        return choices


def _strip_dot(label: str) -> str:
    return _canonical_client_id(label)


def _resolve_client_id(label: str) -> str:
    """Resolve `node2` to one registered ID such as `node2-hnode070`."""
    candidate = _strip_dot(label)
    if not candidate:
        return ""
    with LOCK:
        if candidate in CLIENTS:
            return candidate
        matches = sorted(
            client_id for client_id in CLIENTS
            if client_id.startswith(candidate + "-")
        )
    return matches[0] if len(matches) == 1 else candidate


def _queue_exec(cid: str, session: str, command: str, timeout: int,
                created: float | None = None) -> str:
    """Queue one exec command. Call while holding LOCK."""
    cmd_id = uuid.uuid4().hex
    COMMANDS[cmd_id] = {
        "id": cmd_id, "client_id": cid, "session": session,
        "command": command, "timeout": int(timeout),
        "status": "pending", "created": created if created is not None else _now(),
    }
    _append_and_trim(cid, cmd_id)
    return cmd_id


def ui_send(client_label: str, session: str, command: str, timeout: int):
    cid = _resolve_client_id(client_label)
    if not cid:
        return "⚠️ pick a container first", command
    if not command.strip():
        return "⚠️ empty command", command
    session = (session or "main").strip()
    with LOCK:
        _queue_exec(cid, session, command, timeout)
    return f"➡️ sent to `{cid}` [{session}]", ""


def ui_send_for_member(member: str, client_label: str, session: str,
                       command: str, timeout: int):
    """UI-only guard that prevents a stale selection crossing team ownership."""
    member = _normalize_member(member)
    cid = _resolve_client_id(client_label)
    if not member:
        return "⚠️ choose a valid team member", command
    if not cid:
        return "⚠️ pick a container first", command
    with LOCK:
        client = CLIENTS.get(cid)
        selected_member = _client_member(cid, client) if client else ""
    if selected_member != member:
        config = TEAM_LAYOUT[member]
        return (
            f"⚠️ `{cid}` is not assigned to {config['display']} "
            f"(nodes {config['nodes']}); select one of the pinned containers",
            command,
        )
    return ui_send(cid, session, command, timeout)


def ui_broadcast(session: str, command: str, timeout: int):
    """Queue the same command on the given session for every online container."""
    if not command.strip():
        return "⚠️ empty command", command
    session = (session or "main").strip()
    now = _now()
    sent = []
    with LOCK:
        online = sorted(
            (
                (cid, c) for cid, c in CLIENTS.items()
                if now - c.get("last_seen", 0) < CLIENT_STALE_AFTER
            ),
            key=_client_sort_key,
        )
        for cid, _ in online:
            _queue_exec(cid, session, command, timeout, now)
            sent.append(cid)
    if not sent:
        return "⚠️ no online containers to broadcast to", command
    return (f"📡 broadcast to {len(sent)} container(s) [{session}]: "
            + ", ".join(f"`{c}`" for c in sent), "")


def ui_team_broadcast(member: str, session: str, command: str, timeout: int):
    """Queue one command on the selected member's assigned online nodes."""
    member = _normalize_member(member)
    if not member:
        return "⚠️ choose a valid team member", command
    if not command.strip():
        return "⚠️ empty command", command
    session = (session or "main").strip()
    now = _now()
    sent = []
    with LOCK:
        online = sorted(
            (
                (cid, c) for cid, c in CLIENTS.items()
                if now - c.get("last_seen", 0) < CLIENT_STALE_AFTER
                and _client_member(cid, c) == member
            ),
            key=_client_sort_key,
        )
        for cid, _ in online:
            _queue_exec(cid, session, command, timeout, now)
            sent.append(cid)
    config = TEAM_LAYOUT[member]
    if not sent:
        return (
            f"⚠️ no online containers assigned to {config['display']} "
            f"(nodes {config['nodes']})",
            command,
        )
    return (
        f"📡 sent to {config['display']} ({len(sent)}/2 online) [{session}]: "
        + ", ".join(f"`{cid}`" for cid in sent),
        "",
    )


def ui_put_file(client_label: str, fileobj, dest: str):
    """Queue a push: send an uploaded file to the selected container."""
    cid = _resolve_client_id(client_label)
    if not cid:
        return "⚠️ pick a container first", None
    if not fileobj:
        return "⚠️ choose a file to send", None
    sz = os.path.getsize(fileobj)
    if sz > MAX_FILE_BYTES:
        return f"⚠️ file is {_human(sz)} — over the {_human(MAX_FILE_BYTES)} limit", None
    fname = os.path.basename(fileobj)
    file_id = uuid.uuid4().hex
    saved = FILES_DIR / f"{file_id}__{fname}"
    shutil.copy(fileobj, saved)
    dest = (dest or "").strip() or fname
    cmd_id = uuid.uuid4().hex
    with LOCK:
        PUT_FILES[file_id] = {"path": str(saved), "filename": fname}
        COMMANDS[cmd_id] = {
            "id": cmd_id, "client_id": cid, "session": "file", "type": "put",
            "file_id": file_id, "dest": dest, "filename": fname, "size": sz,
            "command": f"📤 send {fname} ({_human(sz)}) → {dest}",
            "timeout": 300, "status": "pending", "created": _now(),
        }
        _append_and_trim(cid, cmd_id)
    return f"➡️ sending `{fname}` to `{cid}`:`{dest}`", None


def ui_get_file(client_label: str, remote_path: str):
    """Queue a pull: ask the container for a file at `remote_path`."""
    cid = _resolve_client_id(client_label)
    if not cid:
        return "⚠️ pick a container first"
    remote_path = (remote_path or "").strip()
    if not remote_path:
        return "⚠️ enter a path on the container"
    cmd_id = uuid.uuid4().hex
    with LOCK:
        COMMANDS[cmd_id] = {
            "id": cmd_id, "client_id": cid, "session": "file", "type": "get",
            "remote_path": remote_path, "command": f"📥 fetch {remote_path}",
            "timeout": 300, "status": "pending", "created": _now(),
        }
        _append_and_trim(cid, cmd_id)
    return f"➡️ requested `{remote_path}` from `{cid}`"


def _fetched_files(client_label: str) -> list[str]:
    cid = _resolve_client_id(client_label)
    paths = []
    with LOCK:
        for cmd_id in BY_CLIENT.get(cid, []):
            r = COMMANDS.get(cmd_id)
            if r and r.get("type") == "get" and r.get("stored_path") \
                    and os.path.isfile(r["stored_path"]):
                paths.append(r["stored_path"])
    return paths


_STATUS_ICON = {"pending": "⏳", "acknowledged": "📨", "done": "✅"}


def _history_block(record: dict, client_id: str = "") -> str:
    icon = _STATUS_ICON.get(record["status"], "•")
    prefix = f"`{client_id}` " if client_id else ""
    head = (
        f"{icon} {prefix}**[{record.get('session', 'main')}]** "
        f"`{record['command']}`"
    )
    if record["status"] == "done":
        exit_code = record.get("exit_code", "?")
        output = record.get("stdout", "") or "(no output)"
        return head + f"  → exit {exit_code}\n```text\n{output}\n```"
    return head + f"\n_{record['status']}…_"


def ui_history(client_label: str, n: int = 12) -> str:
    cid = _resolve_client_id(client_label)
    if not cid:
        return "_Select a container to see its command history._"
    with LOCK:
        ids = list(BY_CLIENT.get(cid, []))[-int(n):][::-1]
        recs = [dict(COMMANDS[i]) for i in ids if i in COMMANDS]
    if not recs:
        return f"_No commands for `{cid}` yet._"
    return "\n\n".join(_history_block(record) for record in recs)


def ui_team_clients(member: str) -> dict:
    member = _normalize_member(member)
    if not member:
        return {"ok": False, "error": "unknown member", "clients": []}
    now = _now()
    with LOCK:
        assigned = sorted(
            (
                (cid, dict(client)) for cid, client in CLIENTS.items()
                if _client_member(cid, client) == member
            ),
            key=_client_sort_key,
        )
    config = TEAM_LAYOUT[member]
    clients = [
        {
            "client_id": cid,
            "online": now - client.get("last_seen", 0) < CLIENT_STALE_AFTER,
            "node_rank": client.get("node_rank", _client_rank(cid)),
        }
        for cid, client in assigned
    ]
    return {
        "ok": True,
        "member": member,
        "display": config["display"],
        "nodes": config["nodes"],
        "clients": clients,
    }


def ui_team_history(member: str, n: int = 24) -> str:
    member = _normalize_member(member)
    if not member:
        return "_Select a team member to see its two-node history._"
    now = _now()
    with LOCK:
        assigned = sorted(
            (
                (cid, client) for cid, client in CLIENTS.items()
                if _client_member(cid, client) == member
            ),
            key=_client_sort_key,
        )
        records = []
        for cid, _ in assigned:
            for command_id in BY_CLIENT.get(cid, []):
                record = COMMANDS.get(command_id)
                if record:
                    records.append((cid, dict(record)))
        records.sort(key=lambda item: item[1].get("created", 0), reverse=True)
        records = records[:int(n)]
        online = [
            cid for cid, client in assigned
            if now - client.get("last_seen", 0) < CLIENT_STALE_AFTER
        ]
    config = TEAM_LAYOUT[member]
    online_text = ", ".join(f"`{cid}`" for cid in online) or "none"
    header = (
        f"**{config['display']} - nodes {config['nodes']}**  \n"
        f"Online ({len(online)}/2): {online_text}"
    )
    if not records:
        return header + "\n\n_No commands for this member yet._"
    return header + "\n\n" + "\n\n".join(
        _history_block(record, cid) for cid, record in records
    )


def _keep_selection(
    choices: list[tuple[str, str]], client_label: str
) -> str | None:
    current = _canonical_client_id(client_label)
    values = [value for _, value in choices]
    if current in values:
        return current
    return values[0] if values else None


def _stale_page_guard(app_id: int) -> str:
    """Reload an open browser tab after the Space backend is rebuilt."""
    expected_app_id = json.dumps(str(app_id))
    return f"""
<script>
(() => {{
  const expectedAppId = {expected_app_id};
  const configUrl = new URL("./config", window.location.href);
  const checkBackendVersion = async () => {{
    try {{
      const response = await fetch(configUrl, {{cache: "no-store"}});
      if (!response.ok) return;
      const config = await response.json();
      if (
        config.app_id !== undefined
        && String(config.app_id) !== expectedAppId
      ) {{
        window.location.reload();
      }}
    }} catch (_) {{
      // A rebuilding Space is briefly unreachable; retry on the next interval.
    }}
  }};
  window.setInterval(checkBackendVersion, 3000);
}})();
</script>
"""


def ui_refresh(client_label: str, member: str):
    choices = _client_choices(member)
    keep = _keep_selection(choices, client_label)
    hist = ui_history(keep or "")
    team_hist = ui_team_history(member)
    files = _fetched_files(keep or "")
    return (
        gr.update(choices=choices, value=keep),
        hist,
        team_hist,
        gr.update(value=files),
        hist,
        team_hist,
        files,
    )


def ui_select_member(member: str):
    """Pin the selected container and related views to one member's nodes."""
    return ui_refresh("", member)


def ui_select_client(client_label: str, member: str):
    """Refresh views for a valid container in the selected member's pair."""
    choices = _client_choices(member)
    keep = _keep_selection(choices, client_label)
    label_for_hist = keep or ""
    hist = ui_history(label_for_hist)
    files = _fetched_files(label_for_hist)
    return hist, gr.update(value=files), hist, files


def ui_tick(client_label: str, member: str, last_hist: str,
            last_team_hist: str, last_files: list[str]):
    """Refresh passive views without mutating either routing dropdown."""
    choices = _client_choices(member)
    keep = _keep_selection(choices, client_label)
    label_for_hist = keep or ""
    hist = ui_history(label_for_hist)
    hist_out = gr.skip() if hist == last_hist else hist
    team_hist = ui_team_history(member)
    team_hist_out = gr.skip() if team_hist == last_team_hist else team_hist
    files = _fetched_files(label_for_hist)
    files_out = gr.skip() if files == last_files else gr.update(value=files)
    return (
        hist_out, team_hist_out, files_out,
        hist, team_hist, files,
    )


# --------------------------------------------------------------------------- #
# Layout                                                                        #
# --------------------------------------------------------------------------- #
with gr.Blocks(title="NII Relay", analytics_enabled=False) as demo:
    gr.Markdown(
        "# 🛰️ NII Relay\n"
        "Send shell commands to no-SSH containers via a polling daemon. "
        "Run on one container, one member's assigned pair, or all online nodes. "
        "Reuse a bash session name to keep its `cd` and environment.\n\n"
        "**Shared filesystem:** run installs and file mutations on one selected "
        "node only. Use member/all-node broadcast only for node-local commands."
    )

    with gr.Row():
        member_dd = gr.Dropdown(
            label="Team member",
            choices=TEAM_CHOICES,
            value="nguyen",
            interactive=True,
            scale=2,
        )
        client_dd = gr.Dropdown(
            label="Container",
            choices=[],
            interactive=True,
            scale=3,
        )
        refresh_btn = gr.Button("↻ Refresh", scale=1)

    with gr.Row():
        session_tb = gr.Textbox(label="bash session", value="main", scale=2)
        timeout_nb = gr.Number(label="timeout (s)", value=120, precision=0, scale=1)

    command_tb = gr.Textbox(label="command", lines=2,
                            placeholder="nvidia-smi    |    cd /mnt/data && ls -la    |    tail -n 50 train.log")
    with gr.Row():
        send_btn = gr.Button("Run on selected node", variant="secondary")
        team_send_btn = gr.Button("Run on member nodes", variant="primary")
        broadcast_btn = gr.Button("Broadcast to all online", variant="secondary")
        send_status = gr.Markdown("")

    with gr.Tabs():
        with gr.Tab("Selected node history"):
            history_md = gr.Markdown("_Select a container to see its command history._")
        with gr.Tab("Member history"):
            team_history_md = gr.Markdown(ui_team_history("nguyen"))

    with gr.Accordion("📁 File transfer (≤ 100 MB)", open=False):
        with gr.Row():
            with gr.Column():
                gr.Markdown("**Send to container** (push)")
                put_file = gr.File(label="file", type="filepath")
                put_dest = gr.Textbox(label="destination path on container",
                                      placeholder="/mnt/data/  or  /mnt/data/foo.bin")
                put_btn = gr.Button("Send ⬆", variant="primary")
                put_status = gr.Markdown("")
            with gr.Column():
                gr.Markdown("**Fetch from container** (pull)")
                get_path = gr.Textbox(label="path on container",
                                      placeholder="/mnt/data/train.log")
                get_btn = gr.Button("Fetch ⬇", variant="primary")
                get_status = gr.Markdown("")
                downloads = gr.Files(label="fetched files (click to download)",
                                     interactive=False)

    # Remember passive view values so the timer can skip no-op repaints.
    last_hist = gr.State("")
    last_team_hist = gr.State("")
    last_files = gr.State([])

    timer = gr.Timer(3.0)

    # wiring
    send_btn.click(
        ui_send_for_member,
        [member_dd, client_dd, session_tb, command_tb, timeout_nb],
        [send_status, command_tb],
    )
    command_tb.submit(
        ui_send_for_member,
        [member_dd, client_dd, session_tb, command_tb, timeout_nb],
        [send_status, command_tb],
    )
    team_send_btn.click(
        ui_team_broadcast,
        [member_dd, session_tb, command_tb, timeout_nb],
        [send_status, command_tb],
        api_name="ui_team_broadcast",
    )
    broadcast_btn.click(ui_broadcast, [session_tb, command_tb, timeout_nb],
                        [send_status, command_tb])
    refresh_btn.click(
        ui_refresh,
        [client_dd, member_dd],
        [
            client_dd, history_md, team_history_md, downloads,
            last_hist, last_team_hist, last_files,
        ],
        queue=False,
        trigger_mode="always_last",
    )
    client_dd.change(
        ui_select_client,
        [client_dd, member_dd],
        [history_md, downloads, last_hist, last_files],
        queue=False,
        trigger_mode="always_last",
    )
    member_dd.change(
        ui_select_member,
        [member_dd],
        [
            client_dd, history_md, team_history_md, downloads,
            last_hist, last_team_hist, last_files,
        ],
        queue=False,
        trigger_mode="always_last",
    )
    put_btn.click(ui_put_file, [client_dd, put_file, put_dest], [put_status, put_file])
    get_btn.click(ui_get_file, [client_dd, get_path], get_status)
    timer.tick(
        ui_tick,
        [client_dd, member_dd, last_hist, last_team_hist, last_files],
        [
            history_md, team_history_md, downloads,
            last_hist, last_team_hist, last_files,
        ],
        queue=False,
        trigger_mode="always_last",
        concurrency_limit=1,
    )
    demo.load(
        ui_select_member,
        [member_dd],
        [
            client_dd, history_md, team_history_md, downloads,
            last_hist, last_team_hist, last_files,
        ],
        queue=False,
    )

    # --- hidden machine API endpoints (called by the daemon) ---
    # Auth is the private Space + HF token (gating who can reach these at all);
    # there is no separate app-level secret.
    with gr.Row(visible=False):
        a_cid = gr.Textbox()
        a_meta = gr.Textbox()
        a_cmdid = gr.Textbox()
        a_session = gr.Textbox()
        a_exit = gr.Number()
        a_stdout = gr.Textbox()
        a_fileid = gr.Textbox()
        a_err = gr.Textbox()
        a_member = gr.Textbox()
        a_ui_status = gr.Textbox()
        a_ui_command = gr.Textbox()
        a_file_in = gr.File()
        a_file_out = gr.File()
        a_out = gr.JSON()
        b_reg = gr.Button()
        b_poll = gr.Button()
        b_res = gr.Button()
        b_fetch = gr.Button()
        b_upload = gr.Button()
        b_send = gr.Button()
        b_history = gr.Button()
        b_team_clients = gr.Button()
        b_team_history = gr.Button()
        b_reg.click(api_register, [a_cid, a_meta], a_out, api_name="register")
        b_poll.click(api_poll, [a_cid], a_out, api_name="poll")
        b_res.click(api_result, [a_cmdid, a_cid, a_exit, a_stdout, a_session],
                    a_out, api_name="result")
        b_fetch.click(api_fetch_file, [a_fileid], a_file_out, api_name="fetch_file")
        b_upload.click(api_upload_file, [a_cmdid, a_cid, a_file_in, a_err],
                       a_out, api_name="upload_result")
        b_send.click(
            ui_send,
            [a_cid, a_session, a_stdout, a_exit],
            [a_ui_status, a_ui_command],
            api_name="ui_send",
        )
        b_history.click(
            ui_history,
            [a_cid],
            a_ui_status,
            api_name="ui_history",
        )
        b_team_clients.click(
            ui_team_clients,
            [a_member],
            a_out,
            api_name="ui_team_clients",
        )
        b_team_history.click(
            ui_team_history,
            [a_member],
            a_ui_status,
            api_name="ui_team_history",
        )


threading.Thread(target=_file_reaper, daemon=True).start()

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=16).launch(
        server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)),
        theme=gr.themes.Soft(), max_file_size="150mb", ssr_mode=False,
        head=_stale_page_guard(demo.app_id),
    )
