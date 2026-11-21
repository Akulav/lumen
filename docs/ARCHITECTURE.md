# Lumen architecture

Lumen is a desktop app for chatting with local language models. It is a **Tauri 2** shell with a **Rust** backend
and a plain **HTML/CSS/JS** window (no framework, no bundler). The model runs in a separate **llama.cpp
`llama-server`** process that Lumen downloads and manages.

```
┌──────────────── Lumen (one process) ────────────────┐
│  Window (WebView): ui/index.html, app.js, i18n.js   │
│        │ invoke(cmd)              ▲ events          │
│        ▼                          │                 │
│  Rust backend: commands.rs → chat / tools / rag /   │
│  research / models / runtime / updater / mcp …      │
└──────┬───────────────┬──────────────┬───────────────┘
       │ HTTP + key    │ HTTP + key   │ stdio / HTTP
       ▼               ▼              ▼
  llama-server     llama-server    MCP servers
  (chat model,     (embeddings,    (configured by
   GPU)             CPU)            the user)
```

## Processes

| Process | Started by | Purpose |
| :-- | :-- | :-- |
| `lumen` | the user | window, tray, all logic |
| `llama-server` (chat) | `server.rs` `start_llm` | the loaded model, `127.0.0.1:<random port>` |
| `llama-server` (embeddings) | `server.rs` `ensure_embed` | Qwen3-Embedding 0.6B on the CPU, for project search |
| MCP servers | `mcp.rs` `connect` | tools from Model Context Protocol servers (stdio ones are child processes) |
| `git`, user-allowed commands | `codetools.rs` | project tools, only after the user allows them |

Every child process goes through `platform::quiet` (no console window; on Linux a parent-death signal) and is
adopted by a Windows job object, so nothing outlives Lumen, even after a crash.

## Modules (`src-tauri/src`)

| Module | Role |
| :-- | :-- |
| `main.rs` | app setup, window, tray, close-to-tray, shutdown of child processes |
| `commands.rs` | every `#[tauri::command]` the window calls (thin wrappers) |
| `state.rs` | `AppState`, `Settings` (`LumenData/data/settings.json`), paths, logs |
| `db.rs` | SQLite: projects, chats, messages, chunks, files |
| `server.rs` | starts/stops the llama-servers, health checks, context size, prompt-cache cap |
| `runtime.rs` | picks and installs the llama.cpp build for this OS and GPU backend |
| `models.rs` | curated catalog, local models, Hugging Face search and downloads |
| `download.rs` | resumable downloads with progress events and SHA-256 verification |
| `llm.rs` | streaming client for `/v1/chat/completions` |
| `chat.rs` | prompt assembly, the tool loop, titles, context budget and compression, export |
| `tools.rs` | model tools: project files (sandboxed), web, calculator, dispatch to `codetools` and `mcp` |
| `codetools.rs` | grep, outline, symbols, compare, git, `run_command` with confirmation |
| `audit.rs` | whole-codebase review in batches |
| `rag.rs` | chunking, BM25, embeddings, rank fusion, incremental index |
| `research.rs` | deep research: plan → search → read → reflect → cited report |
| `web.rs` | search through official APIs, page text extraction |
| `mcp.rs` | MCP client: config parsing, stdio and Streamable HTTP transports, tool listing and calls |
| `metrics.rs` | 1 s sampler (RAM, VRAM, GPU/CPU, server counters), idle unload, crash detection |
| `updater.rs` | update checks and installs (Lumen and llama.cpp) |
| `install.rs` | Linux/macOS self-install, `--here`, `--uninstall` |
| `win.rs` / `unix.rs` | platform specifics (`mod platform`) |

## Data flow of a message

1. The window calls `send_message`; `chat.rs` stores the message and starts a turn (it keeps running when the user
   switches chats; `get_chat` hands the live text back).
2. The prompt is the system prompt (byte-stable for llama.cpp's prompt cache), the history and, in a project,
   retrieved snippets. Tools: the built-in ones plus every tool of the running MCP servers.
3. `llm.rs` streams the answer; tokens reach the window as `chat-event` (`delta`, `progress`…).
4. Tool calls run through `tools::run`. Changes to files get a backup first and show as diff cards with **Undo**;
   processes and MCP tools wait for the user's answer (`tool_confirm` → `confirm_tool`).
5. The loop ends with a text answer (at most 40 rounds; a last round without tools forces one).

## Security model

- **Window**: strict CSP (`default-src 'self'`, no remote scripts, styles, fonts or images; IPC only). Model output is
  Markdown rendered by marked and sanitised by DOMPurify; tool results are inserted as text. Links open in the
  system browser only for `http(s)`.
- **Capabilities**: only `core:default`; the dialog and opener plugins are used from Rust, not from the window.
- **Project files**: `tools::resolve` keeps every path inside the project folder, also through symlinks/junctions.
- **Processes**: `run_command`, git and MCP tools run only after the user allows them (once, or for the chat).
- **Links**: with Web off, the model may only open links the user gave (messages or project files) — never links
  from web pages or MCP results.
- **Local servers**: llama-server listens on `127.0.0.1` and requires a random per-run API key (`--api-key`), so
  other programs and web pages can't drive the model.
- **Downloads**: Lumen updates must match the release's `SHA256SUMS.txt`; the llama.cpp engine must match the
  SHA-256 GitHub publishes for the asset (a release without one is skipped); models and the embedding model are
  checked against Hugging Face's LFS SHA-256. A mismatching file is deleted.

## Storage

Everything lives in `LumenData` next to the program (macOS: next to `Lumen.app`): `models/`, `runtime/`,
`data/` (SQLite database, settings, backups for Undo), `logs/` and `webview/`. Only when that folder isn't
writable does Lumen use the OS app-data folder.

## Releases

`package.json` holds the version. Pushing a version bump to `main` runs `.github/workflows/release.yml`: tests on
Windows and Ubuntu, builds for Windows (Inno Setup), Linux and macOS, then a GitHub release with exactly
`Lumen-setup.exe`, `lumen-linux-x64`, `lumen-macos-arm64` and `SHA256SUMS.txt`.
