//! MCP (Model Context Protocol) client: tools from MCP servers the user configured in Settings → MCP
//! servers, in the usual `mcpServers` JSON (the same shape other MCP apps use):
//!
//! ```json
//! { "mcpServers": {
//!     "files": { "command": "npx", "args": ["-y", "@modelcontextprotocol/server-filesystem", "C:/notes"] },
//!     "remote": { "url": "https://example.com/mcp", "headers": { "Authorization": "Bearer …" } }
//! } }
//! ```
//!
//! A server with `command` runs as a child process (stdio: one JSON-RPC message per line); one with `url`
//! is reached over Streamable HTTP (POST, answered with JSON or an SSE stream). Optional per server:
//! `args`, `env`, `cwd`, `headers`, `disabled: true`, and `trust: true` (its tools run without asking).
//!
//! Servers start when a chat first needs tools (or on Reconnect in Settings) and stay up until the
//! configuration changes or Lumen quits. Their tools reach the model as `mcp__<server>__<tool>`; every
//! call is confirmed in the chat like `run_command` (Allow / Always in this chat / Deny) unless the server
//! is trusted. What a tool returns is treated like a web page: data for the model, never a source of links
//! that may be opened with Web off.

use anyhow::{anyhow, Context, Result};
use parking_lot::Mutex;
use serde::Serialize;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tauri::{AppHandle, Emitter, Manager};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};

use crate::state::{push_log, AppState, LUMEN_UA};

/// Protocol revision Lumen speaks (servers answer with the one they use; older ones work the same here).
const PROTOCOL: &str = "2025-06-18";
const INIT_TIMEOUT: Duration = Duration::from_secs(45);
const CALL_TIMEOUT: Duration = Duration::from_secs(600);
/// Tools per server and description length sent to the model (keeps the prompt within reason).
const MAX_TOOLS: usize = 64;
const MAX_DESC: usize = 1024;

/// One server's entry from the configuration.
#[derive(Clone, Debug, PartialEq)]
pub struct ServerConfig {
    pub name: String,
    pub command: Option<String>,
    pub args: Vec<String>,
    pub env: Vec<(String, String)>,
    pub cwd: Option<String>,
    pub url: Option<String>,
    pub headers: Vec<(String, String)>,
    pub trust: bool,
    pub disabled: bool,
}

/// Parses the Settings text: `{"mcpServers": {...}}` or the bare `{name: {...}}` map. Empty = no servers.
pub fn parse_config(text: &str) -> Result<Vec<ServerConfig>> {
    if text.trim().is_empty() {
        return Ok(vec![]);
    }
    let v: Value = serde_json::from_str(text).context("MCP servers: not valid JSON")?;
    let map = v.get("mcpServers").or_else(|| v.get("servers")).unwrap_or(&v);
    let map = map.as_object().ok_or_else(|| anyhow!("MCP servers: expected an object of servers"))?;
    let strs = |x: &Value| -> Vec<String> { x.as_array().into_iter().flatten().filter_map(|s| s.as_str().map(String::from)).collect() };
    let pairs = |x: &Value| -> Vec<(String, String)> {
        x.as_object().into_iter().flatten().filter_map(|(k, v)| Some((k.clone(), v.as_str()?.to_string()))).collect()
    };
    let mut out = Vec::new();
    for (name, s) in map {
        let text = |k: &str| s[k].as_str().map(str::trim).filter(|x| !x.is_empty()).map(String::from);
        let c = ServerConfig {
            name: name.clone(),
            command: text("command"),
            args: strs(&s["args"]),
            env: pairs(&s["env"]),
            cwd: text("cwd"),
            url: text("url"),
            headers: pairs(&s["headers"]),
            trust: s["trust"].as_bool().unwrap_or(false),
            disabled: s["disabled"].as_bool().unwrap_or(false),
        };
        if c.command.is_none() && c.url.is_none() {
            return Err(anyhow!("MCP server `{name}` needs a \"command\" (local program) or a \"url\""));
        }
        if let Some(u) = &c.url {
            if !(u.starts_with("https://") || u.starts_with("http://")) {
                return Err(anyhow!("MCP server `{name}`: the url must start with http:// or https://"));
            }
        }
        out.push(c);
    }
    out.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(out)
}

/// `[A-Za-z0-9_-]` only (what tool names may contain); other characters become `_`.
fn clean(s: &str) -> String {
    s.chars().map(|c| if c.is_ascii_alphanumeric() || c == '-' { c } else { '_' }).collect()
}

/// The name the model sees: `mcp__<server>__<tool>`, at most 64 characters.
fn exposed_name(server: &str, tool: &str) -> String {
    let name = format!("mcp__{}__{}", clean(server), clean(tool));
    name.chars().take(64).collect()
}

#[derive(Clone, Debug, Serialize)]
pub struct ToolInfo {
    /// as the server names it
    pub name: String,
    /// as the model sees it
    pub exposed: String,
    pub description: String,
    #[serde(skip)]
    pub schema: Value,
}

/// What Settings shows per server.
#[derive(Clone, Debug, Serialize)]
pub struct Status {
    pub name: String,
    /// "off" | "connecting" | "ready" | "error"
    pub state: String,
    pub error: String,
    pub tools: Vec<String>,
    pub transport: String,
    pub trust: bool,
}

type Pending = Arc<Mutex<HashMap<u64, tokio::sync::oneshot::Sender<Value>>>>;

enum Conn {
    Stdio {
        stdin: Arc<tokio::sync::Mutex<tokio::process::ChildStdin>>,
        child: Mutex<tokio::process::Child>,
        pending: Pending,
    },
    Http {
        url: String,
        headers: reqwest::header::HeaderMap,
        session: Mutex<Option<String>>,
        version: Mutex<Option<String>>,
    },
}

struct Server {
    cfg: ServerConfig,
    conn: Conn,
    next_id: AtomicU64,
    tools: Vec<ToolInfo>,
}

/// All configured servers. Lives in `AppState::mcp`.
#[derive(Default)]
pub struct Mcp {
    /// config text the servers below were started from
    config: Mutex<String>,
    servers: Mutex<HashMap<String, Arc<Server>>>,
    status: Mutex<HashMap<String, Status>>,
    /// a connect pass is running (others wait for it instead of starting the servers twice)
    connecting: tokio::sync::Mutex<()>,
}

impl Mcp {
    /// Stops every server (configuration changed, Reconnect, quit).
    pub fn reset(&self) {
        let servers: Vec<Arc<Server>> = self.servers.lock().drain().map(|(_, s)| s).collect();
        for s in servers {
            s.shutdown();
        }
        self.status.lock().clear();
        self.config.lock().clear();
    }

    pub fn statuses(&self, state: &AppState) -> Value {
        let cfgs = parse_config(&state.settings.read().mcp_config);
        let st = self.status.lock();
        match cfgs {
            Err(e) => json!({"error": e.to_string(), "servers": []}),
            Ok(cfgs) => {
                let list: Vec<Status> = cfgs
                    .iter()
                    .map(|c| {
                        st.get(&c.name).cloned().unwrap_or_else(|| Status {
                            name: c.name.clone(),
                            state: if c.disabled { "off".into() } else { "idle".into() },
                            error: String::new(),
                            tools: vec![],
                            transport: transport(c).into(),
                            trust: c.trust,
                        })
                    })
                    .collect();
                json!({"error": "", "servers": list})
            }
        }
    }
}

fn transport(c: &ServerConfig) -> &'static str {
    if c.command.is_some() { "stdio" } else { "http" }
}

impl Server {
    fn shutdown(&self) {
        if let Conn::Stdio { child, .. } = &self.conn {
            let mut c = child.lock();
            let _ = c.start_kill();
        }
    }

    fn alive(&self) -> bool {
        match &self.conn {
            Conn::Stdio { child, .. } => child.lock().try_wait().ok().flatten().is_none(),
            Conn::Http { .. } => true,
        }
    }

    async fn notify(&self, state: &AppState, method: &str, params: Value) -> Result<()> {
        let msg = json!({"jsonrpc": "2.0", "method": method, "params": params});
        match &self.conn {
            Conn::Stdio { stdin, .. } => write_line(stdin, &msg).await,
            Conn::Http { .. } => self.http_post(state, &msg, None).await.map(|_| ()),
        }
    }

    async fn request(&self, state: &AppState, method: &str, params: Value, timeout: Duration, cancel: Option<&AtomicBool>) -> Result<Value> {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let msg = json!({"jsonrpc": "2.0", "id": id, "method": method, "params": params});
        let fut = async {
            match &self.conn {
                Conn::Stdio { stdin, pending, .. } => {
                    let (tx, rx) = tokio::sync::oneshot::channel();
                    pending.lock().insert(id, tx);
                    if let Err(e) = write_line(stdin, &msg).await {
                        pending.lock().remove(&id);
                        return Err(e);
                    }
                    rx.await.map_err(|_| anyhow!("the server closed"))
                }
                Conn::Http { .. } => self.http_post(state, &msg, Some(id)).await?.ok_or_else(|| anyhow!("the server sent no answer")),
            }
        };
        let stopped = async {
            loop {
                tokio::time::sleep(Duration::from_millis(250)).await;
                if cancel.is_some_and(|c| c.load(Ordering::Relaxed)) {
                    break;
                }
            }
        };
        let reply = tokio::select! {
            r = tokio::time::timeout(timeout, fut) => match r {
                Ok(r) => r,
                Err(_) => Err(anyhow!("no answer within {} s", timeout.as_secs())),
            },
            _ = stopped => Err(anyhow!("stopped")),
        };
        if let Conn::Stdio { pending, .. } = &self.conn {
            pending.lock().remove(&id);
        }
        let reply = match reply {
            Ok(r) => r,
            Err(e) => {
                let _ = self.notify(state, "notifications/cancelled", json!({"requestId": id, "reason": e.to_string()})).await;
                return Err(e);
            }
        };
        if let Some(err) = reply.get("error") {
            return Err(anyhow!("{}", err["message"].as_str().unwrap_or("error from the MCP server")));
        }
        Ok(reply.get("result").cloned().unwrap_or(Value::Null))
    }

    /// Streamable HTTP: one POST per message; the answer is JSON or an SSE stream that carries it.
    async fn http_post(&self, state: &AppState, msg: &Value, id: Option<u64>) -> Result<Option<Value>> {
        let Conn::Http { url, headers, session, version } = &self.conn else { unreachable!() };
        let mut req = state
            .http
            .post(url)
            .headers(headers.clone())
            .header("User-Agent", LUMEN_UA)
            .header("Accept", "application/json, text/event-stream")
            .json(msg);
        if let Some(s) = session.lock().clone() {
            req = req.header("Mcp-Session-Id", s);
        }
        if let Some(v) = version.lock().clone() {
            req = req.header("MCP-Protocol-Version", v);
        }
        let resp = req.send().await?;
        if let Some(s) = resp.headers().get("mcp-session-id").and_then(|v| v.to_str().ok()) {
            *session.lock() = Some(s.to_string());
        }
        let status = resp.status();
        if !status.is_success() {
            let body: String = resp.text().await.unwrap_or_default().chars().take(300).collect();
            return Err(anyhow!("HTTP {status} {body}"));
        }
        let Some(id) = id else { return Ok(None) };
        let sse = resp.headers().get("content-type").and_then(|v| v.to_str().ok()).is_some_and(|c| c.contains("text/event-stream"));
        if !sse {
            let v: Value = resp.json().await?;
            return Ok(find_reply(&v, id));
        }
        // read events until the one that answers this request
        use futures_util::StreamExt;
        let mut stream = resp.bytes_stream();
        let mut buf = String::new();
        while let Some(chunk) = stream.next().await {
            buf.push_str(&String::from_utf8_lossy(&chunk?));
            while let Some(end) = buf.find("\n\n").or_else(|| buf.find("\r\n\r\n")) {
                let event: String = buf.drain(..end).collect();
                buf = buf.trim_start_matches(['\r', '\n']).to_string();
                let data: String = event.lines().filter_map(|l| l.strip_prefix("data:")).map(str::trim_start).collect::<Vec<_>>().join("\n");
                if let Ok(v) = serde_json::from_str::<Value>(&data) {
                    if let Some(r) = find_reply(&v, id) {
                        return Ok(Some(r));
                    }
                }
            }
        }
        Ok(None)
    }
}

/// The JSON-RPC answer with `id` in a message or a batch.
fn find_reply(v: &Value, id: u64) -> Option<Value> {
    let matches = |m: &Value| m["id"].as_u64() == Some(id) && (m.get("result").is_some() || m.get("error").is_some());
    match v {
        Value::Array(a) => a.iter().find(|m| matches(m)).cloned(),
        m if matches(m) => Some(m.clone()),
        _ => None,
    }
}

async fn write_line(stdin: &tokio::sync::Mutex<tokio::process::ChildStdin>, msg: &Value) -> Result<()> {
    let mut line = serde_json::to_string(msg)?;
    line.push('\n');
    let mut w = stdin.lock().await;
    w.write_all(line.as_bytes()).await?;
    w.flush().await?;
    Ok(())
}

/// Finds a program the way a shell would (Windows: `npx` → `npx.cmd` through PATH and PATHEXT).
fn find_program(cmd: &str) -> std::path::PathBuf {
    let p = std::path::Path::new(cmd);
    if p.components().count() > 1 || p.extension().is_some() {
        return p.to_path_buf();
    }
    let exts: Vec<String> = if cfg!(windows) {
        std::env::var("PATHEXT").unwrap_or_else(|_| ".COM;.EXE;.BAT;.CMD".into()).split(';').filter(|e| !e.is_empty()).map(|e| e.to_ascii_lowercase()).collect()
    } else {
        vec![String::new()]
    };
    for dir in std::env::var_os("PATH").map(|p| std::env::split_paths(&p).collect::<Vec<_>>()).unwrap_or_default() {
        for ext in &exts {
            let cand = dir.join(format!("{cmd}{ext}"));
            if cand.is_file() {
                return cand;
            }
        }
    }
    p.to_path_buf()
}

async fn start(state: &AppState, cfg: &ServerConfig) -> Result<Server> {
    let conn = if let Some(command) = &cfg.command {
        let mut cmd = tokio::process::Command::new(find_program(command));
        cmd.args(&cfg.args)
            .envs(cfg.env.iter().map(|(k, v)| (k.as_str(), v.as_str())))
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .kill_on_drop(true);
        if let Some(d) = &cfg.cwd {
            cmd.current_dir(d);
        }
        crate::platform::quiet(&mut cmd);
        let mut child = cmd.spawn().with_context(|| format!("could not start `{command}`"))?;
        state.job.adopt(child.id().unwrap_or(0));
        let stdin = Arc::new(tokio::sync::Mutex::new(child.stdin.take().ok_or_else(|| anyhow!("no stdin"))?));
        let stdout = child.stdout.take().ok_or_else(|| anyhow!("no stdout"))?;
        if let Some(err) = child.stderr.take() {
            let (logs, tag) = (state.logs.clone(), format!("[mcp:{}]", cfg.name));
            tokio::spawn(async move {
                let mut r = BufReader::new(err).lines();
                while let Ok(Some(line)) = r.next_line().await {
                    push_log(&logs, format!("{tag} {line}"));
                }
            });
        }
        let pending: Pending = Arc::default();
        let (reader_pending, reader_stdin) = (pending.clone(), stdin.clone());
        // answers go to whoever waits for them; requests from the server get a short answer
        tokio::spawn(async move {
            let mut r = BufReader::new(stdout).lines();
            while let Ok(Some(line)) = r.next_line().await {
                let Ok(v) = serde_json::from_str::<Value>(line.trim()) else { continue };
                for m in if let Value::Array(a) = v { a } else { vec![v] } {
                    let is_reply = m.get("result").is_some() || m.get("error").is_some();
                    if let (Some(id), true) = (m["id"].as_u64(), is_reply) {
                        if let Some(tx) = reader_pending.lock().remove(&id) {
                            let _ = tx.send(m);
                        }
                    } else if m.get("method").is_some() && m.get("id").is_some() {
                        let answer = if m["method"] == "ping" {
                            json!({"jsonrpc": "2.0", "id": m["id"], "result": {}})
                        } else {
                            json!({"jsonrpc": "2.0", "id": m["id"], "error": {"code": -32601, "message": "not supported by Lumen"}})
                        };
                        let _ = write_line(&reader_stdin, &answer).await;
                    }
                }
            }
            // (the process ended: waiting requests fail as their senders drop)
            reader_pending.lock().clear();
        });
        Conn::Stdio { stdin, child: Mutex::new(child), pending }
    } else {
        let url = cfg.url.clone().unwrap_or_default();
        let mut headers = reqwest::header::HeaderMap::new();
        for (k, v) in &cfg.headers {
            let name = reqwest::header::HeaderName::from_bytes(k.as_bytes()).with_context(|| format!("bad header name `{k}`"))?;
            headers.insert(name, v.parse().with_context(|| format!("bad value for header `{k}`"))?);
        }
        Conn::Http { url, headers, session: Mutex::new(None), version: Mutex::new(None) }
    };
    let mut server = Server { cfg: cfg.clone(), conn, next_id: AtomicU64::new(1), tools: vec![] };
    if let Err(e) = initialize(state, &mut server).await {
        server.shutdown();
        return Err(e);
    }
    Ok(server)
}

/// The MCP handshake, then the tool list.
async fn initialize(state: &AppState, server: &mut Server) -> Result<()> {
    let params = json!({
        "protocolVersion": PROTOCOL,
        "capabilities": {},
        "clientInfo": {"name": "Lumen", "version": env!("CARGO_PKG_VERSION")}
    });
    let init = server.request(state, "initialize", params, INIT_TIMEOUT, None).await.context("no MCP handshake")?;
    if let Conn::Http { version, .. } = &server.conn {
        *version.lock() = Some(init["protocolVersion"].as_str().unwrap_or(PROTOCOL).to_string());
    }
    server.notify(state, "notifications/initialized", json!({})).await?;
    let mut tools = Vec::new();
    let mut cursor: Option<String> = None;
    for _ in 0..20 {
        let params = match &cursor {
            Some(c) => json!({"cursor": c}),
            None => json!({}),
        };
        let r = server.request(state, "tools/list", params, INIT_TIMEOUT, None).await?;
        tools.extend(parse_tools(&server.cfg.name, &r));
        cursor = r["nextCursor"].as_str().map(String::from);
        if cursor.is_none() || tools.len() >= MAX_TOOLS {
            break;
        }
    }
    tools.truncate(MAX_TOOLS);
    server.tools = tools;
    Ok(())
}

fn parse_tools(server: &str, list: &Value) -> Vec<ToolInfo> {
    list["tools"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|t| {
            let name = t["name"].as_str()?;
            let mut schema = t["inputSchema"].clone();
            if !schema.is_object() {
                schema = json!({"type": "object", "properties": {}});
            }
            let description: String = t["description"].as_str().unwrap_or("").chars().take(MAX_DESC).collect();
            Some(ToolInfo { name: name.to_string(), exposed: exposed_name(server, name), description, schema })
        })
        .collect()
}

/// Starts every enabled server that isn't running yet (in parallel). With `retry`, servers that failed
/// before are tried again (Reconnect); otherwise a failed server waits for Reconnect or a config change.
pub async fn connect(app: &AppHandle, retry: bool) {
    let state = app.state::<Arc<AppState>>().inner().clone();
    let mcp = &state.mcp;
    let _pass = mcp.connecting.lock().await;
    let text = state.settings.read().mcp_config.clone();
    if *mcp.config.lock() != text {
        mcp.reset();
        *mcp.config.lock() = text.clone();
    }
    let Ok(cfgs) = parse_config(&text) else { return };
    let todo: Vec<ServerConfig> = cfgs
        .into_iter()
        .filter(|c| !c.disabled)
        .filter(|c| {
            let running = mcp.servers.lock().get(&c.name).is_some_and(|s| s.alive());
            let failed = mcp.status.lock().get(&c.name).is_some_and(|s| s.state == "error");
            !running && (retry || !failed)
        })
        .collect();
    if todo.is_empty() {
        return;
    }
    for c in &todo {
        mcp.status.lock().insert(c.name.clone(), Status { name: c.name.clone(), state: "connecting".into(), error: String::new(), tools: vec![], transport: transport(c).into(), trust: c.trust });
    }
    let _ = app.emit("mcp-status", mcp.statuses(&state));
    let started = futures_util::future::join_all(todo.iter().map(|c| start(&state, c))).await;
    for (c, r) in todo.into_iter().zip(started) {
        let mut st = Status { name: c.name.clone(), state: "ready".into(), error: String::new(), tools: vec![], transport: transport(&c).into(), trust: c.trust };
        match r {
            Ok(server) => {
                st.tools = server.tools.iter().map(|t| t.name.clone()).collect();
                state.log(format!("[mcp] {} ready: {} tools", c.name, st.tools.len()));
                if let Some(old) = mcp.servers.lock().insert(c.name.clone(), Arc::new(server)) {
                    old.shutdown();
                }
            }
            Err(e) => {
                st.state = "error".into();
                st.error = format!("{e:#}");
                state.log(format!("[mcp] {} failed: {}", c.name, st.error));
            }
        }
        mcp.status.lock().insert(c.name.clone(), st);
    }
    let _ = app.emit("mcp-status", mcp.statuses(&state));
}

/// Tool definitions for the model from every running server (sorted by server: a stable prompt).
pub async fn definitions(app: &AppHandle) -> Vec<Value> {
    let state = app.state::<Arc<AppState>>().inner().clone();
    if parse_config(&state.settings.read().mcp_config).map(|c| c.iter().all(|s| s.disabled)).unwrap_or(true) {
        return vec![];
    }
    connect(app, false).await;
    let servers = state.mcp.servers.lock();
    let mut names: Vec<&String> = servers.keys().collect();
    names.sort();
    let mut v = Vec::new();
    for n in names {
        let s = &servers[n];
        for t in &s.tools {
            let desc = if t.description.is_empty() { format!("Tool `{}` of the MCP server `{}`.", t.name, n) } else { format!("[MCP server `{n}`] {}", t.description) };
            v.push(json!({"type": "function", "function": {"name": t.exposed, "description": desc, "parameters": t.schema}}));
        }
    }
    v
}

pub fn is_mcp_tool(name: &str) -> bool {
    name.starts_with("mcp__")
}

/// Runs an MCP tool for the model (after the user allowed it, unless the server is trusted).
pub async fn call(ctx: &crate::tools::ToolCtx<'_>, call_id: &str, exposed: &str, args: &Value) -> Result<crate::tools::ToolOutput> {
    let state = ctx.app.state::<Arc<AppState>>().inner().clone();
    let found = {
        let servers = state.mcp.servers.lock();
        servers.values().find_map(|s| s.tools.iter().find(|t| t.exposed == exposed).map(|t| (s.clone(), t.name.clone())))
    };
    let (server, tool) = found.ok_or_else(|| anyhow!("the MCP tool `{exposed}` is not available (its server is not running)"))?;
    if !server.alive() {
        return Err(anyhow!("the MCP server `{}` has stopped; reconnect it in Settings", server.cfg.name));
    }
    let args = if args.is_object() { args.clone() } else { json!({}) };
    if !server.cfg.trust {
        let shown: String = serde_json::to_string_pretty(&args).unwrap_or_default().chars().take(1500).collect();
        let command = format!("{} › {}\n{}", server.cfg.name, tool, shown);
        crate::codetools::confirm(ctx.app, ctx.chat_id, call_id, &format!("mcp:{}", server.cfg.name), &command, "", ctx.cancel).await?;
    }
    let r = server.request(&state, "tools/call", json!({"name": tool, "arguments": args}), CALL_TIMEOUT, Some(ctx.cancel)).await?;
    let (text, is_error) = result_text(&r);
    let display = json!({"kind": "mcp", "server": server.cfg.name, "tool": tool, "ok": !is_error});
    Ok(crate::tools::ToolOutput { content: if text.is_empty() { "(no output)".into() } else { text }, display, ok: !is_error })
}

/// A `tools/call` result as text for the model: text parts as they are, other parts described.
fn result_text(r: &Value) -> (String, bool) {
    let mut parts = Vec::new();
    for c in r["content"].as_array().into_iter().flatten() {
        match c["type"].as_str().unwrap_or("") {
            "text" => parts.push(c["text"].as_str().unwrap_or("").to_string()),
            "image" | "audio" => parts.push(format!("[{} ({}) - not shown]", c["type"].as_str().unwrap_or(""), c["mimeType"].as_str().unwrap_or("?"))),
            "resource" => {
                let res = &c["resource"];
                parts.push(res["text"].as_str().map(String::from).unwrap_or_else(|| format!("[resource {}]", res["uri"].as_str().unwrap_or("?"))));
            }
            "resource_link" => parts.push(format!("[resource {} {}]", c["name"].as_str().unwrap_or(""), c["uri"].as_str().unwrap_or(""))),
            _ => {}
        }
    }
    if parts.is_empty() {
        if let Some(s) = r.get("structuredContent").filter(|s| !s.is_null()) {
            parts.push(s.to_string());
        }
    }
    (parts.join("\n"), r["isError"].as_bool().unwrap_or(false))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_accepts_the_usual_shapes() {
        let c = parse_config(r#"{"mcpServers": {"b": {"url": "https://x/mcp", "headers": {"Authorization": "Bearer k"}}, "a": {"command": "npx", "args": ["-y", "pkg"], "env": {"K": "v"}, "trust": true}}}"#).unwrap();
        assert_eq!(c.iter().map(|s| s.name.as_str()).collect::<Vec<_>>(), ["a", "b"], "sorted");
        assert_eq!(c[0].command.as_deref(), Some("npx"));
        assert_eq!(c[0].args, ["-y", "pkg"]);
        assert_eq!(c[0].env, [("K".to_string(), "v".to_string())]);
        assert!(c[0].trust && !c[1].trust);
        assert_eq!(c[1].headers[0].1, "Bearer k");
        assert_eq!(parse_config(r#"{"x": {"command": "srv"}}"#).unwrap().len(), 1, "a bare map works too");
        assert!(parse_config("").unwrap().is_empty());
        assert!(parse_config("{").is_err());
        assert!(parse_config(r#"{"mcpServers": {"x": {}}}"#).is_err(), "needs command or url");
        assert!(parse_config(r#"{"mcpServers": {"x": {"url": "file:///etc"}}}"#).is_err());
    }

    #[test]
    fn exposed_names_are_valid_tool_names() {
        assert_eq!(exposed_name("my files", "read.file"), "mcp__my_files__read_file");
        assert!(exposed_name(&"s".repeat(50), &"t".repeat(50)).len() <= 64);
        assert!(is_mcp_tool("mcp__a__b") && !is_mcp_tool("read_file"));
    }

    #[test]
    fn results_become_text() {
        let r = json!({"content": [{"type": "text", "text": "hello"}, {"type": "image", "mimeType": "image/png", "data": "…"}], "isError": false});
        assert_eq!(result_text(&r), ("hello\n[image (image/png) - not shown]".to_string(), false));
        let r = json!({"content": [], "structuredContent": {"n": 1}, "isError": true});
        assert_eq!(result_text(&r), ("{\"n\":1}".to_string(), true));
        assert_eq!(find_reply(&json!([{"id": 1, "method": "x"}, {"id": 2, "result": {}}]), 2), Some(json!({"id": 2, "result": {}})));
        assert_eq!(find_reply(&json!({"id": 3, "result": 1}), 2), None);
    }

    /// A real stdio server: a tiny MCP server written as a script for the shell this system has,
    /// started like any configured server (handshake, tools/list, tools/call).
    #[test]
    fn talks_to_a_stdio_server() {
        let dir = std::env::temp_dir().join(format!("lumen-mcp-{}", crate::state::new_id()));
        std::fs::create_dir_all(&dir).unwrap();
        let script = dir.join("server.js");
        std::fs::write(&script, TEST_SERVER_JS).unwrap();
        let Ok(node) = std::process::Command::new(find_program("node")).arg("--version").output() else {
            eprintln!("node is not installed - skipped");
            return;
        };
        assert!(node.status.success());
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let state = AppState::new().unwrap();
            let cfg = ServerConfig { name: "echo".into(), command: Some("node".into()), args: vec![script.to_string_lossy().into()], env: vec![], cwd: None, url: None, headers: vec![], trust: true, disabled: false };
            let server = start(&state, &cfg).await.unwrap();
            assert_eq!(server.tools.len(), 1);
            assert_eq!(server.tools[0].exposed, "mcp__echo__echo");
            let r = server.request(&state, "tools/call", json!({"name": "echo", "arguments": {"text": "hi"}}), Duration::from_secs(10), None).await.unwrap();
            assert_eq!(result_text(&r), ("echo: hi".to_string(), false));
            server.shutdown();
        });
        let _ = std::fs::remove_dir_all(&dir);
    }

    const TEST_SERVER_JS: &str = r#"
const rl = require('readline').createInterface({ input: process.stdin });
const send = (m) => process.stdout.write(JSON.stringify(m) + '\n');
rl.on('line', (line) => {
  const m = JSON.parse(line);
  if (m.id === undefined) return;
  if (m.method === 'initialize') send({ jsonrpc: '2.0', id: m.id, result: { protocolVersion: m.params.protocolVersion, capabilities: { tools: {} }, serverInfo: { name: 'echo', version: '1' } } });
  else if (m.method === 'tools/list') send({ jsonrpc: '2.0', id: m.id, result: { tools: [{ name: 'echo', description: 'Echoes text', inputSchema: { type: 'object', properties: { text: { type: 'string' } } } }] } });
  else if (m.method === 'tools/call') send({ jsonrpc: '2.0', id: m.id, result: { content: [{ type: 'text', text: 'echo: ' + m.params.arguments.text }] } });
  else send({ jsonrpc: '2.0', id: m.id, error: { code: -32601, message: 'no' } });
});
"#;
}