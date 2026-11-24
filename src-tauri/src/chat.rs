//! Chat turns: prompt assembly, streaming to the UI, tool loop, context accounting and compression.

use anyhow::{anyhow, Result};
use serde_json::{json, Value};
use std::collections::HashSet;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;
use tauri::{AppHandle, Emitter, Manager};

use crate::db::{Chat, Message, Project};
use crate::llm::{self, Delta, StreamResult};
use crate::prompts;
use crate::state::{AppState, LiveReply, DEFAULT_TITLES};
use crate::tools::{self, ToolCtx};

pub fn emit(app: &AppHandle, chat_id: &str, kind: &str, mut payload: Value) {
    payload["chat_id"] = json!(chat_id);
    payload["kind"] = json!(kind);
    let _ = app.emit("chat-event", payload);
}

/// Today's date (UTC) as YYYY-MM-DD.
pub fn today() -> String {
    ymd(crate::state::now_ms() / 1000)
}

/// Unix seconds → YYYY-MM-DD (UTC), using the civil-from-days algorithm.
pub fn ymd(secs: i64) -> String {
    let z = secs.div_euclid(86400) + 719468;
    let era = z.div_euclid(146097);
    let doe = z - era * 146097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    format!("{:04}-{:02}-{:02}", if m <= 2 { y + 1 } else { y }, m, d)
}

/// Context window for a chat as far as it is known: its own cap (if set) bounded by the model's window –
/// the loaded model's, or while none is loaded the last one's (before, an unloaded model showed
/// "40k / 2k"). 0 when nothing is known yet (no cap, no model ever loaded).
pub fn known_budget(state: &AppState, chat: &Chat) -> u32 {
    let loaded = state.info.read().n_ctx;
    let model = if loaded > 0 { loaded } else { state.settings.read().last_n_ctx };
    match (chat.ctx_limit, model) {
        (0, m) => m,
        (cap, 0) => cap,
        (cap, m) => cap.min(m),
    }
}

/// Effective context window for a chat (prompt fitting, compression): `known_budget`, at least 2048.
pub fn budget(state: &AppState, chat: &Chat) -> u32 {
    known_budget(state, chat).max(2048)
}

/// Re-estimates what the chat occupies after its history changed outside a turn (a deleted
/// exchange), stores it for the context meter and returns it.
pub fn refresh_ctx_estimate(state: &AppState, chat_id: &str) -> Result<u32> {
    let chat = state.db.chat(chat_id)?.ok_or_else(|| anyhow!("chat not found"))?;
    let project = state.db.chat_project(&chat)?;
    let msgs = build_messages(state, &chat, project.as_ref(), false)?;
    let used = llm::estimate_tokens(&serde_json::to_string(&msgs)?);
    state.db.set_chat_ctx(chat_id, used)?;
    Ok(used)
}

/// The saved assistant a chat talks to (None: the default one, or it was deleted since).
pub fn chat_assistant(state: &AppState, chat: &Chat) -> Option<crate::state::Assistant> {
    let id = chat.assistant_id.as_deref()?;
    state.settings.read().assistants.iter().find(|a| a.id == id).cloned()
}

pub fn system_prompt(state: &AppState, chat: &Chat, project: Option<&Project>, web: bool) -> String {
    let (lang, custom) = {
        let s = state.settings.read();
        (s.reply_lang.clone(), s.custom_instructions.clone())
    };
    let mut sys = prompts::base(&lang, &today());
    sys.push_str(&prompts::custom_instructions(&custom));
    if let Some(a) = chat_assistant(state, chat) {
        sys.push_str(&prompts::assistant(&a.name, &a.instructions));
    }
    sys.push_str(prompts::GENERAL_TOOLS);
    sys.push_str(prompts::ATTACHMENTS);
    sys.push_str(if web { prompts::WEB_TOOLS } else { prompts::LINKS_ONLY });
    if let Some(p) = project {
        let tree = tools::tree(Path::new(&p.folder), "", 3, 220);
        sys.push_str(&prompts::project(&p.name, &p.folder, &tree, &p.instructions));
    }
    if !chat.summary.trim().is_empty() {
        sys.push_str("\n\n# Summary of the earlier part of this conversation (it was compressed to save context)\n");
        sys.push_str(chat.summary.trim());
    }
    sys
}

/// OpenAI-style message list for the next request: system prompt (+ summary of compressed turns),
/// then every non-compressed message. Tool calls without a stored result (cancelled turns) are
/// dropped so the chat template never sees a dangling call.
pub fn build_messages(state: &AppState, chat: &Chat, project: Option<&Project>, web: bool) -> Result<Vec<Value>> {
    let all = state.db.messages(&chat.id)?;
    let live: Vec<&Message> = all.iter().filter(|m| !m.compressed && m.role != "event").collect();
    let answered: HashSet<String> = live
        .iter()
        .filter(|m| m.role == "tool")
        .filter_map(|m| m.meta["tool_call_id"].as_str().map(String::from))
        .collect();
    let vision = state.info.read().vision;
    let mut out = vec![json!({"role": "system", "content": system_prompt(state, chat, project, web)})];
    for m in live {
        match m.role.as_str() {
            "user" => {
                let c = m.meta["prompt_content"].as_str().unwrap_or(&m.content);
                // attached images go along as data URLs (the same bytes every time: cache-friendly)
                // (a model loaded without its projector would refuse the request: it is told instead)
                let attached_images = m.meta["images"].as_array().map(Vec::len).unwrap_or(0);
                let images: Vec<Value> = if vision {
                    m.meta["images"]
                        .as_array()
                        .into_iter()
                        .flatten()
                        .filter_map(|i| image_data_url(state, i["file"].as_str()?))
                        .map(|url| json!({"type": "image_url", "image_url": {"url": url}}))
                        .collect()
                } else {
                    vec![]
                };
                if images.is_empty() {
                    let c = if attached_images > 0 && !vision {
                        format!("{c}\n\n[{attached_images} image(s) were attached here, but the model loaded now can't see images]")
                    } else {
                        c.to_string()
                    };
                    out.push(json!({"role": "user", "content": c}));
                } else {
                    let mut parts = vec![json!({"type": "text", "text": c})];
                    parts.extend(images);
                    out.push(json!({"role": "user", "content": parts}));
                }
            }
            "assistant" => {
                let calls: Vec<Value> = m.meta["tool_calls"]
                    .as_array()
                    .into_iter()
                    .flatten()
                    .filter(|t| t["id"].as_str().is_some_and(|id| answered.contains(id)))
                    .map(|t| json!({"id": t["id"], "type": "function", "function": {"name": t["name"], "arguments": t["arguments"]}}))
                    .collect();
                if m.content.is_empty() && calls.is_empty() {
                    continue;
                }
                let mut v = json!({"role": "assistant", "content": m.content});
                if !calls.is_empty() {
                    v["tool_calls"] = json!(calls);
                }
                out.push(v);
            }
            "tool" => out.push(json!({"role": "tool", "tool_call_id": m.meta["tool_call_id"], "content": m.content})),
            _ => {}
        }
    }
    Ok(out)
}

pub struct SendOpts {
    pub chat_id: String,
    pub text: String,
    pub think: bool,
    pub web: bool,
    pub research: bool,
    /// files and images dropped, picked or pasted into the composer
    pub attachments: Vec<Attachment>,
    /// regenerate / edit: the earlier message's attachments (`images`, `attached`, `files`) carried over
    pub keep: Option<Value>,
}

/// A file from the composer: a path on disk (dropped or picked) or a pasted image (`data:` URL).
#[derive(Clone, Debug, serde::Deserialize)]
pub struct Attachment {
    pub name: String,
    #[serde(default)]
    pub path: Option<String>,
    #[serde(default)]
    pub data: Option<String>,
}

const IMAGE_EXT: &[&str] = &["png", "jpg", "jpeg", "webp", "gif", "bmp"];
/// Biggest image accepted (bytes) and the text of all attached files together (characters).
const MAX_IMAGE: u64 = 20 << 20;
const MAX_ATTACHED_CHARS: usize = 240_000;

fn mime(ext: &str) -> &'static str {
    match ext {
        "jpg" | "jpeg" => "image/jpeg",
        "webp" => "image/webp",
        "gif" => "image/gif",
        "bmp" => "image/bmp",
        _ => "image/png",
    }
}

/// An attached image (a file in data/attachments) as a `data:` URL for the model or the window.
pub fn image_data_url(state: &AppState, file: &str) -> Option<String> {
    use base64::Engine;
    // (only plain names inside the attachments folder)
    if file.contains(['/', '\\']) || file.starts_with('.') {
        return None;
    }
    let bytes = std::fs::read(state.paths.attachments.join(file)).ok()?;
    let ext = file.rsplit('.').next().unwrap_or("png").to_lowercase();
    Some(format!("data:{};base64,{}", mime(&ext), base64::engine::general_purpose::STANDARD.encode(bytes)))
}

/// Turns the composer's attachments into message meta: images are copied to data/attachments
/// (`meta.images`), the text of other files is read (`meta.attached`, sent with the message, and
/// `meta.files` for the chips the window shows). Fails with a clear message for an image when the
/// loaded model can't see, and for files that have no readable text.
fn take_attachments(state: &AppState, list: &[Attachment], meta: &mut Value) -> Result<()> {
    use base64::Engine;
    if list.is_empty() {
        return Ok(());
    }
    let mut images: Vec<Value> = meta["images"].as_array().cloned().unwrap_or_default();
    let mut files: Vec<Value> = meta["files"].as_array().cloned().unwrap_or_default();
    let mut attached = meta["attached"].as_str().unwrap_or("").to_string();
    for a in list {
        let path = a.path.as_deref().map(Path::new);
        let ext = path.and_then(|p| p.extension()).and_then(|e| e.to_str()).unwrap_or("").to_lowercase();
        let is_image = a.data.as_deref().is_some_and(|d| d.starts_with("data:image/")) || IMAGE_EXT.contains(&ext.as_str());
        if is_image {
            if !state.info.read().vision {
                return Err(anyhow!(
                    "The loaded model can't see images. Open Models and press \"Add image support\" on it (a ~1 GB download), then reload it."
                ));
            }
            let (bytes, ext) = match (&a.data, path) {
                (Some(d), _) => {
                    let (head, b64) = d.split_once(',').ok_or_else(|| anyhow!("bad pasted image"))?;
                    let ext = head.trim_start_matches("data:image/").split(';').next().unwrap_or("png").replace("jpeg", "jpg");
                    (base64::engine::general_purpose::STANDARD.decode(b64.trim())?, ext)
                }
                (None, Some(p)) => {
                    let size = std::fs::metadata(p)?.len();
                    if size > MAX_IMAGE {
                        return Err(anyhow!("{} is too big ({} MB; at most {} MB)", a.name, size >> 20, MAX_IMAGE >> 20));
                    }
                    (std::fs::read(p)?, ext.clone())
                }
                _ => return Err(anyhow!("{}: nothing to read", a.name)),
            };
            if bytes.len() as u64 > MAX_IMAGE {
                return Err(anyhow!("{} is too big (at most {} MB)", a.name, MAX_IMAGE >> 20));
            }
            let ext = if IMAGE_EXT.contains(&ext.as_str()) { ext } else { "png".into() };
            let file = format!("{}.{ext}", crate::state::new_id());
            std::fs::write(state.paths.attachments.join(&file), &bytes)?;
            images.push(json!({"name": a.name, "file": file}));
            continue;
        }
        let p = path.ok_or_else(|| anyhow!("{}: nothing to read", a.name))?;
        if p.is_dir() {
            return Err(anyhow!("{} is a folder: drop it on the window to work in it as a project", a.name));
        }
        let text = crate::extract::extract(p).map_err(|e| anyhow!("{} can't be read as text ({e})", a.name))?;
        let room = MAX_ATTACHED_CHARS.saturating_sub(attached.chars().count());
        let clipped = text.chars().count() > room;
        let body: String = text.chars().take(room).collect();
        attached.push_str(&format!(
            "\n\n<attached_file name=\"{}\">\n{}{}\n</attached_file>",
            a.name.replace('"', "'"),
            body,
            if clipped { "\n[… the rest of the file was cut: too long to attach whole]" } else { "" }
        ));
        files.push(json!({"name": a.name, "chars": text.chars().count(), "clipped": clipped}));
    }
    if !images.is_empty() {
        meta["images"] = json!(images);
    }
    if !files.is_empty() {
        meta["files"] = json!(files);
        meta["attached"] = json!(attached);
    }
    Ok(())
}

/// Marks the chat as busy. One reply (or compression) per chat at a time: a second one would replace
/// the first one's stop flag (Stop no longer reaching it) and clear the busy mark when it ends.
fn register_cancel(state: &AppState, chat_id: &str) -> Result<Arc<AtomicBool>> {
    let mut g = state.generations.lock();
    if g.contains_key(chat_id) {
        return Err(anyhow!("This chat is still answering. Wait for the reply or stop it first."));
    }
    let c = Arc::new(AtomicBool::new(false));
    g.insert(chat_id.to_string(), c.clone());
    Ok(c)
}

/// Shows an error that happened before a reply started (no model, chat busy…) in the chat.
fn report<T>(app: &AppHandle, chat_id: &str, r: Result<T>) -> Result<T> {
    if let Err(e) = &r {
        emit(app, chat_id, "error", json!({"error": e.to_string()}));
    }
    r
}

pub fn stop(state: &AppState, chat_id: &str) {
    if let Some(c) = state.generations.lock().get(chat_id) {
        c.store(true, Ordering::Relaxed);
    }
}

/// Stops every running reply. Called before the user unloads or switches the model (and at quit):
/// the reply then ends as stopped instead of looking like a crash, which used to restart the model
/// the user had just unloaded.
pub fn stop_all(state: &AppState) {
    for c in state.generations.lock().values() {
        c.store(true, Ordering::Relaxed);
    }
}

/// Full user turn: auto-retrieval (projects), pre-emptive compression if the message would not fit,
/// store the user message, run the assistant (normal tool loop or deep research), then title +
/// post-turn compression. Progress reaches the UI only through `chat-event`s.
pub async fn send(app: AppHandle, opts: SendOpts) -> Result<()> {
    let state = app.state::<Arc<AppState>>().inner().clone();
    let start = (|| {
        if state.llm_url().is_none() {
            return Err(anyhow!("No model is loaded yet. Open Models and load one."));
        }
        let chat = state.db.chat(&opts.chat_id)?.ok_or_else(|| anyhow!("chat not found"))?;
        let project = state.db.chat_project(&chat)?;
        let cancel = register_cancel(&state, &chat.id)?;
        Ok((chat, project, cancel))
    })();
    let (chat, project, cancel) = report(&app, &opts.chat_id, start)?;
    emit(&app, &chat.id, "busy", json!({"busy": true}));

    let result = async {
        // attachments: the ones carried over from the message being regenerated / edited, plus new ones
        let mut meta = json!({});
        if let Some(k) = &opts.keep {
            for key in ["images", "files", "attached"] {
                if !k[key].is_null() {
                    meta[key] = k[key].clone();
                }
            }
        }
        take_attachments(&state, &opts.attachments, &mut meta)?;
        let text = format!("{}{}", opts.text, meta["attached"].as_str().unwrap_or(""));
        if meta["attached"].is_string() {
            meta["prompt_content"] = json!(text);
        }
        // auto-retrieval for project chats
        if let Some(p) = &project {
            if let Ok(hits) = crate::rag::search(&app, &p.id, &opts.text, 6).await {
                if !hits.is_empty() {
                    let mut ctx = String::new();
                    for h in &hits {
                        let t: String = h.text.chars().take(1800).collect();
                        ctx.push_str(&format!("\n--- {}:{}-{} ---\n{}\n", h.path, h.line_start, h.line_end, t));
                    }
                    meta["prompt_content"] = json!(format!("{}\n\n<retrieved_context note=\"auto-retrieved excerpts from the project, may be partial\">{}</retrieved_context>", text, ctx));
                    meta["retrieved"] = json!(hits.iter().map(|h| json!({"path": h.path, "lines": format!("{}-{}", h.line_start, h.line_end)})).collect::<Vec<_>>());
                }
            }
        }
        if opts.research {
            meta["research"] = json!(true);
        }
        // compress first if the new message would not fit comfortably
        let fresh = state.db.chat(&chat.id)?.unwrap_or(chat.clone());
        let incoming = llm::estimate_tokens(meta["prompt_content"].as_str().unwrap_or(&opts.text));
        // 0.8, not 0.9: the estimate can be ~10% low, and the reply needs room too
        if fresh.ctx_used + incoming > (budget(&state, &fresh) as f32 * 0.8) as u32 {
            let _ = compress(&app, &chat.id, &cancel).await;
        }
        let user = state.db.add_message(&chat.id, "user", &opts.text, "", &meta)?;
        emit(&app, &chat.id, "message", json!({"message": user}));
        // a name right away (long project turns used to leave "New chat" until the very end, or for
        // good when the turn was stopped); the model's title replaces it when the reply is done
        if DEFAULT_TITLES.contains(&fresh.title.as_str()) {
            if let Some(t) = quick_title(&opts.text) {
                state.db.rename_chat(&chat.id, &t)?;
                state.provisional_titles.lock().insert(chat.id.clone());
                emit(&app, &chat.id, "title", json!({"title": t}));
            }
        }

        if opts.research {
            crate::research::run(&app, &chat.id, &opts.text, opts.think, &cancel).await?;
        } else {
            run_turn(&app, &chat.id, project.as_ref(), opts.think, opts.web, &cancel).await?;
        }
        Ok::<(), anyhow::Error>(())
    }
    .await;

    if let Err(e) = &result {
        emit(&app, &chat.id, "error", json!({"error": e.to_string()}));
    }
    // follow-ups: title + compression (only if not cancelled)
    if !cancel.load(Ordering::Relaxed) {
        if let Ok(Some(c)) = state.db.chat(&chat.id) {
            let provisional = state.provisional_titles.lock().remove(&chat.id);
            if provisional || DEFAULT_TITLES.contains(&c.title.as_str()) {
                let _ = make_title(&app, &c).await;
            }
            let c = state.db.chat(&chat.id)?.unwrap_or(c);
            let compress_at = state.settings.read().compress_at;
            if c.ctx_used as f32 > budget(&state, &c) as f32 * compress_at {
                let _ = compress(&app, &chat.id, &cancel).await;
            }
        }
    }
    state.generations.lock().remove(&chat.id);
    state.live.lock().remove(&chat.id);
    state.provisional_titles.lock().remove(&chat.id);
    emit(&app, &chat.id, "busy", json!({"busy": false}));
    if !cancel.load(Ordering::Relaxed) {
        crate::reply_finished(&app);
    }
    result
}

/// A name for a chat from its first message, set the moment it is sent (the model's title replaces it
/// when the reply is done): the first line without links, Markdown and greetings, at most ~48 chars.
pub fn quick_title(text: &str) -> Option<String> {
    static LINK: std::sync::OnceLock<regex::Regex> = std::sync::OnceLock::new();
    let link = LINK.get_or_init(|| regex::Regex::new(r"(?i)https?://\S+").expect("link regex"));
    let line = text.lines().map(str::trim).find(|l| !l.is_empty())?;
    let line = link.replace_all(line, "");
    let mut s: String = line.chars().filter(|c| !matches!(c, '#' | '*' | '`' | '>' | '_')).collect::<String>().split_whitespace().collect::<Vec<_>>().join(" ");
    // greetings and politeness carry no topic
    const FILLER: &[&str] = &[
        "hey", "hi", "hello", "please", "pls", "so", "ok", "okay", "can you", "could you", "would you", "i want you to",
        "i need you to", "salut", "bună", "buna", "te rog", "poți să", "poti sa", "vreau să", "vreau sa",
    ];
    loop {
        let lower = s.to_lowercase();
        let Some(f) = FILLER.iter().find(|f| lower.starts_with(*f) && lower[f.len()..].starts_with([' ', ',', '!', '.', ':'])) else { break };
        // (checked slice: lowercasing can change a character's byte length)
        let Some(rest) = s.get(f.len()..) else { break };
        s = rest.trim_start_matches([' ', ',', '!', '.', ':']).to_string();
    }
    let s = s.trim().trim_end_matches(['?', '.', '!', ':', ',']).trim().to_string();
    if s.chars().count() < 3 {
        return None;
    }
    let mut t = if s.chars().count() <= 48 {
        s
    } else {
        let cut: String = s.chars().take(48).collect();
        let at = cut.rfind(' ').filter(|&i| i > 20).unwrap_or(cut.len());
        format!("{}…", cut[..at].trim_end_matches([',', ';', ':', ' ']))
    };
    if let Some(first) = t.chars().next() {
        t = first.to_uppercase().collect::<String>() + &t[first.len_utf8()..];
    }
    Some(t)
}

pub async fn regenerate(app: AppHandle, chat_id: String, think: bool, web: bool) -> Result<()> {
    let state = app.state::<Arc<AppState>>().inner().clone();
    let prepared = (|| {
        // (checked before the answer is deleted: send would refuse a busy chat only afterwards)
        if state.generations.lock().contains_key(&chat_id) {
            return Err(anyhow!("This chat is still answering. Wait for the reply or stop it first."));
        }
        let msgs = state.db.messages(&chat_id)?;
        let last_user = msgs.iter().rev().find(|m| m.role == "user").ok_or_else(|| anyhow!("nothing to regenerate"))?;
        let research = last_user.meta["research"].as_bool().unwrap_or(false);
        let text = last_user.content.clone();
        let keep = last_user.meta.clone();
        state.db.truncate_from(&chat_id, last_user.seq)?;
        Ok((text, research, keep))
    })();
    let (text, research, keep) = report(&app, &chat_id, prepared)?;
    send(app, SendOpts { chat_id, text, think, web, research, attachments: vec![], keep: Some(keep) }).await
}

/// Streams one assistant message: emits deltas to the UI and persists it.
pub async fn stream_assistant(
    app: &AppHandle,
    chat_id: &str,
    body: Value,
    cancel: &AtomicBool,
    extra_meta: Value,
) -> Result<(Message, StreamResult)> {
    let state = app.state::<Arc<AppState>>().inner().clone();
    let temp_id = crate::state::new_id();
    state.live.lock().insert(chat_id.to_string(), LiveReply { temp_id: temp_id.clone(), ..Default::default() });
    emit(app, chat_id, "assistant_start", json!({"temp_id": temp_id}));
    let started = Instant::now();
    let mut pend_c = String::new();
    let mut pend_r = String::new();
    let mut last_flush = Instant::now();
    // the UI gets deltas; the snapshot (`state.live`) gets the same text, for a chat opened mid-reply
    let flush = |c: &str, r: &str| {
        if let Some(l) = state.live.lock().get_mut(chat_id) {
            l.content.push_str(c);
            l.reasoning.push_str(r);
        }
        emit(app, chat_id, "delta", json!({"temp_id": temp_id, "content": c, "reasoning": r}));
    };
    let r = llm::stream_chat(&state, body, cancel, |d| {
        match d {
            Delta::Content(s) => pend_c.push_str(s),
            Delta::Reasoning(s) => pend_r.push_str(s),
            Delta::Tool { index, name } => emit(app, chat_id, "tool_pending", json!({"temp_id": temp_id, "index": index, "name": name})),
            Delta::Progress { tps, tokens } => {
                if let Some(l) = state.live.lock().get_mut(chat_id) {
                    l.tps = tps;
                    l.tokens = tokens;
                }
                emit(app, chat_id, "progress", json!({"temp_id": temp_id, "tps": tps, "tokens": tokens}))
            }
        }
        if last_flush.elapsed().as_millis() >= 33 && (!pend_c.is_empty() || !pend_r.is_empty()) {
            flush(&pend_c, &pend_r);
            pend_c.clear();
            pend_r.clear();
            last_flush = Instant::now();
        }
    })
    .await;
    if !pend_c.is_empty() || !pend_r.is_empty() {
        flush(&pend_c, &pend_r);
    }
    state.live.lock().remove(chat_id);
    let r = match r {
        Ok(r) => r,
        Err(e) => {
            emit(app, chat_id, "assistant_failed", json!({"temp_id": temp_id}));
            return Err(e);
        }
    };
    let model = state.info.read().model.clone().unwrap_or_default();
    let mut meta = json!({
        "tool_calls": r.tool_calls.iter().map(|t| json!({"id": t.id, "name": t.name, "arguments": t.arguments})).collect::<Vec<_>>(),
        "tps": r.tps,
        "tokens": r.completion_tokens,
        "prompt_tokens": r.prompt_tokens,
        "cached_tokens": r.cached_tokens,
        "prompt_tps": r.prompt_tps,
        "prompt_evaluated": r.prompt_evaluated,
        "ctx": r.prompt_tokens + r.completion_tokens,
        "model": model,
        "elapsed_ms": started.elapsed().as_millis() as u64,
        "cancelled": r.cancelled,
        "finish": r.finish_reason,
    });
    if let (Some(m), Some(extra)) = (meta.as_object_mut(), extra_meta.as_object()) {
        for (k, v) in extra {
            m.insert(k.clone(), v.clone());
        }
    }
    let msg = state.db.add_message(chat_id, "assistant", &r.content, &r.reasoning, &meta)?;
    if r.prompt_tokens > 0 {
        state.db.set_chat_ctx(chat_id, r.prompt_tokens + r.completion_tokens)?;
    }
    if r.completion_tokens > 8 && !model.is_empty() {
        let _ = state.db.add_stat(&model, r.tps, r.completion_tokens);
    }
    let chat = state.db.chat(chat_id)?;
    let b = chat.as_ref().map(|c| budget(&state, c)).unwrap_or(0);
    emit(app, chat_id, "assistant_done", json!({"temp_id": temp_id, "message": msg, "ctx_used": r.prompt_tokens + r.completion_tokens, "ctx_budget": b}));
    Ok((msg, r))
}

/// Sampling settings (Qwen-recommended top_k/min_p) and the per-request thinking switch. A chat's
/// assistant may set its own temperature.
pub fn gen_params_for(state: &AppState, chat: &Chat, think: bool) -> Value {
    let mut p = gen_params(state, think);
    if let Some(t) = chat_assistant(state, chat).and_then(|a| a.temperature) {
        p["temperature"] = json!(if think { t.clamp(0.5, 0.7) } else { t.clamp(0.0, 2.0) });
    }
    p
}

pub fn gen_params(state: &AppState, think: bool) -> Value {
    let s = state.settings.read();
    json!({
        "temperature": if think { s.temperature.clamp(0.5, 0.7) } else { s.temperature },
        "top_p": s.top_p,
        "top_k": 20,
        "min_p": 0.0,
        "chat_template_kwargs": {"enable_thinking": think},
    })
}

/// Conservative token estimate for context fitting (code tokenises denser than prose).
fn est_tokens(chars: usize) -> u32 {
    (chars as f32 / 3.0).ceil() as u32
}

/// Estimated prompt size of the next request (messages + tool definitions).
fn prompt_estimate(state: &AppState, chat: &Chat, project: Option<&Project>, web: bool, defs: &[Value]) -> Result<u32> {
    let msgs = build_messages(state, chat, project, web)?;
    Ok(est_tokens(llm::without_images(&serde_json::to_string(&msgs)?).len() + serde_json::to_string(defs)?.len()))
}

/// Keeps long tool sessions inside the window. Compression only summarises *earlier* turns, so a
/// single long turn (an audit reading dozens of files) would otherwise grow until llama-server
/// rejects it. When the prompt passes `high` × budget, the oldest tool results (all but the last
/// `keep`) are replaced by a one-line stub until it is under `low` × budget. The original stays in
/// `meta.full` for the UI; the stub is stored, so later prompts stay byte-stable for the cache.
fn fit_context(state: &AppState, chat_id: &str, project: Option<&Project>, web: bool, defs: &[Value], (high, low, keep): (f32, f32, usize)) -> Result<u32> {
    let chat = state.db.chat(chat_id)?.ok_or_else(|| anyhow!("chat deleted"))?;
    let b = budget(state, &chat) as f32;
    let mut est = prompt_estimate(state, &chat, project, web, defs)?;
    if (est as f32) <= b * high {
        return Ok(est);
    }
    let msgs = state.db.messages(chat_id)?;
    let tools_live: Vec<&Message> = msgs.iter().filter(|m| !m.compressed && m.role == "tool").collect();
    // oldest first, sparing the last `keep` results; if that is not enough (a few huge reads in a
    // small window), continue into them too, always sparing the newest one
    let n_tools = tools_live.len();
    let order = (0..n_tools.saturating_sub(keep)).chain(n_tools.saturating_sub(keep)..n_tools.saturating_sub(1));
    let mut pruned = 0;
    for m in order.map(|i| tools_live[i]) {
        if (est as f32) <= b * low {
            break;
        }
        if m.meta["pruned"].as_bool().unwrap_or(false) || m.content.len() < 400 {
            continue;
        }
        let name = m.meta["name"].as_str().unwrap_or("tool");
        let args: Value = serde_json::from_str(m.meta["arguments"].as_str().unwrap_or("{}")).unwrap_or_default();
        let what = ["path", "pattern", "name", "query", "command", "url"].iter().find_map(|k| args[*k].as_str()).unwrap_or("");
        let stub = format!("[{name}({what}) output removed to save context ({} chars). You already used it above; call the tool again if you need it.]", m.content.len());
        let mut meta = m.meta.clone();
        meta["pruned"] = json!(true);
        meta["full"] = json!(m.content);
        state.db.update_message(&m.id, &stub, &m.reasoning, &meta)?;
        est = est.saturating_sub(est_tokens(m.content.len())) + est_tokens(stub.len());
        pruned += 1;
    }
    if pruned > 0 {
        state.log(format!("context: pruned {pruned} old tool results, ~{est} tokens of {b}"));
        let chat = state.db.chat(chat_id)?.ok_or_else(|| anyhow!("chat deleted"))?;
        est = prompt_estimate(state, &chat, project, web, defs)?;
    }
    Ok(est)
}

/// Cuts a tool result down to `max_tokens`, keeping the start and the end, and tells the model how
/// to get the rest instead of overflowing the window.
fn clip_tool_output(content: &str, max_tokens: u32) -> String {
    let max_chars = (max_tokens as usize) * 3;
    if content.len() <= max_chars {
        return content.to_string();
    }
    let head = content.floor_char_boundary(max_chars * 4 / 5);
    let tail = content.ceil_char_boundary(content.len() - max_chars / 5);
    format!(
        "{}\n\n[… {} characters cut to fit the context window. Ask for less at once: a line range with read_file, a narrower path/glob, or audit_code for large reviews …]\n\n{}",
        &content[..head],
        tail - head,
        &content[tail..]
    )
}

fn is_context_overflow(e: &anyhow::Error) -> bool {
    let s = e.to_string().to_lowercase();
    s.contains("context") && (s.contains("exceed") || s.contains("too long") || s.contains("size") || s.contains("n_ctx"))
}

/// Model rounds one user message may take (each round: a reply that may call tools).
const MAX_ROUNDS: usize = 40;
/// Rounds before the limit at which the model is told to wrap up.
const WRAP_UP_AT: usize = 5;
const _: () = assert!(WRAP_UP_AT < MAX_ROUNDS);

/// Told to the model with the last tool results when few rounds are left.
fn rounds_left_note(left: usize) -> String {
    format!("\n\n[Lumen: {left} tool rounds left for this request. Stop exploring soon and give the user your answer.]")
}

/// The one-off instruction for the final round once all rounds are used (not stored in the chat).
const ANSWER_NOW: &str = "[Lumen: you have used all the tool rounds for this request, so tools are switched off now. \
Answer the user's request now from what you have found so far. Say briefly what you could not check yet, \
so the user can ask you to continue.]";

/// Assistant loop: stream a reply; if it requested tools, execute them, store the results and ask
/// again (at most `MAX_ROUNDS`). Mid-loop it keeps long tool sessions inside the window. A turn always
/// ends with an answer: when the rounds run out, a last round without tools makes the model reply
/// (before, the loop just stopped after the last tool result - the user saw reasoning and no answer).
async fn run_turn(app: &AppHandle, chat_id: &str, project: Option<&Project>, think: bool, web: bool, cancel: &AtomicBool) -> Result<()> {
    let state = app.state::<Arc<AppState>>().inner().clone();
    let mut defs = tools::definitions(project.is_some(), web);
    defs.extend(crate::mcp::definitions(app).await);
    let ctx = ToolCtx { app, project, chat_id, cancel, web };
    let mut answered = false;
    for round in 0..MAX_ROUNDS {
        let _ = fit_context(&state, chat_id, project, web, &defs, (0.72, 0.5, 4))?;
        let chat = state.db.chat(chat_id)?.ok_or_else(|| anyhow!("chat deleted"))?;
        let messages = build_messages(&state, &chat, project, web)?;
        let mut body = gen_params_for(&state, &chat, think);
        body["messages"] = json!(messages);
        if !defs.is_empty() {
            body["tools"] = json!(defs);
        }
        let (_msg, r) = match stream_assistant(app, chat_id, body.clone(), cancel, json!({})).await {
            Ok(x) => x,
            // stopped (Stop, unload, model switch, quit): the server may be gone already, so the request
            // fails before it sees the stop - that's the stop, not a crash to recover from
            Err(e) if cancel.load(Ordering::Relaxed) => {
                state.log(format!("[lumen] reply stopped ({e})"));
                break;
            }
            Err(e) if is_context_overflow(&e) => {
                // the estimate was off: drop everything but the latest tool result, summarise the earlier
                // turns too (long conversations overflow without any tool output), and try once more
                state.log(format!("context overflow, shrinking the prompt and retrying: {e}"));
                fit_context(&state, chat_id, project, web, &defs, (0.0, 0.3, 1))?;
                compress(app, chat_id, cancel).await?;
                let chat = state.db.chat(chat_id)?.ok_or_else(|| anyhow!("chat deleted"))?;
                body["messages"] = json!(build_messages(&state, &chat, project, web)?);
                stream_assistant(app, chat_id, body, cancel, json!({})).await.map_err(|e| {
                    if is_context_overflow(&e) {
                        anyhow!("This turn no longer fits in the model's context window. Start a new chat, raise the context size in Settings, or ask for a smaller part (a folder) to be reviewed with audit_code.")
                    } else {
                        e
                    }
                })?
            }
            Err(e) if llm::is_connection_error(&e) && !cancel.load(Ordering::Relaxed) => {
                // llama-server died or dropped the connection (e.g. after a long audit): bring it back and retry
                state.log(format!("lost the connection to llama-server: {e}"));
                emit(app, chat_id, "notice", json!({"key": "serverRestarting"}));
                crate::server::recover(app).await.map_err(|re| anyhow!("The model server stopped and could not be restarted: {re}"))?;
                stream_assistant(app, chat_id, body, cancel, json!({}))
                    .await
                    .map_err(|e2| if llm::is_connection_error(&e2) { anyhow!("The model server stopped again while answering. Its last output is in LumenData\\logs\\lumen.log (Settings → Server logs). Try a smaller request, a smaller context size, or another model.") } else { e2 })?
            }
            Err(e) => return Err(e),
        };
        if r.cancelled || r.tool_calls.is_empty() {
            // (a reply with reasoning but no text and no tool calls is no answer either)
            answered = r.cancelled || !r.content.trim().is_empty();
            break;
        }
        let left = MAX_ROUNDS - round - 1;
        // room left for this round's results, shared between the calls
        let b = budget(&state, &state.db.chat(chat_id)?.ok_or_else(|| anyhow!("chat deleted"))?);
        let mut used = (r.prompt_tokens + r.completion_tokens).max(1);
        let n_calls = r.tool_calls.len() as u32;
        for (i, tc) in r.tool_calls.iter().enumerate() {
            if cancel.load(Ordering::Relaxed) {
                break;
            }
            let running = json!({"id": tc.id, "name": tc.name, "arguments": tc.arguments});
            state.live.lock().insert(chat_id.to_string(), LiveReply { tool: Some(running.clone()), ..Default::default() });
            emit(app, chat_id, "tool_start", running);
            let out = tools::run(&ctx, &tc.id, &tc.name, &tc.arguments).await;
            let free = (b as f32 * 0.8) as u32;
            let share = free.saturating_sub(used) / (n_calls - i as u32).max(1);
            let mut content = clip_tool_output(&out.content, share.clamp(1500, 16_000));
            if left == WRAP_UP_AT && i + 1 == r.tool_calls.len() {
                content.push_str(&rounds_left_note(left));
            }
            used += est_tokens(content.len());
            let mut meta = json!({"tool_call_id": tc.id, "name": tc.name, "arguments": tc.arguments, "display": out.display, "ok": out.ok});
            if content.len() < out.content.len() {
                meta["full"] = json!(out.content);
                meta["clipped"] = json!(true);
            }
            let m = state.db.add_message(chat_id, "tool", &content, "", &meta)?;
            state.live.lock().remove(chat_id);
            emit(app, chat_id, "message", json!({"message": m}));
        }
        // compress mid-loop if long tool sessions approach the limit
        let c = state.db.chat(chat_id)?.ok_or_else(|| anyhow!("chat deleted"))?;
        if c.ctx_used as f32 > budget(&state, &c) as f32 * 0.88 {
            compress(app, chat_id, cancel).await?;
        }
    }
    if !answered && !cancel.load(Ordering::Relaxed) {
        state.log(format!("[lumen] the turn ended without an answer (all {MAX_ROUNDS} rounds used, or no text): asking for it without tools"));
        emit(app, chat_id, "notice", json!({"key": "answerNow"}));
        let _ = fit_context(&state, chat_id, project, web, &defs, (0.72, 0.5, 4))?;
        let chat = state.db.chat(chat_id)?.ok_or_else(|| anyhow!("chat deleted"))?;
        let mut body = gen_params_for(&state, &chat, think);
        body["messages"] = json!(final_round_messages(build_messages(&state, &chat, project, web)?));
        // the same tool list keeps the prompt prefix (and llama-server's cache) of the earlier rounds;
        // tool_choice "none" plus the note ask for text. (llama-server still parses a tool call the model
        // writes anyway, so the result is checked below.)
        if !defs.is_empty() {
            body["tools"] = json!(defs);
            body["tool_choice"] = json!("none");
        }
        let (_msg, r) = stream_assistant(app, chat_id, body, cancel, json!({"forced_answer": true})).await?;
        if r.content.trim().is_empty() && !r.cancelled {
            return Err(anyhow!(
                "The model kept using tools instead of answering ({MAX_ROUNDS} rounds). Ask it to answer with what it has found so far, or to continue with a narrower part."
            ));
        }
    }
    Ok(())
}

/// The request for the last round: the conversation plus the one-off "answer now" note (not stored,
/// so later prompts stay byte-stable).
fn final_round_messages(mut messages: Vec<Value>) -> Vec<Value> {
    messages.push(json!({"role": "user", "content": ANSWER_NOW}));
    messages
}

async fn make_title(app: &AppHandle, chat: &Chat) -> Result<()> {
    let state = app.state::<Arc<AppState>>().inner().clone();
    let msgs = state.db.messages(&chat.id)?;
    let first_user = msgs.iter().find(|m| m.role == "user").map(|m| m.content.clone()).unwrap_or_default();
    let first_ans = msgs.iter().find(|m| m.role == "assistant" && !m.content.is_empty()).map(|m| m.content.clone()).unwrap_or_default();
    if first_user.is_empty() {
        return Ok(());
    }
    let convo = format!(
        "User: {}\n\nAssistant: {}",
        first_user.chars().take(1500).collect::<String>(),
        first_ans.chars().take(800).collect::<String>()
    );
    let t = llm::complete(&state, json!([{"role": "system", "content": prompts::TITLE}, {"role": "user", "content": convo}]), 24, None).await?;
    let t = t.lines().next().unwrap_or("").trim().trim_matches(|c| c == '"' || c == '\'' || c == '*' || c == '#').trim().to_string();
    if !t.is_empty() {
        let t: String = t.chars().take(60).collect();
        state.db.rename_chat(&chat.id, &t)?;
        emit(app, &chat.id, "title", json!({"title": t}));
    }
    Ok(())
}

/// Plain-text transcript line for the compressor (tool output truncated, it is rarely needed verbatim).
fn render_for_summary(m: &Message) -> String {
    match m.role.as_str() {
        "user" => format!("USER: {}", m.content),
        "assistant" => {
            let mut s = format!("ASSISTANT: {}", m.content);
            if let Some(tc) = m.meta["tool_calls"].as_array().filter(|a| !a.is_empty()) {
                let names: Vec<String> = tc
                    .iter()
                    .map(|t| format!("{}({})", t["name"].as_str().unwrap_or(""), t["arguments"].as_str().unwrap_or("").chars().take(200).collect::<String>()))
                    .collect();
                s.push_str(&format!("\n[called tools: {}]", names.join(", ")));
            }
            s
        }
        "tool" => format!("TOOL RESULT ({}): {}", m.meta["name"].as_str().unwrap_or(""), m.content.chars().take(1500).collect::<String>()),
        _ => String::new(),
    }
}

/// Summarise everything except the latest turn(s) into the chat summary. Shown in the chat as a progress card.
pub async fn compress(app: &AppHandle, chat_id: &str, cancel: &AtomicBool) -> Result<()> {
    let state = app.state::<Arc<AppState>>().inner().clone();
    let chat = state.db.chat(chat_id)?.ok_or_else(|| anyhow!("chat not found"))?;
    let all = state.db.messages(chat_id)?;
    let live: Vec<&Message> = all.iter().filter(|m| !m.compressed && m.role != "event").collect();
    let user_seqs: Vec<i64> = live.iter().filter(|m| m.role == "user").map(|m| m.seq).collect();
    if user_seqs.len() < 2 {
        return Ok(()); // nothing older than the current turn
    }
    let b = budget(&state, &chat);
    // keep the last two turns when they are small, otherwise just the last one
    let last2 = user_seqs[user_seqs.len().saturating_sub(2)];
    let tail_tokens: u32 = live.iter().filter(|m| m.seq >= last2).map(|m| llm::estimate_tokens(&m.content)).sum();
    let keep_from = if user_seqs.len() >= 3 && tail_tokens < b / 3 { last2 } else { *user_seqs.last().unwrap() };
    let old: Vec<&&Message> = live.iter().filter(|m| m.seq < keep_from).collect();
    if old.is_empty() {
        return Ok(());
    }
    let before = chat.ctx_used.max(1);
    let ev = state.db.add_message(chat_id, "event", "", "", &json!({"kind": "compress", "state": "running", "before": before, "count": old.len()}))?;
    emit(app, chat_id, "message", json!({"message": ev}));

    // map-reduce over parts that fit into ~55% of the window
    let max_chars = ((b as f32 * 0.55) * 3.4) as usize;
    let mut parts: Vec<String> = vec![String::new()];
    for m in &old {
        let r = render_for_summary(m);
        let r: String = r.chars().take(max_chars / 2).collect();
        if parts.last().unwrap().len() + r.len() > max_chars {
            parts.push(String::new());
        }
        let p = parts.last_mut().unwrap();
        p.push_str(&r);
        p.push_str("\n\n");
    }
    let mut summary = chat.summary.clone();
    let target = 1400u32;
    let n_parts = parts.len();
    for (pi, part) in parts.iter().enumerate() {
        if cancel.load(Ordering::Relaxed) {
            break;
        }
        let mut input = String::new();
        if !summary.trim().is_empty() {
            input.push_str("<previous_summary>\n");
            input.push_str(&summary);
            input.push_str("\n</previous_summary>\n\n");
        }
        input.push_str("<conversation>\n");
        input.push_str(part);
        input.push_str("</conversation>");
        let body = json!({
            "messages": [{"role": "system", "content": prompts::COMPRESS}, {"role": "user", "content": input}],
            "max_tokens": 2600,
            "temperature": 0.2,
            "chat_template_kwargs": {"enable_thinking": false},
        });
        let ev_id = ev.id.clone();
        let mut last = Instant::now();
        let r = llm::stream_chat(&state, body, cancel, |d| {
            if let Delta::Progress { tokens, .. } = d {
                if last.elapsed().as_millis() > 120 {
                    let frac = ((pi as f32) + (tokens as f32 / target as f32).min(0.97)) / n_parts as f32;
                    emit(app, chat_id, "compress_progress", json!({"id": ev_id, "progress": frac}));
                    last = Instant::now();
                }
            }
        })
        .await;
        match r {
            Ok(r) if !r.cancelled && !r.content.trim().is_empty() => summary = r.content.trim().to_string(),
            Ok(_) => break,
            Err(e) => {
                let meta = json!({"kind": "compress", "state": "error", "error": e.to_string()});
                state.db.update_message(&ev.id, "", "", &meta)?;
                emit(app, chat_id, "message_update", json!({"message": {"id": ev.id, "meta": meta}}));
                return Err(e);
            }
        }
    }
    if cancel.load(Ordering::Relaxed) || summary == chat.summary {
        let meta = json!({"kind": "compress", "state": "cancelled"});
        state.db.update_message(&ev.id, "", "", &meta)?;
        emit(app, chat_id, "message_update", json!({"message": {"id": ev.id, "meta": meta}}));
        return Ok(());
    }
    state.db.set_summary(chat_id, &summary)?;
    state.db.mark_compressed(chat_id, keep_from)?;
    // estimate the new prompt size
    let chat2 = state.db.chat(chat_id)?.ok_or_else(|| anyhow!("chat not found"))?;
    let project = state.db.chat_project(&chat2)?;
    let msgs = build_messages(&state, &chat2, project.as_ref(), false)?;
    let after = llm::estimate_tokens(&serde_json::to_string(&msgs)?);
    state.db.set_chat_ctx(chat_id, after)?;
    let meta = json!({"kind": "compress", "state": "done", "before": before, "after": after, "count": old.len(), "summary": summary});
    state.db.update_message(&ev.id, "", "", &meta)?;
    emit(app, chat_id, "message_update", json!({"message": {"id": ev.id, "meta": meta}}));
    emit(app, chat_id, "compressed", json!({"upto_seq": keep_from, "ctx_used": after, "ctx_budget": budget(&state, &chat2)}));
    Ok(())
}

/// The whole conversation (compressed messages included) as a Markdown document.
pub fn export_markdown(chat: &Chat, messages: &[Message]) -> String {
    let mut out = format!("# {}\n\n_Exported from Lumen · {}_\n", chat.title, ymd(chat.created / 1000));
    for m in messages {
        match m.role.as_str() {
            "user" => {
                out.push_str("\n## You\n\n");
                out.push_str(m.content.trim());
                out.push('\n');
            }
            "assistant" => {
                if m.content.trim().is_empty() {
                    continue;
                }
                out.push_str("\n## Lumen\n\n");
                if !m.reasoning.trim().is_empty() {
                    out.push_str("<details><summary>Reasoning</summary>\n\n");
                    out.push_str(m.reasoning.trim());
                    out.push_str("\n\n</details>\n\n");
                }
                out.push_str(m.content.trim());
                out.push('\n');
            }
            "tool" => {
                let name = m.meta["name"].as_str().unwrap_or("tool");
                let args: Value = serde_json::from_str(m.meta["arguments"].as_str().unwrap_or("{}")).unwrap_or_default();
                let arg = ["path", "query", "url", "pattern", "name", "command", "expression"].iter().find_map(|k| args[*k].as_str()).unwrap_or("");
                out.push_str(&format!("\n> Tool `{name}` {arg}\n"));
            }
            _ => {}
        }
    }
    out
}

/// Manual compression triggered from the UI.
pub async fn compress_now(app: AppHandle, chat_id: String) -> Result<()> {
    let state = app.state::<Arc<AppState>>().inner().clone();
    let cancel = register_cancel(&state, &chat_id)?;
    emit(&app, &chat_id, "busy", json!({"busy": true}));
    let r = compress(&app, &chat_id, &cancel).await;
    state.generations.lock().remove(&chat_id);
    emit(&app, &chat_id, "busy", json!({"busy": false}));
    r
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clipping_keeps_start_and_end() {
        let s: String = (0..5000).map(|i| format!("line {i}\n")).collect();
        let c = clip_tool_output(&s, 1500);
        assert!(c.len() < 5200 && c.starts_with("line 0\n") && c.trim_end().ends_with("line 4999"));
        assert!(c.contains("cut to fit the context window"));
        assert_eq!(clip_tool_output("short", 1500), "short");
    }

    /// The field case: "check the project and see what features may be missing" used all 40 rounds
    /// (85 tool calls) and the turn ended on a tool result with no answer. The last round now asks for
    /// the answer with tools off, with a one-off note after the stored history.
    #[test]
    fn rounds_running_out_end_with_an_answer() {
        let history = vec![
            json!({"role": "system", "content": "sys"}),
            json!({"role": "user", "content": "Hey, check the project and see what features may be missing"}),
            json!({"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "read_file", "arguments": "{}"}}]}),
            json!({"role": "tool", "tool_call_id": "c1", "content": "…"}),
        ];
        let req = final_round_messages(history.clone());
        assert_eq!(req.len(), history.len() + 1);
        assert_eq!(&req[..history.len()], &history[..], "the stored history is sent unchanged (prompt cache)");
        assert_eq!(req.last().unwrap()["role"], "user", "a trailing system message breaks some chat templates");
        assert!(req.last().unwrap()["content"].as_str().unwrap().contains("Answer the user's request now"));
        assert!(rounds_left_note(WRAP_UP_AT).contains("5 tool rounds left"));
    }

    /// Real first messages from the field: each chat gets a readable name the moment it is sent.
    #[test]
    fn first_messages_make_names() {
        assert_eq!(quick_title("Develop what is written inside the TODO.txt").as_deref(), Some("Develop what is written inside the TODO.txt"));
        assert_eq!(
            quick_title("Hey, check the project and see what features may be missing").as_deref(),
            Some("Check the project and see what features may be…")
        );
        assert_eq!(quick_title("can you look at https://github.com/x/y and port it?").as_deref(), Some("Look at and port it"));
        assert_eq!(quick_title("Salut, te rog explică-mi cum funcționează un transformer").as_deref(), Some("Explică-mi cum funcționează un transformer"));
        assert_eq!(quick_title("## Plan\nsecond line").as_deref(), Some("Plan"));
        assert_eq!(quick_title("hi"), None, "no topic: the model's title comes after the reply");
        assert_eq!(quick_title("   \n  "), None);
        assert!(quick_title(&"word ".repeat(40)).unwrap().chars().count() <= 49);
    }

    /// An unloaded model keeps the meter's scale from the last load instead of falling to 2048.
    #[test]
    fn context_window_survives_unload() {
        let state = AppState::new().unwrap();
        let mut chat = state.db.create_chat(None, "New chat").unwrap();
        state.settings.write().last_n_ctx = 0;
        state.info.write().n_ctx = 0;
        assert_eq!(known_budget(&state, &chat), 0, "nothing known: the meter shows only what is used");
        assert_eq!(budget(&state, &chat), 2048);
        state.settings.write().last_n_ctx = 65536;
        assert_eq!(known_budget(&state, &chat), 65536, "unloaded: last model's window");
        state.info.write().n_ctx = 40960;
        assert_eq!(known_budget(&state, &chat), 40960, "loaded model wins");
        chat.ctx_limit = 16384;
        assert_eq!(budget(&state, &chat), 16384, "the chat's own cap still applies");
        state.info.write().n_ctx = 0;
        state.settings.write().last_n_ctx = 0;
        assert_eq!(known_budget(&state, &chat), 16384);
    }

    #[test]
    fn overflow_errors_are_recognised() {
        assert!(is_context_overflow(&anyhow!("llama-server 400 Bad Request: the request exceeds the available context size, try increasing it")));
        // the exact message reported from the field
        assert!(is_context_overflow(&anyhow!("llama-server 400 Bad Request: request (40678 tokens) exceeds the available context size (39680 tokens), try increasing it")));
        assert!(!is_context_overflow(&anyhow!("connection refused")));
    }

    /// A single-turn audit that read many files: old tool results get stubbed until the prompt fits.
    #[test]
    fn long_turn_is_pruned_to_fit() {
        let state = AppState::new().unwrap();
        state.info.write().n_ctx = 8192;
        let chat = state.db.create_chat(None, "New chat").unwrap();
        state.db.add_message(&chat.id, "user", "Audit the whole project", "", &json!({})).unwrap();
        let big: String = "fn x() { let y = 1; }\n".repeat(400); // ~8.8k chars each
        for i in 0..8 {
            let id = format!("call{i}");
            state.db.add_message(&chat.id, "assistant", "", "", &json!({"tool_calls": [{"id": id, "name": "read_file", "arguments": format!("{{\"path\":\"f{i}.rs\"}}")}]})).unwrap();
            state.db.add_message(&chat.id, "tool", &big, "", &json!({"tool_call_id": id, "name": "read_file", "arguments": format!("{{\"path\":\"f{i}.rs\"}}")})).unwrap();
        }
        let c = state.db.chat(&chat.id).unwrap().unwrap();
        let before = prompt_estimate(&state, &c, None, false, &[]).unwrap();
        assert!(before as f32 > 8192.0 * 0.72, "test setup must overflow: {before}");
        let after = fit_context(&state, &chat.id, None, false, &[], (0.72, 0.5, 4)).unwrap();
        println!("estimate {before} -> {after} tokens (window 8192)");
        assert!(after as f32 <= 8192.0 * 0.6);
        let msgs = state.db.messages(&chat.id).unwrap();
        let tools: Vec<&Message> = msgs.iter().filter(|m| m.role == "tool").collect();
        assert!(tools[0].meta["pruned"].as_bool().unwrap_or(false));
        assert_eq!(tools[0].meta["full"].as_str().unwrap(), big);
        assert!(tools[0].content.contains("read_file(f0.rs) output removed"));
        assert!(!tools.last().unwrap().meta["pruned"].as_bool().unwrap_or(false), "the latest results stay");
        // already fitting: nothing more changes, so the prompt stays byte-stable
        let again = fit_context(&state, &chat.id, None, false, &[], (0.72, 0.5, 4)).unwrap();
        assert_eq!(again, after);
        state.db.delete_chat(&chat.id).unwrap();
    }

    /// Attached files and images reach the prompt (text inline, images as data URLs) and survive a
    /// reload of the chat; an image is refused while the model can't see; a chat's assistant sets the
    /// role and the temperature.
    #[test]
    fn attachments_and_assistants_reach_the_prompt() {
        use base64::Engine;
        let state = AppState::new().unwrap();
        let dir = std::env::temp_dir().join(format!("lumen-attach-{}", crate::state::new_id()));
        std::fs::create_dir_all(&dir).unwrap();
        let note = dir.join("notes.md");
        std::fs::write(&note, "# Plan\nShip the release on Friday.").unwrap();
        let png = dir.join("dot.png");
        let png_bytes = base64::engine::general_purpose::STANDARD
            .decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGNgYPj/HwADAgH/eL9NAAAAAElFTkSuQmCC")
            .unwrap();
        std::fs::write(&png, &png_bytes).unwrap();
        let att = |p: &Path| Attachment { name: p.file_name().unwrap().to_string_lossy().into(), path: Some(p.to_string_lossy().into()), data: None };

        // a model without its projector: images are refused with a hint, files still work
        state.info.write().vision = false;
        let mut meta = json!({});
        let err = take_attachments(&state, &[att(&png)], &mut meta).unwrap_err().to_string();
        assert!(err.contains("can't see images"), "{err}");
        take_attachments(&state, &[att(&note)], &mut meta).unwrap();
        assert!(meta["attached"].as_str().unwrap().contains("<attached_file name=\"notes.md\">\n# Plan\nShip the release on Friday."));
        assert_eq!(meta["files"][0]["name"], "notes.md");

        // with vision: the image is copied into data/attachments
        state.info.write().vision = true;
        let pasted = Attachment { name: "Pasted".into(), path: None, data: Some(format!("data:image/png;base64,{}", base64::engine::general_purpose::STANDARD.encode(&png_bytes))) };
        take_attachments(&state, &[att(&png), pasted], &mut meta).unwrap();
        let images = meta["images"].as_array().unwrap().clone();
        assert_eq!(images.len(), 2);
        for i in &images {
            assert_eq!(std::fs::read(state.paths.attachments.join(i["file"].as_str().unwrap())).unwrap(), png_bytes);
        }
        // a folder is not a file to attach
        assert!(take_attachments(&state, &[att(&dir)], &mut json!({})).unwrap_err().to_string().contains("folder"));

        // the message as the model gets it
        let chat = state.db.create_chat(None, "New chat").unwrap();
        meta["prompt_content"] = json!(format!("What is in these?{}", meta["attached"].as_str().unwrap()));
        state.db.add_message(&chat.id, "user", "What is in these?", "", &meta).unwrap();
        state.settings.write().assistants = crate::state::default_assistants();
        state.db.set_chat_assistant(&chat.id, Some("translator")).unwrap();
        let c = state.db.chat(&chat.id).unwrap().unwrap();
        assert_eq!(c.assistant_id.as_deref(), Some("translator"));
        let msgs = build_messages(&state, &c, None, false).unwrap();
        assert!(msgs[0]["content"].as_str().unwrap().contains("# Your role in this chat: Translator"));
        let parts = msgs[1]["content"].as_array().expect("text + images");
        assert!(parts[0]["text"].as_str().unwrap().contains("Ship the release on Friday."));
        assert_eq!(parts.iter().filter(|p| p["type"] == "image_url").count(), 2);
        assert!(parts[1]["image_url"]["url"].as_str().unwrap().starts_with("data:image/png;base64,iVBOR"));
        assert_eq!(gen_params_for(&state, &c, false)["temperature"], json!(0.3f32));
        // an image costs ~1000 tokens in the estimate, not its base64 length
        let est = llm::estimate_tokens(&serde_json::to_string(&msgs).unwrap());
        assert!(est < 6000, "{est}");
        // after switching to a model without vision the images become a note instead of a refused request
        state.info.write().vision = false;
        let msgs = build_messages(&state, &c, None, false).unwrap();
        assert!(msgs[1]["content"].as_str().unwrap().contains("2 image(s) were attached here"));
        // a deleted assistant: back to the default
        state.settings.write().assistants.clear();
        assert!(!build_messages(&state, &c, None, false).unwrap()[0]["content"].as_str().unwrap().contains("Your role in this chat"));

        for i in &images {
            let _ = std::fs::remove_file(state.paths.attachments.join(i["file"].as_str().unwrap()));
        }
        state.db.delete_chat(&chat.id).unwrap();
        let _ = std::fs::remove_dir_all(&dir);
    }
}
