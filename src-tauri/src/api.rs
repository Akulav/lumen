//! Local OpenAI-compatible API (Settings → Local API, off by default): other programs on this PC – editor
//! plugins, scripts, apps that speak the OpenAI protocol – use the model Lumen has loaded through
//! `http://127.0.0.1:<port>/v1`. A small HTTP/1.1 proxy in front of llama-server: it adds llama-server's
//! private key, loads the last model when none is loaded, and streams the answer back as it comes.
//! Only this PC can reach it unless "network access" is on, which requires a key.
//!
//! Routes: `GET /v1/models`, `POST /v1/chat/completions`, `/v1/completions`, `/v1/embeddings` (also
//! without `/v1`), `GET /health`. No CORS headers: web pages can't read its answers.

use anyhow::{anyhow, Result};
use futures_util::StreamExt;
use serde_json::{json, Value};
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use tauri::{AppHandle, Manager};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};

use crate::state::AppState;

const MAX_HEADERS: usize = 64 * 1024;
const MAX_BODY: usize = 64 << 20;

/// (Re)starts the server from the settings; stops it when it is off. The window reads the result with
/// `api_status`.
pub fn restart(app: &AppHandle) {
    let state = app.state::<Arc<AppState>>().inner().clone();
    if let Some(h) = state.api.lock().take() {
        h.abort();
    }
    let (on, port, key, lan) = {
        let s = state.settings.read();
        (s.api_enabled, s.api_port, s.api_key.trim().to_string(), s.api_lan)
    };
    if !on {
        *state.api_status.write() = String::new();
        return;
    }
    if lan && key.is_empty() {
        *state.api_status.write() = "error: network access needs a key".into();
        return;
    }
    let app = app.clone();
    let task = tauri::async_runtime::spawn(async move {
        let state = app.state::<Arc<AppState>>().inner().clone();
        let addr = if lan { format!("0.0.0.0:{port}") } else { format!("127.0.0.1:{port}") };
        let listener = match TcpListener::bind(&addr).await {
            Ok(l) => l,
            Err(e) => {
                state.log(format!("[api] can't listen on {addr}: {e}"));
                *state.api_status.write() = format!("error: port {port} is not free ({e})");
                return;
            }
        };
        state.log(format!("[api] OpenAI-compatible API on http://{addr}/v1"));
        *state.api_status.write() = format!("http://127.0.0.1:{port}/v1");
        let upstream: Upstream = {
            let app = app.clone();
            Arc::new(move || {
                let app = app.clone();
                Box::pin(async move { ensure_model(&app).await })
            })
        };
        loop {
            let Ok((sock, peer)) = listener.accept().await else { continue };
            let (state, upstream, key) = (state.clone(), upstream.clone(), key.clone());
            tokio::spawn(async move {
                if let Err(e) = handle(sock, &key, &state.local, upstream, || models_list(&state), || state.touch()).await {
                    state.log(format!("[api] {peer}: {e}"));
                }
            });
        }
    });
    *state.api.lock() = Some(task);
}

/// Base URL of the running chat model; loads the last one when nothing is loaded (waits while one is
/// loading).
async fn ensure_model(app: &AppHandle) -> Result<String> {
    let state = app.state::<Arc<AppState>>().inner().clone();
    for _ in 0..1200 {
        if let Some(u) = state.llm_url() {
            return Ok(u);
        }
        let busy = matches!(state.info.read().state.as_str(), "starting" | "runtime");
        if !busy {
            let last = state.settings.read().last_model.clone().filter(|m| state.paths.models.join(m).exists());
            let m = last.ok_or_else(|| anyhow!("No model is loaded in Lumen. Load one in Lumen's Models page."))?;
            state.log(format!("[api] a request came in with no model loaded: loading {m}"));
            crate::server::start_llm(app, &m).await?;
            continue;
        }
        tokio::time::sleep(std::time::Duration::from_millis(500)).await;
    }
    Err(anyhow!("the model took too long to load"))
}

fn models_list(state: &AppState) -> Value {
    let model = state.info.read().model.clone().or_else(|| state.settings.read().last_model.clone());
    let data: Vec<Value> = model.into_iter().map(|m| json!({"id": m, "object": "model", "owned_by": "lumen", "created": 0})).collect();
    json!({"object": "list", "data": data})
}

type Upstream = Arc<dyn Fn() -> Pin<Box<dyn Future<Output = Result<String>> + Send>> + Send + Sync>;

struct Request {
    method: String,
    path: String,
    auth: Option<String>,
    body: Vec<u8>,
}

/// Reads one request (headers + Content-Length body). Errors carry the HTTP status to answer with.
async fn read_request(sock: &mut TcpStream) -> Result<Request, (u16, String)> {
    let mut buf = Vec::with_capacity(4096);
    let mut chunk = [0u8; 8192];
    let head_end = loop {
        if let Some(i) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
            break i;
        }
        if buf.len() > MAX_HEADERS {
            return Err((431, "headers too large".into()));
        }
        let n = sock.read(&mut chunk).await.map_err(|e| (400, e.to_string()))?;
        if n == 0 {
            return Err((400, "connection closed".into()));
        }
        buf.extend_from_slice(&chunk[..n]);
    };
    let head = String::from_utf8_lossy(&buf[..head_end]).into_owned();
    let mut lines = head.split("\r\n");
    let mut first = lines.next().unwrap_or("").split_whitespace();
    let (method, target) = (first.next().unwrap_or("").to_string(), first.next().unwrap_or("/").to_string());
    let mut len = 0usize;
    let mut auth = None;
    for l in lines {
        let Some((k, v)) = l.split_once(':') else { continue };
        match k.trim().to_ascii_lowercase().as_str() {
            "content-length" => len = v.trim().parse().map_err(|_| (400, "bad Content-Length".to_string()))?,
            "authorization" => auth = Some(v.trim().to_string()),
            "transfer-encoding" if v.to_ascii_lowercase().contains("chunked") => return Err((411, "send the body with a Content-Length".into())),
            _ => {}
        }
    }
    if len > MAX_BODY {
        return Err((413, "request too large".into()));
    }
    let mut body = buf[head_end + 4..].to_vec();
    while body.len() < len {
        let n = sock.read(&mut chunk).await.map_err(|e| (400, e.to_string()))?;
        if n == 0 {
            return Err((400, "body cut short".into()));
        }
        body.extend_from_slice(&chunk[..n]);
    }
    body.truncate(len);
    let path = target.split('?').next().unwrap_or("/").trim_end_matches('/').to_string();
    Ok(Request { method, path, auth, body })
}

fn reason(code: u16) -> &'static str {
    match code {
        200 => "OK",
        400 => "Bad Request",
        401 => "Unauthorized",
        404 => "Not Found",
        405 => "Method Not Allowed",
        411 => "Length Required",
        413 => "Payload Too Large",
        431 => "Request Header Fields Too Large",
        503 => "Service Unavailable",
        _ => "Error",
    }
}

async fn reply_json(sock: &mut TcpStream, code: u16, v: &Value) -> std::io::Result<()> {
    let body = v.to_string();
    let head = format!("HTTP/1.1 {code} {}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n", reason(code), body.len());
    sock.write_all(head.as_bytes()).await?;
    sock.write_all(body.as_bytes()).await?;
    sock.flush().await
}

fn error_body(msg: &str, kind: &str) -> Value {
    json!({"error": {"message": msg, "type": kind}})
}

/// One connection, one request (answered with `Connection: close`).
async fn handle(
    mut sock: TcpStream,
    key: &str,
    client: &reqwest::Client,
    upstream: Upstream,
    models: impl Fn() -> Value,
    touch: impl Fn(),
) -> Result<()> {
    let req = match tokio::time::timeout(std::time::Duration::from_secs(60), read_request(&mut sock)).await {
        Ok(Ok(r)) => r,
        Ok(Err((code, msg))) => return Ok(reply_json(&mut sock, code, &error_body(&msg, "invalid_request_error")).await?),
        Err(_) => return Ok(()),
    };
    if !key.is_empty() && req.auth.as_deref() != Some(&format!("Bearer {key}")) {
        return Ok(reply_json(&mut sock, 401, &error_body("missing or wrong API key (Authorization: Bearer <key>)", "authentication_error")).await?);
    }
    let path = req.path.strip_prefix("/v1").unwrap_or(&req.path);
    match (req.method.as_str(), path) {
        ("GET", "" | "/health") => Ok(reply_json(&mut sock, 200, &json!({"status": "ok", "server": "Lumen"})).await?),
        ("GET", "/models") => Ok(reply_json(&mut sock, 200, &models()).await?),
        ("POST", "/chat/completions" | "/completions" | "/embeddings") => {
            let base = match upstream().await {
                Ok(b) => b,
                Err(e) => return Ok(reply_json(&mut sock, 503, &error_body(&e.to_string(), "server_error")).await?),
            };
            touch();
            let resp = client
                .post(format!("{base}/v1{path}"))
                .header("Content-Type", "application/json")
                .body(req.body)
                .send()
                .await;
            let resp = match resp {
                Ok(r) => r,
                Err(e) => return Ok(reply_json(&mut sock, 503, &error_body(&format!("the model server did not answer: {e}"), "server_error")).await?),
            };
            let code = resp.status().as_u16();
            let ctype = resp.headers().get("content-type").and_then(|v| v.to_str().ok()).unwrap_or("application/json").to_string();
            let head = format!("HTTP/1.1 {code} {}\r\nContent-Type: {ctype}\r\nTransfer-Encoding: chunked\r\nCache-Control: no-cache\r\nConnection: close\r\n\r\n", reason(code));
            sock.write_all(head.as_bytes()).await?;
            let mut stream = resp.bytes_stream();
            while let Some(chunk) = stream.next().await {
                let chunk = chunk?;
                if chunk.is_empty() {
                    continue;
                }
                // (a client that hangs up ends the copy, which closes the request: llama-server stops)
                sock.write_all(format!("{:x}\r\n", chunk.len()).as_bytes()).await?;
                sock.write_all(&chunk).await?;
                sock.write_all(b"\r\n").await?;
                sock.flush().await?;
                touch();
            }
            sock.write_all(b"0\r\n\r\n").await?;
            sock.flush().await?;
            touch();
            Ok(())
        }
        (_, "/chat/completions" | "/completions" | "/embeddings" | "/models") => Ok(reply_json(&mut sock, 405, &error_body("method not allowed", "invalid_request_error")).await?),
        _ => Ok(reply_json(&mut sock, 404, &error_body(&format!("unknown endpoint {}", req.path), "invalid_request_error")).await?),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A fake llama-server: checks it got llama-server's key, answers with a two-event SSE stream.
    async fn fake_upstream() -> String {
        let l = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = l.local_addr().unwrap().port();
        tokio::spawn(async move {
            loop {
                let (mut s, _) = l.accept().await.unwrap();
                tokio::spawn(async move {
                    let req = read_request(&mut s).await.unwrap();
                    assert_eq!(req.auth.as_deref(), Some("Bearer inner"), "the proxy sends llama-server's key, not the client's");
                    let body: Value = serde_json::from_slice(&req.body).unwrap();
                    assert_eq!(body["messages"][0]["content"], "hi");
                    let sse = "data: {\"choices\":[{\"delta\":{\"content\":\"Hel\"}}]}\n\ndata: {\"choices\":[{\"delta\":{\"content\":\"lo\"}}]}\n\ndata: [DONE]\n\n";
                    let head = format!("HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n", sse.len());
                    s.write_all(head.as_bytes()).await.unwrap();
                    s.write_all(sse.as_bytes()).await.unwrap();
                });
            }
        });
        format!("http://127.0.0.1:{port}")
    }

    /// The whole path with a real HTTP client: key check, model list, a streamed chat completion
    /// relayed piece by piece, unknown routes, and the error when no model can be loaded.
    #[test]
    fn proxies_openai_requests() {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let upstream_url = fake_upstream().await;
            let l = TcpListener::bind("127.0.0.1:0").await.unwrap();
            let port = l.local_addr().unwrap().port();
            let inner = {
                let mut h = reqwest::header::HeaderMap::new();
                h.insert(reqwest::header::AUTHORIZATION, "Bearer inner".parse().unwrap());
                reqwest::Client::builder().default_headers(h).no_proxy().build().unwrap()
            };
            tokio::spawn(async move {
                let mut n = 0;
                loop {
                    let (s, _) = l.accept().await.unwrap();
                    n += 1;
                    // the last connection pretends no model can be loaded
                    let up: Upstream = if n < 6 {
                        let u = upstream_url.clone();
                        Arc::new(move || {
                            let u = u.clone();
                            Box::pin(async move { Ok(u) })
                        })
                    } else {
                        Arc::new(|| Box::pin(async { Err(anyhow!("No model is loaded in Lumen.")) }))
                    };
                    let inner = inner.clone();
                    tokio::spawn(async move {
                        handle(s, "secret", &inner, up, || json!({"object": "list", "data": [{"id": "m.gguf"}]}), || {}).await.unwrap();
                    });
                }
            });
            let base = format!("http://127.0.0.1:{port}/v1");
            let c = reqwest::Client::builder().no_proxy().build().unwrap();
            // 1: no key
            let r = c.get(format!("{base}/models")).send().await.unwrap();
            assert_eq!(r.status(), 401);
            // 2: models
            let r = c.get(format!("{base}/models")).bearer_auth("secret").send().await.unwrap();
            assert_eq!(r.json::<Value>().await.unwrap()["data"][0]["id"], "m.gguf");
            // 3: streamed chat completion
            let r = c
                .post(format!("{base}/chat/completions"))
                .bearer_auth("secret")
                .json(&json!({"model": "x", "stream": true, "messages": [{"role": "user", "content": "hi"}]}))
                .send()
                .await
                .unwrap();
            assert_eq!(r.status(), 200);
            assert!(r.headers()["content-type"].to_str().unwrap().starts_with("text/event-stream"));
            let text = r.text().await.unwrap();
            assert!(text.contains("\"Hel\"") && text.contains("\"lo\"") && text.contains("[DONE]"), "{text}");
            // 4: unknown route, 5: wrong method
            assert_eq!(c.get(format!("{base}/nope")).bearer_auth("secret").send().await.unwrap().status(), 404);
            assert_eq!(c.get(format!("{base}/chat/completions")).bearer_auth("secret").send().await.unwrap().status(), 405);
            // 6: no model
            let r = c.post(format!("{base}/chat/completions")).bearer_auth("secret").json(&json!({"messages": []})).send().await.unwrap();
            assert_eq!(r.status(), 503);
            assert!(r.json::<Value>().await.unwrap()["error"]["message"].as_str().unwrap().contains("No model"));
        });
    }
}
