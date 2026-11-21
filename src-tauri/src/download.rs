//! Resumable HTTP downloads used for models, the embedding model, the llama.cpp runtime and updates.
//! A download with a known SHA-256 is checked before it is moved into place.

use anyhow::{anyhow, Result};
use futures_util::StreamExt;
use serde_json::json;
use sha2::{Digest, Sha256};
use std::io::Read;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;
use tauri::{AppHandle, Emitter, Manager};
use tokio::io::AsyncWriteExt;

/// Resumable download to `dest` (via `dest.part`). Emits `download-progress` events tagged with `id`.
/// With `sha256` (lowercase hex) the finished file must match it, or it is deleted and nothing is kept.
pub async fn download(
    app: &AppHandle,
    id: &str,
    label: &str,
    url: &str,
    dest: &Path,
    sha256: Option<&str>,
    cancel: Arc<AtomicBool>,
) -> Result<()> {
    let part = dest.with_extension(format!(
        "{}.part",
        dest.extension().and_then(|e| e.to_str()).unwrap_or("bin")
    ));
    if let Some(dir) = dest.parent() {
        tokio::fs::create_dir_all(dir).await?;
    }
    let mut have = tokio::fs::metadata(&part).await.map(|m| m.len()).unwrap_or(0);

    let http = app.state::<Arc<crate::state::AppState>>().http.clone();
    let mut req = http.get(url);
    if have > 0 {
        req = req.header("Range", format!("bytes={have}-"));
    }
    let resp = req.send().await?;
    let status = resp.status();
    if status.as_u16() == 416 {
        // already complete
        verify(&part, sha256).await?;
        tokio::fs::rename(&part, dest).await?;
        return Ok(());
    }
    if !status.is_success() {
        return Err(anyhow!("HTTP {status} for {url}"));
    }
    if status.as_u16() != 206 {
        have = 0; // server ignored range
    }
    let total = resp.content_length().map(|l| l + have).unwrap_or(0);

    let mut file = tokio::fs::OpenOptions::new()
        .create(true)
        .write(true)
        .append(have > 0)
        .truncate(have == 0)
        .open(&part)
        .await?;

    let mut stream = resp.bytes_stream();
    let mut done = have;
    let started = Instant::now();
    let mut last_emit = Instant::now();
    let mut speed_mark = (Instant::now(), done);
    let mut speed = 0.0f64;

    while let Some(chunk) = stream.next().await {
        if cancel.load(Ordering::Relaxed) {
            file.flush().await.ok();
            let _ = app.emit("download-progress", json!({"id": id, "label": label, "cancelled": true}));
            return Err(anyhow!("cancelled"));
        }
        let chunk = chunk?;
        file.write_all(&chunk).await?;
        done += chunk.len() as u64;
        if last_emit.elapsed().as_millis() > 300 {
            let dt = speed_mark.0.elapsed().as_secs_f64();
            if dt > 0.8 {
                let inst = (done - speed_mark.1) as f64 / dt;
                speed = if speed == 0.0 { inst } else { speed * 0.6 + inst * 0.4 };
                speed_mark = (Instant::now(), done);
            }
            let _ = app.emit(
                "download-progress",
                json!({"id": id, "label": label, "done": done, "total": total, "speed": speed, "elapsed": started.elapsed().as_secs()}),
            );
            last_emit = Instant::now();
        }
    }
    file.flush().await?;
    drop(file);
    if total > 0 && done < total {
        return Err(anyhow!("download interrupted ({done}/{total} bytes) - try again to resume"));
    }
    verify(&part, sha256).await?;
    tokio::fs::rename(&part, dest).await?;
    let _ = app.emit("download-progress", json!({"id": id, "label": label, "done": done, "total": total, "finished": true}));
    Ok(())
}

/// Checks a finished download against its published SHA-256; a mismatch deletes it (a resumed download
/// with a broken start would otherwise fail the same way forever).
async fn verify(file: &Path, sha256: Option<&str>) -> Result<()> {
    let Some(expected) = sha256 else { return Ok(()) };
    let f = file.to_path_buf();
    let actual = tokio::task::spawn_blocking(move || sha256_file(&f)).await??;
    if !actual.eq_ignore_ascii_case(expected.trim()) {
        let _ = tokio::fs::remove_file(file).await;
        return Err(anyhow!("the download is damaged or not the published file (SHA-256 mismatch) - try again"));
    }
    Ok(())
}

/// SHA-256 of a file as lowercase hex.
pub fn sha256_file(path: &Path) -> Result<String> {
    let mut f = std::fs::File::open(path)?;
    let mut h = Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 {
            break;
        }
        h.update(&buf[..n]);
    }
    Ok(h.finalize().iter().map(|b| format!("{b:02x}")).collect())
}

/// A 64-digit hex SHA-256 from `text` (`"sha256:<hex>"` as GitHub gives it, or the bare hex).
pub fn parse_sha256(text: &str) -> Option<String> {
    let hex = text.trim().strip_prefix("sha256:").unwrap_or(text.trim());
    (hex.len() == 64 && hex.chars().all(|c| c.is_ascii_hexdigit())).then(|| hex.to_ascii_lowercase())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sha256_matches_known_value() {
        let p = std::env::temp_dir().join(format!("lumen-sha-{}", crate::state::new_id()));
        std::fs::write(&p, b"abc").unwrap();
        let h = sha256_file(&p).unwrap();
        let _ = std::fs::remove_file(&p);
        assert_eq!(h, "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    }

    #[test]
    fn digests_parse_from_github_and_hugging_face() {
        let hex = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";
        assert_eq!(parse_sha256(&format!("sha256:{hex}")).as_deref(), Some(hex));
        assert_eq!(parse_sha256(&hex.to_uppercase()).as_deref(), Some(hex));
        assert_eq!(parse_sha256("sha1:abc"), None);
        assert_eq!(parse_sha256(""), None);
    }

    #[test]
    fn a_damaged_download_is_rejected_and_removed() {
        let rt = tokio::runtime::Runtime::new().unwrap();
        let p = std::env::temp_dir().join(format!("lumen-verify-{}", crate::state::new_id()));
        std::fs::write(&p, b"abc").unwrap();
        rt.block_on(async {
            assert!(verify(&p, Some("ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")).await.is_ok());
            assert!(verify(&p, None).await.is_ok());
            assert!(verify(&p, Some(&"0".repeat(64))).await.is_err());
        });
        assert!(!p.exists(), "a mismatching file is deleted");
    }
}