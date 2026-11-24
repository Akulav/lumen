//! Portable llama.cpp runtime: downloaded once from the official GitHub releases into LumenData/runtime.
//! Windows: the `win-<backend>-x64.zip` builds; Linux: `ubuntu-<backend>-x64.tar.gz`; macOS (Apple
//! Silicon): `macos-arm64.tar.gz`, which runs on the GPU through Metal.
//! CUDA (NVIDIA, Windows + Linux) comes as two files: the build (`…-bin-win-cuda-12.4-x64.zip`) and NVIDIA's
//! runtime libraries (`cudart-…` with the same suffix), which go next to llama-server. CUDA 12 or 13 is
//! picked from what `nvidia-smi` reports (`cuda_major`).

use anyhow::{anyhow, Context, Result};
use serde_json::Value;
use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use tauri::{AppHandle, Manager};

use crate::state::{AppState, LUMEN_UA};

/// llama-server's file name on this system.
const SERVER: &str = if cfg!(windows) { "llama-server.exe" } else { "llama-server" };

/// The backend actually used: macOS has one build (Metal), so any GPU choice means "metal" there.
pub fn effective_backend(backend: &str) -> String {
    if cfg!(target_os = "macos") && backend != "cpu" {
        "metal".into()
    } else if backend == "auto" {
        // (only until the first start has resolved it: `auto_backend`)
        "vulkan".into()
    } else {
        backend.to_string()
    }
}

pub fn runtime_dir(state: &AppState, backend: &str) -> PathBuf {
    state.paths.runtime.join(format!("llama-{}", effective_backend(backend)))
}

pub fn server_exe(state: &AppState, backend: &str) -> PathBuf {
    runtime_dir(state, backend).join(SERVER)
}

pub fn installed_version(state: &AppState, backend: &str) -> Option<String> {
    let dir = runtime_dir(state, backend);
    if !dir.join(SERVER).exists() {
        return None;
    }
    Some(std::fs::read_to_string(dir.join("version.txt")).unwrap_or_else(|_| "unknown".into()).trim().to_string())
}

/// Is `name` the llama.cpp release file for this system and build (`backend`, or `cuda-12` / `cuda-13`)?
fn asset_matches(name: &str, build: &str) -> bool {
    if !name.starts_with("llama-") {
        return false;
    }
    if cfg!(windows) {
        let part = match build {
            "rocm" => "-bin-win-rocm",
            "cpu" => "-bin-win-cpu-x64",
            "cuda-12" => "-bin-win-cuda-12.",
            "cuda-13" => "-bin-win-cuda-13.",
            _ => "-bin-win-vulkan-x64",
        };
        name.contains(part) && name.ends_with(".zip") && name.contains("x64")
    } else if cfg!(target_os = "macos") {
        // (the same Metal build also runs CPU-only with --gpu-layers 0)
        name.ends_with("-bin-macos-arm64.tar.gz")
    } else {
        match build {
            "rocm" => name.contains("-bin-ubuntu-rocm-") && name.ends_with("-x64.tar.gz"),
            "cpu" => name.ends_with("-bin-ubuntu-x64.tar.gz"),
            "cuda-12" | "cuda-13" => name.contains(&format!("-bin-ubuntu-{build}.")) && name.ends_with("-x64.tar.gz"),
            _ => name.ends_with("-bin-ubuntu-vulkan-x64.tar.gz"),
        }
    }
}

/// NVIDIA's runtime libraries for a CUDA build: the `cudart-…` file of the same release with the same
/// ending (`cudart-llama-bin-win-cuda-12.4-x64.zip` for `llama-b123-bin-win-cuda-12.4-x64.zip`).
fn cudart_matches(name: &str, build_file: &str, tag: &str) -> bool {
    let Some(suffix) = build_file.strip_prefix(&format!("llama-{tag}")) else { return false };
    name.starts_with("cudart-") && name.ends_with(suffix)
}

/// CUDA build for this NVIDIA setup: 13.x needs a driver that supports CUDA 13 and a Turing or newer GPU
/// (compute capability 7.5+; CUDA 13 dropped Maxwell, Pascal and Volta), anything else gets 12.x.
fn cuda_major(driver_cuda: u32, lowest_cap: f32) -> u32 {
    if driver_cuda >= 13 && lowest_cap >= 7.5 { 13 } else { 12 }
}

/// (CUDA version the driver supports, lowest compute capability of the NVIDIA GPUs) from `nvidia-smi`.
fn parse_nvidia_smi(summary: &str, caps: &str) -> Option<(u32, f32)> {
    let driver = regex::Regex::new(r"CUDA Version:\s*(\d+)").ok()?.captures(summary)?[1].parse().ok()?;
    let cap = caps.lines().filter_map(|l| l.trim().parse::<f32>().ok()).fold(None, |m: Option<f32>, c| Some(m.map_or(c, |m| m.min(c))));
    Some((driver, cap.unwrap_or(0.0)))
}

async fn nvidia_smi(args: &[&str]) -> Option<String> {
    let mut cmd = tokio::process::Command::new("nvidia-smi");
    cmd.args(args);
    crate::platform::quiet(&mut cmd);
    let out = cmd.output().await.ok()?;
    out.status.success().then(|| String::from_utf8_lossy(&out.stdout).into_owned())
}

/// The build to download for `backend`: itself, or for CUDA `cuda-12` / `cuda-13` after asking the driver.
/// Fails early (before a ~600 MB download) without an NVIDIA driver or with one too old for CUDA 12.
async fn build_for(state: &AppState, backend: &str) -> Result<String> {
    if backend != "cuda" {
        return Ok(backend.to_string());
    }
    let summary = nvidia_smi(&[]).await.ok_or_else(|| {
        anyhow!("CUDA needs an NVIDIA graphics card and its driver (nvidia-smi was not found). Pick Vulkan in Settings for other cards.")
    })?;
    let caps = nvidia_smi(&["--query-gpu=compute_cap", "--format=csv,noheader"]).await.unwrap_or_default();
    let (driver, cap) = parse_nvidia_smi(&summary, &caps).ok_or_else(|| anyhow!("could not read the CUDA version from nvidia-smi"))?;
    if driver < 12 {
        return Err(anyhow!("the NVIDIA driver supports only CUDA {driver}; CUDA 12 or newer is needed - update the driver, or pick Vulkan in Settings"));
    }
    let major = cuda_major(driver, cap);
    state.log(format!("[runtime] NVIDIA driver supports CUDA {driver}, lowest compute capability {cap} -> CUDA {major} build"));
    Ok(format!("cuda-{major}"))
}

/// A llama.cpp download: the build, and for CUDA NVIDIA's runtime libraries.
pub struct Release {
    pub tag: String,
    pub url: String,
    pub name: String,
    /// SHA-256 GitHub publishes for the file (its `digest`): a download must match it
    pub sha256: String,
    /// (url, file name, SHA-256) of the `cudart-…` file
    pub cudart: Option<(String, String, String)>,
}

/// Newest llama.cpp release that has a build for this system and `backend` (a settings value).
pub async fn latest_release(state: &AppState, backend: &str) -> Result<Release> {
    let build = build_for(state, &effective_backend(backend)).await?;
    let releases: Value = state
        .http
        .get("https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=20")
        .header("User-Agent", LUMEN_UA)
        .header("Accept", "application/vnd.github+json")
        .timeout(std::time::Duration::from_secs(30))
        .send()
        .await?
        .error_for_status()?
        .json()
        .await?;
    releases
        .as_array()
        .into_iter()
        .flatten()
        .find_map(|r| pick_release(r, &build))
        .ok_or_else(|| anyhow!("no llama.cpp release has a build for '{build}' on this system"))
}

/// The files of one GitHub release for `build`; None if it lacks one (a CUDA build also needs its cudart file)
/// or GitHub has no SHA-256 for it (the runtime is a program: it is never run unchecked).
fn pick_release(r: &Value, build: &str) -> Option<Release> {
    let tag = r["tag_name"].as_str()?.to_string();
    let assets = r["assets"].as_array()?;
    let file = |a: &Value| {
        let sha = crate::download::parse_sha256(a["digest"].as_str()?)?;
        Some((a["browser_download_url"].as_str()?.to_string(), a["name"].as_str()?.to_string(), sha))
    };
    let (url, name, sha256) = assets.iter().filter_map(file).find(|(_, n, _)| asset_matches(n, build))?;
    let cudart = if build.starts_with("cuda") {
        Some(assets.iter().filter_map(file).find(|(_, n, _)| cudart_matches(n, &name, &tag))?)
    } else {
        None
    };
    Some(Release { tag, url, name, sha256, cudart })
}

/// Unpacks a llama.cpp archive (.zip or .tar.gz) into `dest`, keeping the executable bits.
fn unpack(archive: &Path, dest: &Path) -> Result<()> {
    let f = std::fs::File::open(archive)?;
    if archive.to_string_lossy().ends_with(".tar.gz") {
        tar::Archive::new(flate2::read::GzDecoder::new(f)).unpack(dest)?;
    } else {
        zip::ZipArchive::new(f)?.extract(dest)?;
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        // (zip entries may carry no Unix mode: make the tools runnable either way)
        if let Some(dir) = find_server_dir(dest) {
            for e in std::fs::read_dir(dir)?.flatten() {
                let n = e.file_name().to_string_lossy().to_string();
                if n.starts_with("llama-") && !n.contains('.') {
                    let _ = std::fs::set_permissions(e.path(), std::fs::Permissions::from_mode(0o755));
                }
            }
        }
    }
    Ok(())
}

/// Replaces the runtime with the newest build (stops the servers first, reloads the model after).
/// Returns the installed version.
pub async fn update(app: &AppHandle) -> Result<String> {
    let state = app.state::<Arc<AppState>>().inner().clone();
    let backend = effective_backend(&state.settings.read().backend);
    let model = state.info.read().model.clone();
    // the runtime folder is replaced, so both servers must release their exe first
    crate::server::stop_all(app).await;
    crate::server::stop_llm(app).await; // resets the status shown in the UI
    ensure(app, &backend, true).await?;
    let version = installed_version(&state, &backend);
    crate::updater::runtime_updated(app, version.clone());
    if let Some(m) = model {
        let app2 = app.clone();
        tauri::async_runtime::spawn(async move {
            let _ = crate::server::start_llm(&app2, &m).await;
        });
    }
    Ok(version.unwrap_or_default())
}

/// Download (or update) the llama.cpp build for the chosen backend.
pub async fn ensure(app: &AppHandle, backend: &str, force: bool) -> Result<PathBuf> {
    let state = app.state::<Arc<AppState>>();
    let backend = &effective_backend(backend);
    let exe = server_exe(&state, backend);
    if exe.exists() && !force {
        return Ok(exe);
    }
    state.log(format!("[runtime] fetching llama.cpp ({backend}) release list"));
    let Release { tag, url, name, sha256, cudart } = latest_release(&state, backend).await?;

    let tmp_zip = state.paths.runtime.join(&name);
    let cancel = Arc::new(AtomicBool::new(false));
    crate::download::download(app, "runtime", &format!("llama.cpp {tag} ({backend})"), &url, &tmp_zip, Some(&sha256), cancel.clone())
        .await
        .context("downloading llama.cpp")?;
    let tmp_cudart = match &cudart {
        Some((url, name, sha)) => {
            let path = state.paths.runtime.join(name);
            crate::download::download(app, "runtime", "NVIDIA CUDA runtime", url, &path, Some(sha), cancel)
                .await
                .context("downloading the CUDA runtime")?;
            Some(path)
        }
        None => None,
    };

    let dir = runtime_dir(&state, backend);
    let staging = state.paths.runtime.join(format!("staging-{backend}"));
    let _ = std::fs::remove_dir_all(&staging);
    std::fs::create_dir_all(&staging)?;
    let root = {
        let (archive, cudart, staging) = (tmp_zip.clone(), tmp_cudart.clone(), staging.clone());
        tokio::task::spawn_blocking(move || unpack_build(&archive, cudart.as_deref(), &staging)).await??
    };
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::rename(&root, &dir).or_else(|_| copy_dir(&root, &dir))?;
    let _ = std::fs::remove_dir_all(&staging);
    let _ = std::fs::remove_file(&tmp_zip);
    if let Some(c) = tmp_cudart {
        let _ = std::fs::remove_file(c);
    }
    std::fs::write(dir.join("version.txt"), &tag)?;
    state.log(format!("[runtime] installed llama.cpp {tag} ({name})"));
    Ok(dir.join(SERVER))
}

/// Unpacks a build (and NVIDIA's runtime libraries, for CUDA) under `staging`; returns the folder that holds
/// llama-server with everything it needs (some archives nest their files in a folder).
fn unpack_build(build: &Path, cudart: Option<&Path>, staging: &Path) -> Result<PathBuf> {
    unpack(build, &staging.join("build"))?;
    let root = find_server_dir(&staging.join("build")).ok_or_else(|| anyhow!("{SERVER} missing from the llama.cpp archive"))?;
    if let Some(c) = cudart {
        unpack(c, &staging.join("cudart"))?;
        copy_files_flat(&staging.join("cudart"), &root)?;
    }
    Ok(root)
}

/// Copies every file under `from` (any depth) straight into `to`, keeping files already there: NVIDIA's
/// libraries must sit next to llama-server (Windows DLL search, Linux LD_LIBRARY_PATH).
fn copy_files_flat(from: &Path, to: &Path) -> std::io::Result<()> {
    for e in std::fs::read_dir(from)?.flatten() {
        let p = e.path();
        if p.is_dir() {
            copy_files_flat(&p, to)?;
        } else {
            let target = to.join(e.file_name());
            if !target.exists() {
                std::fs::copy(&p, target)?;
            }
        }
    }
    Ok(())
}

fn find_server_dir(dir: &std::path::Path) -> Option<PathBuf> {
    if dir.join(SERVER).exists() {
        return Some(dir.to_path_buf());
    }
    for e in std::fs::read_dir(dir).ok()?.flatten() {
        if e.path().is_dir() {
            if let Some(p) = find_server_dir(&e.path()) {
                return Some(p);
            }
        }
    }
    None
}

fn copy_dir(from: &std::path::Path, to: &std::path::Path) -> std::io::Result<()> {
    std::fs::create_dir_all(to)?;
    for e in std::fs::read_dir(from)?.flatten() {
        let t = to.join(e.file_name());
        if e.path().is_dir() {
            copy_dir(&e.path(), &t)?;
        } else {
            std::fs::copy(e.path(), t)?;
        }
    }
    Ok(())
}

/// The backend a fresh install starts with ("auto" in the settings): Metal on macOS, CUDA when an NVIDIA
/// driver new enough for CUDA 12 answers, Vulkan otherwise (AMD, Intel, and no GPU – the Vulkan build falls
/// back to the CPU by itself).
pub async fn auto_backend() -> String {
    if cfg!(target_os = "macos") {
        return "metal".into();
    }
    match nvidia_smi(&[]).await.as_deref().and_then(|s| parse_nvidia_smi(s, "")) {
        Some((driver, _)) if driver >= 12 => "cuda".into(),
        _ => "vulkan".into(),
    }
}

/// What kind of device `llama-server --list-devices` reported, from its name: "software" (a CPU pretending
/// to be a GPU: much slower than the CPU backend), "integrated" (shares system RAM; the MiB it reports are
/// not VRAM) or "discrete".
pub fn device_kind(name: &str) -> &'static str {
    let n = name.to_lowercase();
    if ["llvmpipe", "swiftshader", "softpipe", "lavapipe", "basic render", "microsoft basic"].iter().any(|s| n.contains(s)) {
        return "software";
    }
    let discrete_hint = regex::Regex::new(r"\b(rtx|gtx|quadro|tesla|rx\s?\d|arc\s?(\(tm\)\s?)?[ab]\d{3}|radeon\s?(\(tm\)\s?)?(pro|vii|r9|hd)|instinct|apple)").unwrap();
    if discrete_hint.is_match(&n) {
        return "discrete";
    }
    // Intel UHD / Iris / "Intel(R) Graphics" / Meteor Lake "Arc(TM) Graphics", AMD APUs ("Radeon(TM)
    // Graphics", 680M…8060S), Qualcomm Adreno
    let integrated = regex::Regex::new(r"(intel.*(uhd|iris|hd graphics|\(r\) graphics|arc\(tm\) graphics|arc(\(tm\))?\s?\d{3}[vth]\b))|radeon(\(tm\))?\s*(\d{3,4}[ms]\s*)?graphics|radeon\s?\d{3,4}[ms]\b|adreno|vega \d+ graphics").unwrap();
    if integrated.is_match(&n) {
        return "integrated";
    }
    "discrete"
}

/// One device of `llama-server --list-devices`: name, total memory (bytes) and `device_kind`.
#[derive(Clone, Debug, PartialEq)]
pub struct Device {
    pub name: String,
    pub total: u64,
    pub kind: &'static str,
}

/// Every line `  Vulkan0: AMD Radeon RX 9070 XT (16304 MiB, 15890 MiB free)` of `--list-devices`.
pub fn parse_devices(text: &str) -> Vec<Device> {
    let re = regex::Regex::new(r"(?m)^\s*\w+\d+:\s*(.+?)\s*\((\d+)\s*MiB").unwrap();
    re.captures_iter(text)
        .filter_map(|c| {
            let name = c[1].to_string();
            Some(Device { kind: device_kind(&name), total: c[2].parse::<u64>().ok()? * 1024 * 1024, name })
        })
        .collect()
}

/// The device the recommendations are sized for: the discrete GPU with the most memory, else an integrated
/// one (its memory counted as 0 VRAM: it shares the RAM), else none. Software devices never count.
pub fn best_device(devices: &[Device]) -> Option<Device> {
    let discrete = devices.iter().filter(|d| d.kind == "discrete").max_by_key(|d| d.total);
    discrete.or_else(|| devices.iter().find(|d| d.kind == "integrated")).cloned()
}

/// Asks the runtime which GPUs it sees and stores the result (GPU name, VRAM, integrated or not, and that
/// the check happened) in the status the window shows. False if the runtime could not be asked.
pub async fn detect_hardware(state: &AppState, exe: &std::path::Path) -> bool {
    let mut cmd = tokio::process::Command::new(exe);
    cmd.arg("--list-devices");
    crate::platform::quiet(&mut cmd);
    #[cfg(target_os = "linux")]
    if let Some(dir) = exe.parent() {
        cmd.env("LD_LIBRARY_PATH", dir);
    }
    let Ok(out) = cmd.output().await else { return false };
    let text = format!("{}{}", String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr));
    if !text.contains("Available devices") {
        state.log(format!("[runtime] --list-devices gave no device list: {}", text.trim()));
        return false;
    }
    let devices = parse_devices(&text);
    let best = best_device(&devices);
    state.log(format!(
        "[runtime] devices: {} -> {}",
        if devices.is_empty() { "none".to_string() } else { devices.iter().map(|d| format!("{} ({} MiB, {})", d.name, d.total >> 20, d.kind)).collect::<Vec<_>>().join(", ") },
        best.as_ref().map(|d| d.name.as_str()).unwrap_or("CPU only")
    ));
    let mut i = state.info.write();
    i.hw_checked = true;
    i.gpu_integrated = best.as_ref().is_some_and(|d| d.kind == "integrated");
    i.gpu_name = best.as_ref().map(|d| d.name.clone()).unwrap_or_default();
    i.vram_total = best.filter(|d| d.kind == "discrete").map(|d| d.total).unwrap_or(0);
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Real file names of llama.cpp release b12179: exactly one matches per system and backend.
    #[test]
    fn picks_this_systems_llama_cpp_build() {
        let names = [
            "llama-b12179-bin-win-vulkan-x64.zip",
            "llama-b12179-bin-win-cpu-x64.zip",
            "llama-b12179-bin-win-cpu-arm64.zip",
            "cudart-llama-b12179-bin-ubuntu-cuda-12.8-x64.tar.gz",
            "llama-b12179-bin-macos-arm64.tar.gz",
            "llama-b12179-bin-macos-x64.tar.gz",
            "llama-b12179-bin-ubuntu-arm64.tar.gz",
            "llama-b12179-bin-ubuntu-rocm-10.0-x64.tar.gz",
            "llama-b12179-bin-ubuntu-vulkan-arm64.tar.gz",
            "llama-b12179-bin-ubuntu-vulkan-x64.tar.gz",
            "llama-b12179-bin-ubuntu-x64.tar.gz",
        ];
        let pick = |backend: &str| -> Vec<&str> { names.iter().copied().filter(|n| asset_matches(n, &effective_backend(backend))).collect() };
        if cfg!(windows) {
            assert_eq!(pick("vulkan"), ["llama-b12179-bin-win-vulkan-x64.zip"]);
            assert_eq!(pick("cpu"), ["llama-b12179-bin-win-cpu-x64.zip"]);
        } else if cfg!(target_os = "macos") {
            assert_eq!(effective_backend("vulkan"), "metal");
            assert_eq!(pick("vulkan"), ["llama-b12179-bin-macos-arm64.tar.gz"]);
        } else {
            assert_eq!(pick("vulkan"), ["llama-b12179-bin-ubuntu-vulkan-x64.tar.gz"]);
            assert_eq!(pick("cpu"), ["llama-b12179-bin-ubuntu-x64.tar.gz"]);
            assert_eq!(pick("rocm"), ["llama-b12179-bin-ubuntu-rocm-10.0-x64.tar.gz"]);
        }
    }

    /// Asset list of llama.cpp release b12194 (2026-11-08), the CUDA part and its neighbours.
    const B12194: &[&str] = &[
        "cudart-llama-b12194-bin-ubuntu-cuda-12.8-x64.tar.gz",
        "cudart-llama-b12194-bin-ubuntu-cuda-13.5-arm64.tar.gz",
        "cudart-llama-b12194-bin-ubuntu-cuda-13.5-x64.tar.gz",
        "cudart-llama-bin-win-cuda-12.4-x64.zip",
        "cudart-llama-bin-win-cuda-13.5-arm64.zip",
        "cudart-llama-bin-win-cuda-13.5-x64.zip",
        "llama-b12194-bin-ubuntu-cuda-12.8-x64.tar.gz",
        "llama-b12194-bin-ubuntu-cuda-13.5-arm64.tar.gz",
        "llama-b12194-bin-ubuntu-cuda-13.5-x64.tar.gz",
        "llama-b12194-bin-ubuntu-rocm-10.0-x64.tar.gz",
        "llama-b12194-bin-ubuntu-vulkan-x64.tar.gz",
        "llama-b12194-bin-ubuntu-x64.tar.gz",
        "llama-b12194-bin-win-cpu-x64.zip",
        "llama-b12194-bin-win-cuda-12.4-x64.zip",
        "llama-b12194-bin-win-cuda-13.5-arm64.zip",
        "llama-b12194-bin-win-cuda-13.5-x64.zip",
        "llama-b12194-bin-win-rocm-10.0-x64.zip",
        "llama-b12194-bin-win-vulkan-x64.zip",
    ];

    fn release_json(names: &[&str]) -> Value {
        let assets: Vec<Value> = names.iter().map(|n| serde_json::json!({"name": n, "browser_download_url": format!("https://x/{n}"), "digest": format!("sha256:{}", "ab".repeat(32))})).collect();
        serde_json::json!({"tag_name": "b12194", "assets": assets})
    }

    /// A CUDA build comes with the NVIDIA runtime file of the same flavour; other builds come alone.
    #[test]
    fn cuda_build_comes_with_its_runtime() {
        let r = release_json(B12194);
        let files = |build: &str| pick_release(&r, build).map(|r| (r.name, r.cudart.map(|c| c.1)));
        if cfg!(windows) {
            assert_eq!(files("cuda-12"), Some(("llama-b12194-bin-win-cuda-12.4-x64.zip".into(), Some("cudart-llama-bin-win-cuda-12.4-x64.zip".into()))));
            assert_eq!(files("cuda-13"), Some(("llama-b12194-bin-win-cuda-13.5-x64.zip".into(), Some("cudart-llama-bin-win-cuda-13.5-x64.zip".into()))));
            assert_eq!(files("vulkan"), Some(("llama-b12194-bin-win-vulkan-x64.zip".into(), None)));
        } else if cfg!(target_os = "linux") {
            assert_eq!(files("cuda-12"), Some(("llama-b12194-bin-ubuntu-cuda-12.8-x64.tar.gz".into(), Some("cudart-llama-b12194-bin-ubuntu-cuda-12.8-x64.tar.gz".into()))));
            assert_eq!(files("cuda-13"), Some(("llama-b12194-bin-ubuntu-cuda-13.5-x64.tar.gz".into(), Some("cudart-llama-b12194-bin-ubuntu-cuda-13.5-x64.tar.gz".into()))));
            assert_eq!(files("vulkan"), Some(("llama-b12194-bin-ubuntu-vulkan-x64.tar.gz".into(), None)));
        }
        if !cfg!(target_os = "macos") {
            // a release whose runtime file is missing is skipped, not installed half
            let partial: Vec<&str> = B12194.iter().copied().filter(|n| !n.starts_with("cudart-")).collect();
            assert!(pick_release(&release_json(&partial), "cuda-12").is_none());
            assert!(pick_release(&release_json(&partial), "vulkan").is_some());
            // no published SHA-256: not installed
            let mut unsigned = release_json(B12194);
            for a in unsigned["assets"].as_array_mut().unwrap() {
                a.as_object_mut().unwrap().remove("digest");
            }
            assert!(pick_release(&unsigned, "vulkan").is_none());
        }
    }

    /// Real CUDA 12 install from GitHub's newest release (~650 MB on Windows, ~750 MB on Linux; kept in
    /// `LUMEN_TEST_CUDA_DIR` for reruns): both files are found, unpacked into one folder, NVIDIA's
    /// libraries sit next to llama-server, and llama-server starts. Works without an NVIDIA card.
    /// `LUMEN_TEST_CUDA_DIR=… cargo test live_cuda_install -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn live_cuda_install() {
        use futures_util::StreamExt;
        use tokio::io::AsyncWriteExt;
        let cache = PathBuf::from(std::env::var("LUMEN_TEST_CUDA_DIR").expect("set LUMEN_TEST_CUDA_DIR"));
        std::fs::create_dir_all(&cache).unwrap();
        let rt = tokio::runtime::Runtime::new().unwrap();
        let http = reqwest::Client::builder().user_agent(LUMEN_UA).build().unwrap();
        let rel = rt.block_on(async {
            let list: Value = http.get("https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=5").send().await.unwrap().json().await.unwrap();
            let rel = list.as_array().unwrap().iter().find_map(|r| pick_release(r, "cuda-12")).expect("a release with a CUDA 12 build");
            let mut files = vec![(rel.url.clone(), rel.name.clone())];
            files.extend(rel.cudart.clone().map(|c| (c.0, c.1)));
            for (url, name) in files {
                let path = cache.join(&name);
                if path.exists() {
                    continue;
                }
                println!("downloading {name}");
                let mut out = tokio::fs::File::create(cache.join("partial")).await.unwrap();
                let mut body = http.get(&url).send().await.unwrap().error_for_status().unwrap().bytes_stream();
                while let Some(chunk) = body.next().await {
                    out.write_all(&chunk.unwrap()).await.unwrap();
                }
                out.flush().await.unwrap();
                drop(out);
                std::fs::rename(cache.join("partial"), &path).unwrap();
            }
            rel
        });
        println!("{} + {:?}", rel.name, rel.cudart.as_ref().map(|c| &c.1));
        let staging = cache.join("staging");
        let _ = std::fs::remove_dir_all(&staging);
        let cudart = cache.join(&rel.cudart.as_ref().unwrap().1);
        let root = unpack_build(&cache.join(&rel.name), Some(&cudart), &staging).unwrap();
        let names: Vec<String> = std::fs::read_dir(&root).unwrap().flatten().map(|e| e.file_name().to_string_lossy().into_owned()).collect();
        let has = |p: &str| names.iter().any(|n| n.starts_with(p));
        assert!(names.iter().any(|n| n == SERVER));
        assert!(has("ggml-cuda") || has("libggml-cuda"), "the CUDA backend: {names:?}");
        assert!(has("cudart64_12") || has("libcudart.so.12"), "NVIDIA's runtime next to llama-server: {names:?}");
        assert!(has("cublas64_12") || has("libcublas.so.12"), "cuBLAS next to llama-server: {names:?}");
        let mut cmd = std::process::Command::new(root.join(SERVER));
        cmd.arg("--version");
        if cfg!(target_os = "linux") {
            cmd.env("LD_LIBRARY_PATH", &root);
        }
        let out = cmd.output().unwrap();
        let text = format!("{}{}", String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr));
        println!("{text}");
        assert!(out.status.success() && text.contains("version"), "llama-server starts");
        let _ = std::fs::remove_dir_all(&staging);
    }

    /// The GPU the recommendations are sized for: discrete cards win, integrated GPUs share the RAM (0 VRAM),
    /// software renderers and "(none)" mean the CPU.
    #[test]
    fn picks_the_gpu_that_matters() {
        let out = "ggml_vulkan: Found 2 Vulkan devices:\nAvailable devices:\n  Vulkan0: Intel(R) UHD Graphics 770 (16245 MiB, 16000 MiB free)\n  Vulkan1: AMD Radeon RX 9070 XT (16304 MiB, 15890 MiB free)\n";
        let d = parse_devices(out);
        assert_eq!(d.len(), 2);
        assert_eq!(d[0].kind, "integrated");
        let best = best_device(&d).unwrap();
        assert_eq!((best.name.as_str(), best.total >> 20, best.kind), ("AMD Radeon RX 9070 XT", 16304, "discrete"));
        assert!(best_device(&parse_devices("Available devices:\n  (none)\n")).is_none());
        assert!(best_device(&parse_devices("Available devices:\n  Vulkan0: llvmpipe (LLVM 19.1.1, 256 bits) (32000 MiB, 32000 MiB free)\n")).is_none());
        let igpu = best_device(&parse_devices("Available devices:\n  Vulkan0: AMD Radeon(TM) 780M Graphics (8192 MiB, 8000 MiB free)\n")).unwrap();
        assert_eq!(igpu.kind, "integrated");
        let two = parse_devices("Available devices:\n  CUDA0: NVIDIA GeForce RTX 3060 (12288 MiB, 11000 MiB free)\n  CUDA1: NVIDIA GeForce RTX 4090 (24564 MiB, 23000 MiB free)\n");
        assert_eq!(best_device(&two).unwrap().name, "NVIDIA GeForce RTX 4090", "the biggest card");
        // the RAM the recommendations are sized against is read for real
        assert!(crate::platform::total_memory_mb().unwrap_or(0) > 1024);
        for (name, kind) in [
            ("NVIDIA GeForce GTX 1650", "discrete"),
            ("NVIDIA GeForce MX450", "discrete"),
            ("Intel(R) Arc(TM) A770 Graphics", "discrete"),
            ("Intel(R) Arc(TM) B580 Graphics", "discrete"),
            ("Intel(R) Arc(TM) Graphics", "integrated"),
            ("Intel(R) Arc(TM) 140V GPU (16GB)", "integrated"),
            ("Intel(R) Iris(R) Xe Graphics", "integrated"),
            ("Intel(R) Graphics", "integrated"),
            ("AMD Radeon(TM) Graphics", "integrated"),
            ("AMD Radeon 8060S Graphics", "integrated"),
            ("AMD Radeon RX 7600", "discrete"),
            ("AMD Radeon PRO W7900", "discrete"),
            ("Apple M2 Pro", "discrete"),
            ("Microsoft Basic Render Driver", "software"),
        ] {
            assert_eq!(device_kind(name), kind, "{name}");
        }
    }

    #[test]
    fn cuda_flavour_follows_driver_and_gpu() {
        let summary = "| NVIDIA-SMI 581.42                 Driver Version: 581.42         CUDA Version: 13.0     |";
        assert_eq!(parse_nvidia_smi(summary, "8.9\n"), Some((13, 8.9)));
        assert_eq!(parse_nvidia_smi(summary, "8.9\n6.1\n"), Some((13, 6.1)), "the oldest card decides");
        assert_eq!(parse_nvidia_smi("no header here", "8.9"), None);
        assert_eq!(cuda_major(13, 8.9), 13); // RTX 40 on a new driver
        assert_eq!(cuda_major(13, 12.0), 13); // RTX 50
        assert_eq!(cuda_major(13, 6.1), 12); // GTX 10xx: CUDA 13 dropped Pascal
        assert_eq!(cuda_major(12, 8.9), 12); // driver too old for 13
        assert_eq!(cuda_major(13, 0.0), 12); // capability unknown: the safe choice
    }
}
