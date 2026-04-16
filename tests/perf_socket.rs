/// Performance tests using the CLI control socket.
///
/// Each test launches the release binary, waits for the socket, exercises a
/// specific scenario, and prints timing results.  Assertions are generous
/// upper-bounds; the real value is the printed numbers.
///
/// Run with:
///   cargo build --release
///   cargo test --test perf_socket -- --nocapture --test-threads=1
///
/// Tests MUST run single-threaded (--test-threads=1) because they all share
/// /tmp/lightningview.sock.
use std::{
    collections::HashMap,
    io::{Read, Write},
    os::unix::net::UnixStream,
    path::PathBuf,
    process::{Child, Command},
    sync::Mutex,
    thread,
    time::{Duration, Instant},
};

// ── Constants ─────────────────────────────────────────────────────────────────

const SOCKET: &str = "/tmp/lightningview.sock";
const NEF_DIR: &str = "/Users/bjorn/Photography/PhotographySessions/Shootings/.xsessions/2026 - 02 FEB - Natalia/Capture";
const BINARY:  &str = "target/release/lightningview";

/// Serialise all tests so they don't race on the socket.
static LOCK: Mutex<()> = Mutex::new(());

/// Acquire the global lock, recovering from a poisoned state (previous test panic).
fn lock() -> std::sync::MutexGuard<'static, ()> {
    LOCK.lock().unwrap_or_else(|p| p.into_inner())
}

// ── App harness ───────────────────────────────────────────────────────────────

struct App {
    child: Child,
}

impl App {
    /// Launch the release binary in windowed mode and wait until the socket
    /// accepts connections.
    fn launch(nef_path: &str) -> Self {
        let _ = std::fs::remove_file(SOCKET);

        let child = Command::new(BINARY)
            .args(["/windowed", nef_path])
            .spawn()
            .unwrap_or_else(|_| panic!(
                "failed to launch {BINARY} — did you run `cargo build --release`?"
            ));

        let deadline = Instant::now() + Duration::from_secs(20);
        loop {
            assert!(Instant::now() < deadline, "app did not bind socket within 20s");
            thread::sleep(Duration::from_millis(50));
            if UnixStream::connect(SOCKET).is_ok() {
                break;
            }
        }
        // One extra settle tick before issuing commands.
        thread::sleep(Duration::from_millis(100));
        App { child }
    }

    /// Send one command line and return the trimmed response.
    fn cmd(&self, command: &str) -> String {
        let mut stream = UnixStream::connect(SOCKET).expect("connect");
        stream.set_read_timeout(Some(Duration::from_secs(10))).ok();
        writeln!(stream, "{command}").expect("write");
        stream.shutdown(std::net::Shutdown::Write).ok();
        let mut resp = String::new();
        stream.read_to_string(&mut resp).ok();
        resp.trim().to_string()
    }

    /// Parse `key=value` tokens from a status line into a map.
    fn parse(line: &str) -> HashMap<String, String> {
        line.split_whitespace()
            .filter_map(|tok| tok.split_once('='))
            .map(|(k, v)| (k.to_owned(), v.to_owned()))
            .collect()
    }

    /// Poll `status` until at least `min_loaded` non-thumbnail images are in
    /// cache.  Does NOT wait for `pending == 0`: background warming of the
    /// whole folder keeps pending non-zero for a long time.
    fn wait_loaded(&self, min_loaded: usize, timeout: Duration) -> Duration {
        let start = Instant::now();
        let mut last_print = start;
        loop {
            thread::sleep(Duration::from_millis(50));
            let elapsed = start.elapsed();
            let status = self.cmd("status");
            // Print a heartbeat every 3 s so we can see what is happening.
            if elapsed.saturating_sub(last_print.elapsed()) == Duration::ZERO
                && last_print.elapsed() >= Duration::from_secs(3)
            {
                eprintln!("  [wait_loaded({min_loaded}) @{:.1}s] {status}", elapsed.as_secs_f32());
                last_print = Instant::now();
            }
            let m = Self::parse(&status);
            let loaded: usize = m.get("loaded").and_then(|v| v.parse().ok()).unwrap_or(0);
            if loaded >= min_loaded {
                return elapsed;
            }
            assert!(
                elapsed < timeout,
                "wait_loaded({min_loaded}) timed out after {timeout:?} — last status: {status}",
            );
        }
    }

    /// Poll `status` until the current image is a full (non-thumbnail) preview.
    /// Returns elapsed time.  Panics on timeout.
    fn wait_current_loaded(&self, timeout: Duration) -> Duration {
        let start = Instant::now();
        loop {
            thread::sleep(Duration::from_millis(50));
            let elapsed = start.elapsed();
            let status = self.cmd("status");
            let m = Self::parse(&status);
            let cur_loaded: u8 = m.get("cur_loaded").and_then(|v| v.parse().ok()).unwrap_or(0);
            if cur_loaded == 1 {
                return elapsed;
            }
            assert!(elapsed < timeout,
                "wait_current_loaded timed out — last status: {status}");
        }
    }

    /// Return `(sorted_nef_paths)` from the capture directory.
    fn all_nefs() -> Vec<PathBuf> {
        let mut files: Vec<PathBuf> = std::fs::read_dir(NEF_DIR)
            .expect("read dir")
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| matches!(p.extension().and_then(|e| e.to_str()), Some("NEF") | Some("nef")))
            .collect();
        files.sort_by_key(|p| p.to_string_lossy().to_lowercase());
        files
    }

    /// Poll `status` until eye detection is no longer active.
    /// Returns elapsed time.  Panics on timeout.
    fn wait_eye_done(&self, timeout: Duration) -> Duration {
        let start = Instant::now();
        loop {
            assert!(start.elapsed() < timeout,
                "wait_eye_done timed out after {timeout:?}");
            thread::sleep(Duration::from_millis(50));
            let m = Self::parse(&self.cmd("status"));
            let active = m.get("eye_active").map(String::as_str).unwrap_or("-");
            let queue:  usize = m.get("eye_queue").and_then(|v| v.parse().ok()).unwrap_or(1);
            if active == "-" && queue == 0 {
                return start.elapsed();
            }
        }
    }

    /// Return the most-recent decode-time entries from the `perf` command as
    /// a vec of `(order_pos, filename_stem, decode_ms)`.
    fn perf_entries(&self) -> Vec<(usize, String, u64)> {
        self.cmd("perf")
            .lines()
            .filter(|l| l.trim_start().starts_with("pos="))
            .filter_map(|l| {
                // format: "  pos=  41 DSC_0042.NEF    187ms"
                let l = l.trim();
                let pos_part = l.strip_prefix("pos=")?;
                let mut iter = pos_part.splitn(3, char::is_whitespace);
                let pos:  usize = iter.next()?.trim().parse().ok()?;
                let file: &str  = iter.next()?.trim();
                let ms_str = iter.next()?.trim().trim_end_matches("ms");
                let ms: u64 = ms_str.parse().ok()?;
                Some((pos, file.to_owned(), ms))
            })
            .collect()
    }
}

impl Drop for App {
    fn drop(&mut self) {
        self.child.kill().ok();
        self.child.wait().ok();
        let _ = std::fs::remove_file(SOCKET);
    }
}

// ── Directory helper ──────────────────────────────────────────────────────────

fn first_nef() -> PathBuf {
    std::fs::read_dir(NEF_DIR)
        .unwrap_or_else(|e| panic!("cannot read {NEF_DIR}: {e}"))
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| matches!(
            p.extension().and_then(|e| e.to_str()),
            Some("NEF") | Some("nef")
        ))
        .min_by_key(|p| p.to_string_lossy().to_lowercase())
        .expect("no NEF files in directory")
}

// ── Test 1: First-image load time ─────────────────────────────────────────────

#[test]
fn perf_1_first_image_load() {
    let _guard = lock();

    let path = first_nef();
    println!("\n══ TEST 1: first-image load ══");
    println!("  file: {}", path.display());

    // Measure wall time from launch until the image is in cache.
    let wall_start = Instant::now();
    let app = App::launch(path.to_str().unwrap());
    let wait = app.wait_loaded(1, Duration::from_secs(30));
    let wall_total = wall_start.elapsed();

    println!("  wall time (launch → loaded):  {:>6}ms", wall_total.as_millis());
    println!("  wait-for-loaded portion:       {:>6}ms", wait.as_millis());
    println!("  app-startup overhead:          {:>6}ms",
        wall_total.as_millis().saturating_sub(wait.as_millis()));

    // Decode time reported by the app itself.
    let entries = app.perf_entries();
    if let Some((pos, file, ms)) = entries.first() {
        println!("  first decode: pos={pos}  file={file}  decode_time={ms}ms");
    }

    let status = app.cmd("status");
    println!("  status: {status}");

    // Generous upper bound — must load within 30 s on any drive.
    assert!(wall_total < Duration::from_secs(30), "first image took too long");
}

// ── Test 2: Compare mode — add two images ────────────────────────────────────

#[test]
fn perf_2_compare_add_two_images() {
    let _guard = lock();

    let path = first_nef();
    println!("\n══ TEST 2: add two images to compare view ══");

    let app = App::launch(path.to_str().unwrap());
    println!("  waiting for first image …");
    app.wait_loaded(1, Duration::from_secs(30));
    println!("  status: {}", app.cmd("status"));

    // Add second image.
    let t2 = Instant::now();
    app.cmd("compare_add");
    let d2 = app.wait_loaded(2, Duration::from_secs(30));
    println!("  2nd image loaded in: {:>6}ms  (wall {:>6}ms)",
        d2.as_millis(), t2.elapsed().as_millis());
    println!("  status: {}", app.cmd("status"));

    // Add third image.
    let t3 = Instant::now();
    app.cmd("compare_add");
    let d3 = app.wait_loaded(3, Duration::from_secs(30));
    println!("  3rd image loaded in: {:>6}ms  (wall {:>6}ms)",
        d3.as_millis(), t3.elapsed().as_millis());
    println!("  status: {}", app.cmd("status"));

    let entries = app.perf_entries();
    println!("  perf — last {} entries:", entries.len());
    for (pos, file, ms) in &entries {
        println!("    pos={pos:4}  {file:<44}  {ms}ms");
    }

    assert!(t3.elapsed() < Duration::from_secs(30));
}

// ── Test 3: Browse 10 frames ──────────────────────────────────────────────────
//
// Starts at image #20 (past the 10-image prefetch window seeded from image #0),
// so the first several navigations hit uncached images and show real latency.

#[test]
fn perf_3_browse_10_frames() {
    let _guard = lock();

    let nefs = App::all_nefs();
    assert!(nefs.len() >= 25, "need at least 25 NEF files for this test");

    let app = App::launch(nefs[0].to_str().unwrap());
    app.wait_loaded(1, Duration::from_secs(30));

    // Jump to image #20 — cold territory beyond the prefetch window.
    let start_path = nefs[20].to_str().unwrap().to_owned();
    println!("\n══ TEST 3: browse 10 frames starting at image #20 ══");
    println!("  jumping to: {}", nefs[20].file_name().unwrap().to_string_lossy());
    app.cmd(&format!("open {start_path}"));
    let jump_ms = app.wait_current_loaded(Duration::from_secs(30)).as_millis();
    println!("  jump load: {:>6}ms   {}", jump_ms, app.cmd("status"));

    // Navigate forward 10 times, measuring per-frame load time.
    let mut frame_times: Vec<u64> = Vec::with_capacity(10);
    let total_start = Instant::now();

    for frame in 1..=10 {
        app.cmd("next");
        let ms = app.wait_current_loaded(Duration::from_secs(30)).as_millis();
        frame_times.push(ms as u64);
        println!("  frame {frame:2}: {:>6}ms   {}", ms, app.cmd("status"));
    }

    let total_ms = total_start.elapsed().as_millis();
    let avg_ms = frame_times.iter().sum::<u64>() / frame_times.len() as u64;
    let max_ms = frame_times.iter().copied().max().unwrap_or(0);
    let min_ms = frame_times.iter().copied().min().unwrap_or(0);
    println!("  ── summary ──");
    println!("  total: {total_ms}ms  avg/frame: {avg_ms}ms  min: {min_ms}ms  max: {max_ms}ms");

    let entries = app.perf_entries();
    println!("  perf — last {} decode entries:", entries.len().min(12));
    for (pos, file, ms) in entries.iter().rev().take(12) {
        println!("    pos={pos:4}  {file:<44}  {ms}ms");
    }

    assert!(Duration::from_millis(total_ms as u64) < Duration::from_secs(300));
}

// ── Test 4: Eye detection — cold vs warm ─────────────────────────────────────

#[test]
fn perf_4_eye_detection_cold_and_warm() {
    let _guard = lock();

    let path = first_nef();
    println!("\n══ TEST 4: eye detection (cold + warm) ══");
    println!("  file: {}", path.display());

    let app = App::launch(path.to_str().unwrap());
    println!("  waiting for first image to fully load …");
    app.wait_loaded(1, Duration::from_secs(30));
    println!("  status before eye: {}", app.cmd("status"));

    // ── Cold run ─────────────────────────────────────────────────────────────
    // Model is pre-warmed in RAM (prewarm_model at startup) but detection
    // itself has never run on this image.
    let t_cold = Instant::now();
    app.cmd("eye");
    let cold_ms = app.wait_eye_done(Duration::from_secs(60)).as_millis();
    let cold_wall = t_cold.elapsed().as_millis();

    let status_after = app.cmd("status");
    println!("  cold run:  detect={cold_ms}ms  wall={cold_wall}ms");
    println!("  status after: {status_after}");

    // ── Warm run ─────────────────────────────────────────────────────────────
    // Result is now cached; cycle_eye_focus() should return instantly.
    let t_warm = Instant::now();
    app.cmd("eye");  // cycles through cached results — no detection thread spawned
    let warm_wall = t_warm.elapsed().as_millis();
    // eye_active should remain "-" since no detection was needed.
    let status_warm = app.cmd("status");
    println!("  warm run:  wall={warm_wall}ms  (should be <10ms from cache)");
    println!("  status: {status_warm}");

    assert!(cold_ms   < 60_000, "cold detection exceeded 60s");
    assert!(warm_wall <    500, "warm (cached) eye cycle should be instant");
}
