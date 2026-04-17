/// Eye-focus helpers: face detection and zoom-offset math.
///
/// Detection is split into two steps so the UI thread only does the cheap part:
///
///   1. `prepare_gray`      — downscale the full-res ColorImage to a small
///                            grayscale buffer (~5 ms, runs on UI thread).
///   2. `detect_from_gray`  — load the model and run the cascade on that buffer
///                            (~100–400 ms, must run on a background thread).
///
/// # Face model
/// Requires `seeta_fd_frontal_v1.0.bin` in one of these locations (in order):
///   1. `<project_root>/assets/seeta_fd_frontal_v1.0.bin`  (development)
///   2. Next to the running executable                      (installed release)
///   3. `~/.config/lightningview/seeta_fd_frontal_v1.0.bin` (user config)
use egui::{ColorImage, Pos2, Vec2};
use image::GenericImageView;
use std::{
    io::Cursor,
    path::{Path, PathBuf},
    sync::OnceLock,
};

const MODEL_FILENAME: &str = "seeta_fd_frontal_v1.0.bin";

/// Zoom level (image pixels → screen pixels) applied when centring on an eye.
pub const EYE_ZOOM: f32 = 4.0;

// ── model discovery ──────────────────────────────────────────────────────────

fn model_path() -> Option<PathBuf> {
    let dev = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("assets")
        .join(MODEL_FILENAME);
    if dev.exists() {
        return Some(dev);
    }

    if let Ok(exe) = std::env::current_exe() {
        let p = exe.parent().unwrap_or(Path::new(".")).join(MODEL_FILENAME);
        if p.exists() {
            return Some(p);
        }
    }

    #[cfg(target_os = "windows")]
    if let Ok(base) = std::env::var("APPDATA") {
        let p = PathBuf::from(base).join("lightningview").join(MODEL_FILENAME);
        if p.exists() {
            return Some(p);
        }
    }
    #[cfg(not(target_os = "windows"))]
    if let Ok(home) = std::env::var("HOME") {
        let p = PathBuf::from(home)
            .join(".config")
            .join("lightningview")
            .join(MODEL_FILENAME);
        if p.exists() {
            return Some(p);
        }
    }

    None
}

// ── step 1: UI thread ─────────────────────────────────────────────────────────

/// Downscale `image` to at most 640 px on the longest side, converting to
/// 8-bit grayscale.  Returns `(gray_bytes, width, height, scale_inv)`.
///
/// Iterates only the *output* pixels so it is O(640²) ≈ 273 K ops regardless
/// of source resolution — safe to call on the UI thread.
pub fn prepare_gray(image: &ColorImage) -> (Vec<u8>, u32, u32, f32) {
    const MAX_SIDE: u32 = 640;
    let w = image.width() as u32;
    let h = image.height() as u32;
    let scale = (MAX_SIDE as f32 / w.max(h) as f32).min(1.0);
    let dw = ((w as f32 * scale).round() as u32).max(1);
    let dh = ((h as f32 * scale).round() as u32).max(1);

    let mut out = vec![0u8; (dw * dh) as usize];
    for y in 0..dh {
        for x in 0..dw {
            let sx = ((x as f32 / scale) as u32).min(w - 1);
            let sy = ((y as f32 / scale) as u32).min(h - 1);
            let [r, g, b, _] = image.pixels[(sy * w + sx) as usize].to_array();
            out[(y * dw + x) as usize] =
                (r as f32 * 0.299 + g as f32 * 0.587 + b as f32 * 0.114) as u8;
        }
    }
    (out, dw, dh, 1.0 / scale)
}

/// Decode a JPEG byte slice and produce a grayscale buffer ready for detection.
/// Equivalent to decoding the JPEG and calling `prepare_gray`, but avoids
/// allocating an intermediate `ColorImage`.  Returns `None` if the JPEG is
/// unreadable.
pub fn prepare_gray_from_jpeg(jpeg_bytes: &[u8]) -> Option<(Vec<u8>, u32, u32, f32)> {
    let img = image::load_from_memory_with_format(jpeg_bytes, image::ImageFormat::Jpeg).ok()?;
    let (iw, ih) = img.dimensions();
    let rgb = img.to_rgb8();

    const MAX_SIDE: u32 = 640;
    let scale = (MAX_SIDE as f32 / iw.max(ih) as f32).min(1.0);
    let dw = ((iw as f32 * scale).round() as u32).max(1);
    let dh = ((ih as f32 * scale).round() as u32).max(1);

    let mut out = vec![0u8; (dw * dh) as usize];
    for y in 0..dh {
        for x in 0..dw {
            let sx = ((x as f32 / scale) as u32).min(iw - 1);
            let sy = ((y as f32 / scale) as u32).min(ih - 1);
            let [r, g, b] = rgb.get_pixel(sx, sy).0;
            out[(y * dw + x) as usize] =
                (r as f32 * 0.299 + g as f32 * 0.587 + b as f32 * 0.114) as u8;
        }
    }
    Some((out, dw, dh, 1.0 / scale))
}

// ── model cache ───────────────────────────────────────────────────────────────

/// Raw bytes of the face-detection model, loaded once and kept in RAM.
/// Avoids re-reading the 1.2 MB file on every detection call (particularly
/// important when the disk is busy with background prefetch I/O).
static MODEL_CACHE: OnceLock<Vec<u8>> = OnceLock::new();

fn cached_model_bytes() -> Option<&'static Vec<u8>> {
    if let Some(bytes) = MODEL_CACHE.get() {
        return Some(bytes);
    }
    let path = model_path()?;
    let bytes = std::fs::read(&path)
        .map_err(|e| log::warn!("Failed to read face model: {}", e))
        .ok()?;
    log::info!("Face model loaded into cache ({} bytes)", bytes.len());
    // OnceLock::set may lose the race if another thread loaded first — that's fine.
    let _ = MODEL_CACHE.set(bytes);
    MODEL_CACHE.get()
}

// ── step 2: background thread ─────────────────────────────────────────────────

/// Run face detection on the downscaled grayscale buffer, then localise each
/// eye precisely by searching for the darkest compact region (iris/pupil) inside
/// the upper half of each detected face bounding box.
///
/// Coordinate pipeline:
///   1. Rustface operates in the small (640 px) buffer — face bbox in small coords.
///   2. `find_darkest_patch` searches the same small buffer for the actual iris.
///   3. All small-buffer positions are multiplied by `scale_inv` to produce
///      coordinates in the full-resolution display image.
///
/// This avoids the fixed-percentage heuristic (which was frequently off by
/// hundreds of pixels at the display resolution) and instead reads the actual
/// pixel values to locate each eye.
pub fn detect_from_gray(gray: Vec<u8>, w: u32, h: u32, scale_inv: f32) -> Vec<Pos2> {
    let model_bytes = match cached_model_bytes() {
        Some(b) => b,
        None => {
            log::warn!(
                "Eye detection disabled: {MODEL_FILENAME} not found. \
                 To enable: mkdir assets && curl -sSL -o assets/{MODEL_FILENAME} \
                 https://github.com/atomashpolskiy/rustface/raw/master/model/{MODEL_FILENAME}"
            );
            return vec![];
        }
    };

    let model = match rustface::read_model(Cursor::new(model_bytes)) {
        Ok(m) => m,
        Err(e) => {
            log::warn!("Failed to parse face detection model: {}", e);
            return vec![];
        }
    };

    let mut detector = rustface::create_detector_with_model(model);
    detector.set_min_face_size(40);
    detector.set_score_thresh(2.0);
    detector.set_pyramid_scale_factor(0.8);
    detector.set_slide_window_step(4, 4);

    let img_data = rustface::ImageData::new(&gray, w, h);
    let faces = detector.detect(&img_data);

    let mut eyes = Vec::with_capacity(faces.len() * 2);
    for face in &faces {
        let b = face.bbox();
        // All coordinates in the small (640 px) buffer until the final scale.
        let bx = b.x().max(0) as u32;
        let by = b.y().max(0) as u32;
        let bw = b.width() as u32;
        let bh = b.height() as u32;

        // Search region: upper 55 % of the face bbox (eyes are never in the
        // lower half). Skip the top 10 % to avoid forehead padding artifacts.
        let y0 = by + bh / 10;
        let y1 = (by + bh * 55 / 100).min(h);
        let mid = bx + bw / 2;

        let left  = find_darkest_patch(&gray, w, bx,  y0, mid,        y1);
        let right = find_darkest_patch(&gray, w, mid, y0, (bx + bw).min(w), y1);

        // Scale from detection space → display space.
        eyes.push(Pos2::new(left.0  as f32 * scale_inv, left.1  as f32 * scale_inv));
        eyes.push(Pos2::new(right.0 as f32 * scale_inv, right.1 as f32 * scale_inv));
    }
    log::info!("Eye detection: found {} face(s), {} eye position(s)", faces.len(), eyes.len());
    eyes
}

/// Find the (x, y) of the darkest 7×7 patch inside the rectangle
/// [x0, x1) × [y0, y1) of the grayscale buffer.
///
/// A 7×7 window average suppresses single-pixel noise (specular highlights,
/// JPEG artefacts) while remaining sensitive enough to locate the iris, whose
/// radius is typically 4–10 pixels at 640 px scale.
///
/// Complexity: O((x1-x0) × (y1-y0)) — for a typical 100×60 px eye search
/// region this is < 10 000 operations.
fn find_darkest_patch(gray: &[u8], stride: u32, x0: u32, y0: u32, x1: u32, y1: u32) -> (u32, u32) {
    const R: u32 = 3; // half-window; 7×7 total
    let cx_fallback = (x0 + x1) / 2;
    let cy_fallback = (y0 + y1) / 2;

    if x1 <= x0 + 2 * R || y1 <= y0 + 2 * R {
        return (cx_fallback, cy_fallback);
    }

    let mut best: u64 = u64::MAX;
    let mut best_x = cx_fallback;
    let mut best_y = cy_fallback;

    for cy in (y0 + R)..(y1 - R) {
        for cx in (x0 + R)..(x1 - R) {
            let mut sum: u64 = 0;
            for dy in 0..(2 * R + 1) {
                let row = ((cy + dy - R) * stride) as usize;
                for dx in 0..(2 * R + 1) {
                    let col = (cx + dx - R) as usize;
                    sum += gray[row + col] as u64;
                }
            }
            if sum < best {
                best = sum;
                best_x = cx;
                best_y = cy;
            }
        }
    }
    (best_x, best_y)
}

/// Load the face-detection model into the in-memory cache without running any
/// detection.  Call this once at startup in a background thread so the first
/// actual detection call doesn't pay the disk-I/O cost.
pub fn prewarm_model() {
    cached_model_bytes();
}

// ── convenience wrapper (used by tests) ──────────────────────────────────────

/// Detect eye positions in a `ColorImage` in one call.
/// Combines `prepare_gray` + `detect_from_gray` — **do not call on the UI thread**.
pub fn detect_eye_positions(image: &ColorImage) -> Vec<Pos2> {
    let (gray, w, h, scale_inv) = prepare_gray(image);
    detect_from_gray(gray, w, h, scale_inv)
}

// ── zoom math ─────────────────────────────────────────────────────────────────

/// Return the `offset` that centres `eye_pos` on a viewport of `view_size`
/// at the given `zoom` level.
///
/// The image is drawn at `available_rect.min + offset`; each image pixel
/// occupies `zoom` screen pixels.  Pure function, no side effects.
pub fn eye_zoom_offset(eye_pos: Pos2, zoom: f32, view_size: Vec2) -> Vec2 {
    view_size / 2.0 - eye_pos.to_vec2() * zoom
}
