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
use std::{
    fs::File,
    io::BufReader,
    path::{Path, PathBuf},
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

// ── step 2: background thread ─────────────────────────────────────────────────

/// Run face detection on a pre-scaled grayscale buffer and return estimated
/// eye positions in **full-resolution** image coordinates.
///
/// `scale_inv` is the reciprocal of the scale that was used to produce `gray`
/// (returned by `prepare_gray`).
///
/// This loads the face model from disk on every call — it must run on a
/// background thread, never on the UI thread.
pub fn detect_from_gray(gray: Vec<u8>, w: u32, h: u32, scale_inv: f32) -> Vec<Pos2> {
    let Some(path) = model_path() else {
        log::warn!(
            "Eye detection disabled: {MODEL_FILENAME} not found. \
             To enable: mkdir assets && curl -sSL -o assets/{MODEL_FILENAME} \
             https://github.com/atomashpolskiy/rustface/raw/master/model/{MODEL_FILENAME}"
        );
        return vec![];
    };

    let file = match File::open(&path) {
        Ok(f) => f,
        Err(e) => {
            log::warn!("Failed to open face detection model at {}: {}", path.display(), e);
            return vec![];
        }
    };
    let model = match rustface::read_model(BufReader::new(file)) {
        Ok(m) => m,
        Err(e) => {
            log::warn!("Failed to parse face detection model: {}", e);
            return vec![];
        }
    };

    let mut detector = rustface::create_detector_with_model(model);
    // min_face_size relative to a 640 px image: 80 px ≈ face 1/8 of frame width —
    // appropriate for portraits. Keeps the pyramid to ~3 levels and detection fast.
    detector.set_min_face_size(80);
    detector.set_score_thresh(2.0);
    detector.set_pyramid_scale_factor(0.85);
    detector.set_slide_window_step(4, 4);

    let img_data = rustface::ImageData::new(&gray, w, h);
    let faces = detector.detect(&img_data);

    let mut eyes = Vec::with_capacity(faces.len() * 2);
    for face in &faces {
        let b = face.bbox();
        let fx = b.x() as f32 * scale_inv;
        let fy = b.y() as f32 * scale_inv;
        let fw = b.width() as f32 * scale_inv;
        let fh = b.height() as f32 * scale_inv;
        // Left eye: ~30 % from left, ~37 % from top of bounding box
        eyes.push(Pos2::new(fx + fw * 0.30, fy + fh * 0.37));
        // Right eye: ~70 % from left, ~37 % from top of bounding box
        eyes.push(Pos2::new(fx + fw * 0.70, fy + fh * 0.37));
    }
    log::info!("Eye detection: found {} face(s), {} eye position(s)", faces.len(), eyes.len());
    eyes
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
