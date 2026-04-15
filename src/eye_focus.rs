/// Eye-focus helpers: face detection and zoom-offset math.
///
/// # Face model
/// Detection requires `seeta_fd_frontal_v1.0.bin` in one of these locations
/// (checked in order):
///   1. `<project_root>/assets/seeta_fd_frontal_v1.0.bin`  (development)
///   2. Next to the running executable                      (installed release)
///   3. `~/.config/lightningview/seeta_fd_frontal_v1.0.bin` (user config)
///
/// If no model is found, `detect_eye_positions` returns an empty `Vec` and logs
/// a `warn!` message with download instructions.
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
    // 1. Development: assets/ next to Cargo.toml
    let dev = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("assets")
        .join(MODEL_FILENAME);
    if dev.exists() {
        return Some(dev);
    }

    // 2. Release: same directory as the running executable
    if let Ok(exe) = std::env::current_exe() {
        let p = exe.parent().unwrap_or(Path::new(".")).join(MODEL_FILENAME);
        if p.exists() {
            return Some(p);
        }
    }

    // 3. User config directory
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

// ── image conversion ─────────────────────────────────────────────────────────

/// Downscale `image` to at most `max_side` on the longest dimension, converting
/// to 8-bit grayscale.  Returns `(bytes, width, height, scale_factor)`.
fn to_gray_scaled(image: &ColorImage, max_side: u32) -> (Vec<u8>, u32, u32, f32) {
    let w = image.width() as u32;
    let h = image.height() as u32;
    let scale = (max_side as f32 / w.max(h) as f32).min(1.0);
    let dw = ((w as f32 * scale).round() as u32).max(1);
    let dh = ((h as f32 * scale).round() as u32).max(1);

    let mut out = vec![0u8; (dw * dh) as usize];
    for y in 0..dh {
        for x in 0..dw {
            let sx = ((x as f32 / scale) as u32).min(w - 1);
            let sy = ((y as f32 / scale) as u32).min(h - 1);
            let [r, g, b, _] = image.pixels[(sy * w + sx) as usize].to_array();
            // ITU-R BT.601 luminance
            out[(y * dw + x) as usize] =
                (r as f32 * 0.299 + g as f32 * 0.587 + b as f32 * 0.114) as u8;
        }
    }
    (out, dw, dh, scale)
}

// ── public API ────────────────────────────────────────────────────────────────

/// Detect eye positions in `image`, returning pixel coordinates in **full-res**
/// image space.
///
/// Two eye positions (left, right) are estimated for each detected face from its
/// bounding box.  Returns an empty `Vec` when no faces are found or when the
/// face-detection model is unavailable.
pub fn detect_eye_positions(image: &ColorImage) -> Vec<Pos2> {
    let Some(path) = model_path() else {
        log::warn!(
            "Eye detection disabled: {} not found. \
             Place it in assets/ or run: \
             mkdir assets && curl -sSL -o assets/{0} \
             https://github.com/atomashpolskiy/rustface/raw/master/model/{0}",
            MODEL_FILENAME
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
    detector.set_min_face_size(20);
    detector.set_score_thresh(2.0);
    detector.set_pyramid_scale_factor(0.8);
    detector.set_slide_window_step(4, 4);

    let (gray, dw, dh, scale) = to_gray_scaled(image, 640);
    let img_data = rustface::ImageData::new(&gray, dw, dh);
    let faces = detector.detect(&img_data);

    let inv = 1.0 / scale;
    let mut eyes = Vec::with_capacity(faces.len() * 2);
    for face in &faces {
        let b = face.bbox();
        let fx = b.x() as f32 * inv;
        let fy = b.y() as f32 * inv;
        let fw = b.width() as f32 * inv;
        let fh = b.height() as f32 * inv;
        // Left eye: ~30 % from left edge, ~37 % from top of bounding box
        eyes.push(Pos2::new(fx + fw * 0.30, fy + fh * 0.37));
        // Right eye: ~70 % from left edge, ~37 % from top of bounding box
        eyes.push(Pos2::new(fx + fw * 0.70, fy + fh * 0.37));
    }
    eyes
}

/// Return the `offset` that centres `eye_pos` on a viewport of `view_size`
/// at the given `zoom` level.
///
/// The image is drawn at `available_rect.min + offset`; each image pixel
/// occupies `zoom` screen pixels.  This is a pure function with no side effects.
pub fn eye_zoom_offset(eye_pos: Pos2, zoom: f32, view_size: Vec2) -> Vec2 {
    view_size / 2.0 - eye_pos.to_vec2() * zoom
}
