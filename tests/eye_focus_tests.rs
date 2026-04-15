/// Tests for the eye_focus module.
///
/// These cover the pure math (no ML model required) and the detection
/// edge-case of an image that contains no faces.
use egui::{Color32, ColorImage, Pos2, Vec2};
use lightningview::eye_focus::{detect_eye_positions, eye_zoom_offset};

// ── eye_zoom_offset ──────────────────────────────────────────────────────────

/// Eye at the image origin: offset must equal half the view size (centres 0,0).
#[test]
fn eye_at_origin_offset_is_half_view() {
    let offset = eye_zoom_offset(Pos2::ZERO, 4.0, Vec2::new(800.0, 600.0));
    assert_eq!(offset, Vec2::new(400.0, 300.0));
}

/// General case: offset = view/2 - eye * zoom.
#[test]
fn eye_offset_general_case() {
    // eye=(100,80), zoom=4, view=800x600
    // expected: (400-400, 300-320) = (0, -20)
    let offset = eye_zoom_offset(Pos2::new(100.0, 80.0), 4.0, Vec2::new(800.0, 600.0));
    assert_eq!(offset, Vec2::new(0.0, -20.0));
}

/// Zoom = 1 and eye at the view centre → offset is zero.
#[test]
fn eye_at_view_centre_zoom1_offset_zero() {
    let view = Vec2::new(640.0, 480.0);
    let eye = Pos2::new(view.x / 2.0, view.y / 2.0);
    let offset = eye_zoom_offset(eye, 1.0, view);
    assert_eq!(offset, Vec2::ZERO);
}

/// Doubling the zoom halves the offset distance from the view centre.
#[test]
fn doubling_zoom_adjusts_offset_correctly() {
    let eye = Pos2::new(50.0, 50.0);
    let view = Vec2::new(400.0, 300.0);
    let off2 = eye_zoom_offset(eye, 2.0, view);
    let off4 = eye_zoom_offset(eye, 4.0, view);
    // off2 = (200-100, 150-100) = (100, 50)
    // off4 = (200-200, 150-200) = (0, -50)
    assert_eq!(off2, Vec2::new(100.0, 50.0));
    assert_eq!(off4, Vec2::new(0.0, -50.0));
}

// ── detect_eye_positions ─────────────────────────────────────────────────────

fn solid_image(w: usize, h: usize, color: Color32) -> ColorImage {
    ColorImage { size: [w, h], pixels: vec![color; w * h], source_size: egui::Vec2::new(w as f32, h as f32) }
}

/// A completely black image has no faces: detection must return an empty Vec.
#[test]
fn blank_black_image_returns_no_eyes() {
    assert!(detect_eye_positions(&solid_image(64, 64, Color32::BLACK)).is_empty());
}

/// A solid-colour (non-black) image also has no faces.
#[test]
fn solid_grey_image_returns_no_eyes() {
    assert!(detect_eye_positions(&solid_image(128, 96, Color32::from_gray(180))).is_empty());
}

/// A 1×1 pixel image must not panic and must return an empty Vec.
#[test]
fn single_pixel_image_does_not_panic() {
    assert!(detect_eye_positions(&solid_image(1, 1, Color32::WHITE)).is_empty());
}
