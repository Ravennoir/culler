#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use eframe::egui;
use egui::{epaint::RectShape, Color32, ColorImage, Pos2, Rect, Shape, TextureHandle, Vec2};
use image::{codecs::gif::GifDecoder, imageops, AnimationDecoder, DynamicImage, ImageReader, Luma};
use ndarray::{s, Array, Array2, IxDyn};
use rayon::prelude::*;
use rustronomy_fits as rsf;
use jxl_oxide::integration::JxlDecoder;
use arboard::{Clipboard, ImageData};
use exif::{In, Reader as ExifReader};
use std::{
    cmp::Reverse,
    collections::{HashMap, HashSet},
    sync::mpsc,
    env,
    error::Error,
    fs,
    io::{BufReader, Cursor},
    path::{Path, PathBuf},
    time::Duration,
    borrow::Cow,
};

use lightningview::{eye_focus, raw_preview};

#[cfg(target_os = "windows")]
mod windows;
#[cfg(target_os = "windows")]
use crate::windows::*;

// --- Constants ---
const TILE_SIZE: usize = 1024; // Use tiles of 1024x1024 pixels for the detail view

// --- Supported Formats ---
pub const IMAGEREADER_SUPPORTED_FORMATS: [&str; 4] = ["webp", "tif", "tiff", "tga"];
pub const ANIM_SUPPORTED_FORMATS: [&str; 1] = ["gif"];
pub const IMAGE_RS_SUPPORTED_FORMATS: [&str; 9] = ["jpg", "jpeg", "png", "bmp", "svg", "ico", "pnm", "xbm", "xpm"];
pub const RAW_SUPPORTED_FORMATS: [&str; 24] = ["mrw", "arw", "srf", "sr2", "nef", "mef", "orf", "srw", "erf", "kdc", "dcs", "rw2", "raf", "dcr", "dng", "pef", "crw", "iiq", "3fr", "nrw", "mos", "cr2", "ari", "ori"];
pub const FITS_SUPPORTED_FORMATS: [&str; 2] = ["fits", "fit"];
pub const JXL_SUPPORTED_FORMATS: [&str; 1] = ["jxl"];

// --- Advanced Data Structures for Tiled Viewing ---
struct DisplayableImage {
    /// The full-resolution original image, kept in CPU memory.
    full_res_image: ColorImage,
    /// A single, downscaled texture for fast previews when zoomed out.
    preview_texture: TextureHandle,
    /// Cache for detail tiles to avoid re-uploading them to the GPU every frame.
    tile_cache: HashMap<(usize, usize), (TextureHandle, [usize; 2])>,
    /// Does this image actually need tiling, or is it small enough to fit on the GPU?
    needs_tiling: bool,
    /// True when this slot holds only a tiny thumbnail — a full-res preview is still loading.
    /// The full preview will replace this entry when it arrives.
    is_thumbnail: bool,
}

// Simplified enum for loaded image data before GPU upload
enum LoadedImage {
    Static(ColorImage),
    // For simplicity, this advanced example will treat GIFs as static, showing the first frame.
    // A fully tiled animated viewer is significantly more complex.
}

// --- Undo stack ---
#[derive(Debug, Clone)]
enum UndoAction {
    /// Image was dropped from the compare set with `.`
    DropFromCompareSet { order_pos: usize, slot: usize },
    /// A star rating was changed (stores the rating that was in place before the change)
    SetRating { path: PathBuf, previous_stars: Option<u8> },
}

// --- Main Application State ---
struct ImageViewerApp {
    // Compare / single-view state
    compare_images: HashMap<usize, DisplayableImage>, // image_order_pos -> loaded image
    compare_set: Vec<usize>,   // image_order positions displayed left→right
    compare_focus: usize,      // index into compare_set (which is "active")

    // Prefetch channels — kept separate so background never blocks priority.
    // priority_tx  : unbounded; at most ~13 items in flight (next/prev 10 + compare set).
    // background_tx: bounded sync_channel; rayon threads block when full, capping
    //                the number of decoded ColorImages queued in memory at any time.
    priority_tx:   mpsc::Sender<(usize, ColorImage)>,
    priority_rx:   mpsc::Receiver<(usize, ColorImage)>,
    background_tx: mpsc::SyncSender<(usize, ColorImage)>,
    background_rx: mpsc::Receiver<(usize, ColorImage)>,
    prefetch_pending: HashSet<usize>,

    // Culling mode
    is_culling_mode: bool,
    culling_min_stars: u8,           // 1–5: minimum rating required to appear
    culling_indices: Vec<usize>,     // sorted image_order positions passing the filter
    reference_image: Option<usize>,  // image_order pos pinned as reference (culling only)

    image_files: Vec<PathBuf>,
    current_index: usize,      // always == compare_set[compare_focus]
    image_order: Vec<usize>,
    zoom: f32,
    offset: Vec2,
    velocity: Vec2,
    is_scaled_to_fit: bool,
    is_fullscreen: bool,
    is_randomized: bool,
    show_delete_confirmation: bool,
    last_error: Option<String>,
    clipboard: Option<Clipboard>,
    ratings: HashMap<PathBuf, u8>,
    show_info_overlay: bool,
    undo_stack: Vec<UndoAction>,
    show_exif_overlay: bool,
    /// Detected eye positions (full-res pixel coords) for the currently viewed image.
    eye_positions: Vec<egui::Pos2>,
    /// Index into `eye_positions` — which eye is currently centred.
    eye_index: usize,
    /// The `current_index` for which `eye_positions` was computed.
    eye_positions_for: Option<usize>,
    /// Receives `(order_pos, eye_positions)` from the background detection thread.
    eye_rx: mpsc::Receiver<(usize, Vec<egui::Pos2>)>,
    eye_tx: mpsc::Sender<(usize, Vec<egui::Pos2>)>,
    /// True while a detection thread is running — prevents double-spawning.
    eye_detection_pending: bool,
    /// True when the last detection finished with no faces found.
    eye_no_face: bool,
    /// View size from the previous frame, used when 'E' is handled before layout.
    last_view_size: Vec2,
    show_help_overlay: bool,
    /// Zoom multiplier in compare mode (1.0 = fit each column), synchronized across all slots.
    compare_zoom: f32,
    /// Per-slot pan offsets in compare mode (order_pos → column-relative screen pixels).
    compare_offsets: HashMap<usize, Vec2>,
    /// Cached EXIF rows: (ifd_label, tag_name, value_string).
    /// Loaded on demand; cleared when the current image changes.
    exif_data: Vec<(String, String, String)>,
    exif_for_index: Option<usize>,
}

impl ImageViewerApp {
    fn new(cc: &eframe::CreationContext<'_>, path: Option<PathBuf>, initial_fullscreen: bool) -> Self {
        let (priority_tx, priority_rx) = mpsc::channel();
        // Capacity 8: allows up to 8 decoded background images to queue before
        // rayon threads block. Each embedded JPEG ColorImage ≈ 10 MB → ≤ 80 MB
        // of channel backlog at any time, regardless of folder size.
        let (background_tx, background_rx) = mpsc::sync_channel(8);
        // Eye detection results arrive here from the background thread.
        let (eye_tx, eye_rx) = mpsc::channel::<(usize, Vec<egui::Pos2>)>();
        let mut app = Self {
            compare_images: HashMap::new(),
            compare_set: Vec::new(),
            compare_focus: 0,
            priority_tx,
            priority_rx,
            background_tx,
            background_rx,
            prefetch_pending: HashSet::new(),
            is_culling_mode: false,
            culling_min_stars: 1,
            culling_indices: Vec::new(),
            reference_image: None,
            image_files: Vec::new(),
            current_index: 0,
            image_order: Vec::new(),
            zoom: 1.0,
            offset: Vec2::ZERO,
            velocity: Vec2::ZERO,
            is_scaled_to_fit: true,
            is_fullscreen: initial_fullscreen,
            is_randomized: false,
            show_delete_confirmation: false,
            last_error: None,
            clipboard: Clipboard::new().ok(),
            ratings: HashMap::new(),
            show_info_overlay: false,
            undo_stack: Vec::new(),
            show_exif_overlay: false,
            show_help_overlay: false,
            eye_positions: Vec::new(),
            eye_index: 0,
            eye_positions_for: None,
            eye_rx,
            eye_tx,
            eye_detection_pending: false,
            eye_no_face: false,
            last_view_size: Vec2::new(1920.0, 1080.0),
            compare_zoom: 1.0,
            compare_offsets: HashMap::new(),
            exif_data: Vec::new(),
            exif_for_index: None,
        };
        if let Some(path) = path {
            app.gather_images_from_directory(&path);
            if !app.image_files.is_empty() {
                let cur = app.current_index;
                app.compare_set = vec![cur];
                app.compare_focus = 0;

                // Stage 0 — synchronous thumbnail for instant first paint.
                // Reads only the tiny embedded thumbnail (~160×120, a few KB) from the
                // RAW container. Completes in ~2 ms and puts something on screen
                // before the full-res preview arrives from the background thread.
                app.load_thumbnail_sync(cur, &cc.egui_ctx);

                // Stage 1 — full preview for the current image (one dedicated Rayon task).
                // Triggers stages 2 & 3 (prefetch_adjacent + prefetch_all_remaining)
                // automatically in poll_prefetch once this image arrives.
                app.prefetch_images(&[cur]);

                cc.egui_ctx.request_repaint();
            } else {
                app.last_error = Some(format!("No supported images found in directory of '{}'", path.display()));
            }
        } else {
            app.last_error = Some("No image file specified.".to_string());
        }
        app.load_ratings();
        app
    }

    fn load_image_at_index(&mut self, index: usize, ctx: &egui::Context) {
        self.log_cache_state("navigate");
        self.current_index = index;
        // Enter single-image view and clear any compare/reference state,
        // but DO NOT clear compare_images — it is the prefetch cache.
        self.compare_set = vec![index];
        self.compare_focus = 0;
        self.reference_image = None;
        self.is_scaled_to_fit = true;
        self.last_error = None;

        // If already fully loaded by prefetch, it is instantly visible — nothing to do.
        let already_full = self.compare_images.get(&index)
            .map(|d| !d.is_thumbnail)
            .unwrap_or(false);

        if !already_full {
            // Show the tiny embedded thumbnail immediately (~2 ms) while the
            // full preview decodes in the background.
            if !self.compare_images.contains_key(&index) {
                self.load_thumbnail_sync(index, ctx);
            }
            self.prefetch_images(&[index]);
        }

        // Evict images far from the new position to keep memory bounded.
        self.evict_distant_images(index);
        self.prefetch_adjacent(index);
        ctx.request_repaint();
    }

    /// Drop cached images that are far from the current compare window.
    ///
    /// Two-pass eviction:
    ///   Pass 1 — distance: drop everything beyond KEEP_RADIUS from the compare window edges.
    ///   Pass 2 — memory cap: if the cache still exceeds CACHE_LIMIT_MB, evict the images
    ///            furthest from center one by one until we are under the limit.
    ///
    /// Background threads for evicted positions are left to finish — their results
    /// arrive in poll_prefetch and are evicted again on the same frame, so memory
    /// stays flat. Evicting from prefetch_pending here would race with the thread's
    /// send and could leave the pending set inconsistent.
    fn evict_distant_images(&mut self, center: usize) {
        /// Never evict images within this many positions of the compare window edges.
        const KEEP_RADIUS:    usize = 10;
        /// Hard RSS cap for ColorImage pixel data.  Embedded JPEG previews from
        /// high-resolution cameras (Nikon Z7 II, Sony A7R, …) decode to ~40–60 MB
        /// each; 800 MB comfortably holds ~15–20 such images while leaving headroom
        /// for the OS, GPU textures, and the bounded sync_channel(8) backlog.
        const CACHE_LIMIT_MB: f64   = 800.0;

        let keep_set: HashSet<usize> = self.compare_set.iter().copied().collect();
        let right = self.compare_set.iter().copied().max().unwrap_or(center);
        let left  = self.compare_set.iter().copied().min().unwrap_or(center);

        // ── Pass 1: distance-based ────────────────────────────────────────────
        let before = self.compare_images.len();
        self.compare_images.retain(|&pos, _| {
            keep_set.contains(&pos)
                || pos.abs_diff(right) <= KEEP_RADIUS
                || pos.abs_diff(left)  <= KEEP_RADIUS
        });

        // ── Pass 2: memory cap ────────────────────────────────────────────────
        // Build a sorted list of (distance_from_center, pos) for everything not
        // in the compare_set, then drop the most-distant entries until under cap.
        let cache_bytes: u64 = self.compare_images.values()
            .map(|d| (d.full_res_image.width() * d.full_res_image.height() * 4) as u64)
            .sum();
        let limit_bytes = (CACHE_LIMIT_MB * 1_048_576.0) as u64;

        if cache_bytes > limit_bytes {
            // Collect evictable positions sorted farthest-first.
            let mut evictable: Vec<usize> = self.compare_images.keys()
                .copied()
                .filter(|p| !keep_set.contains(p))
                .collect();
            evictable.sort_by_key(|&p| Reverse(p.abs_diff(center)));

            let mut current_bytes = cache_bytes;
            for pos in evictable {
                if current_bytes <= limit_bytes { break; }
                if let Some(d) = self.compare_images.remove(&pos) {
                    current_bytes -= (d.full_res_image.width() * d.full_res_image.height() * 4) as u64;
                }
            }
        }

        let evicted = before - self.compare_images.len();
        let mem_mb: f64 = self.compare_images.values()
            .map(|d| (d.full_res_image.width() * d.full_res_image.height() * 4) as f64 / 1_048_576.0)
            .sum();
        if evicted > 0 {
            log::debug!(
                "evict: removed={} retained={} cache_mem={:.1}MB center={} pending={}",
                evicted, self.compare_images.len(), mem_mb, center, self.prefetch_pending.len()
            );
        } else {
            log::debug!(
                "evict: no change cache_count={} cache_mem={:.1}MB center={} pending={}",
                self.compare_images.len(), mem_mb, center, self.prefetch_pending.len()
            );
        }
    }

    /// Queue a background load for `order_pos` if not already cached or in-flight.
    /// Shows a thumbnail immediately for RAW files while the full preview loads.
    fn ensure_image_loaded(&mut self, order_pos: usize, ctx: &egui::Context) {
        if self.compare_images.get(&order_pos).map(|d| !d.is_thumbnail).unwrap_or(false) { return; }
        if !self.compare_images.contains_key(&order_pos) {
            self.load_thumbnail_sync(order_pos, ctx);
        }
        self.prefetch_images(&[order_pos]);
    }

    /// Synchronously extract and display the tiny embedded thumbnail (~160×120) for RAW files.
    /// Completes in ~2 ms and gives instant first paint before the full preview arrives.
    /// For non-RAW files this is a no-op; the full image loads via the normal prefetch path.
    fn load_thumbnail_sync(&mut self, order_pos: usize, ctx: &egui::Context) {
        let Some(&file_idx) = self.image_order.get(order_pos) else { return };
        let Some(path) = self.image_files.get(file_idx).cloned() else { return };
        let ext = path.extension().and_then(|s| s.to_str()).unwrap_or("").to_lowercase();
        if !RAW_SUPPORTED_FORMATS.contains(&ext.as_str()) { return; }

        let Some(thumb_bytes) = raw_preview::extract_thumbnail_jpeg(&path) else { return };
        // If the smallest embedded JPEG is large (some cameras embed only one big preview),
        // decoding it synchronously would block the UI. Skip it and let the priority
        // thread deliver it — the threshold is ~500 KB which decodes in under 20 ms.
        if thumb_bytes.len() > 500_000 { return; }
        let Ok(img) = image::load_from_memory_with_format(&thumb_bytes, image::ImageFormat::Jpeg)
            else { return };

        let color_image = to_egui_color_image(apply_exif_orientation(img, &path));
        let texture = ctx.load_texture(
            format!("{}_thumb", path.display()),
            color_image.clone(),
            Default::default(),
        );
        self.compare_images.insert(order_pos, DisplayableImage {
            full_res_image: color_image,
            preview_texture: texture,
            tile_cache: HashMap::new(),
            needs_tiling: false, // thumbnails are tiny, never need tiling
            is_thumbnail: true,
        });
    }

    fn copy_to_clipboard(&mut self) {
        let data = self.compare_images.get(&self.current_index).map(|img| {
            let fr = &img.full_res_image;
            let bytes: Vec<u8> = fr.pixels.iter().flat_map(|c| c.to_array()).collect();
            (fr.width(), fr.height(), bytes)
        });
        if let (Some((w, h, bytes)), Some(clipboard)) = (data, &mut self.clipboard) {
            let image_data = ImageData { width: w, height: h, bytes: Cow::from(bytes) };
            log::info!("Copying image: {}x{}", w, h);
            if let Err(e) = clipboard.set_image(image_data) {
                self.last_error = Some(format!("Failed to copy to clipboard: {}", e));
            } else {
                log::info!("Image copied to clipboard.");
            }
        }
    }

    fn gather_images_from_directory(&mut self, file_path: &Path) {
        let Some(parent_dir) = file_path.parent() else {
            self.last_error = Some("Failed to get parent directory.".to_string());
            return;
        };

        let all_supported_formats: Vec<&str> = [
            &IMAGEREADER_SUPPORTED_FORMATS[..],
            &ANIM_SUPPORTED_FORMATS[..],
            &IMAGE_RS_SUPPORTED_FORMATS[..],
            &RAW_SUPPORTED_FORMATS[..],
            &FITS_SUPPORTED_FORMATS[..],
            &JXL_SUPPORTED_FORMATS[..],
        ]
        .concat();

        let Ok(entries) = fs::read_dir(parent_dir) else { return };
        let mut files: Vec<PathBuf> = entries
            .filter_map(|entry| entry.ok().map(|e| e.path()))
            .filter(|path| {
                path.is_file() && {
                    let path_str = path.to_string_lossy().to_lowercase();
                    all_supported_formats.iter().any(|fmt| path_str.ends_with(fmt))
                }
            })
            .collect();

        files.sort_by_key(|name| name.to_string_lossy().to_lowercase());

        if let Some(index) = files.iter().position(|p| p == file_path) {
            self.current_index = index;
        }

        self.image_files = files;
        self.image_order = (0..self.image_files.len()).collect();
    }
    
    fn next_image(&mut self, ctx: &egui::Context) {
        if self.is_culling_mode {
            let next = self.culling_indices.iter().find(|&&p| p > self.current_index).copied();
            if let Some(pos) = next { self.navigate_to(pos, ctx); }
        } else if !self.image_files.is_empty() {
            self.navigate_to((self.current_index + 1) % self.image_files.len(), ctx);
        }
    }

    fn prev_image(&mut self, ctx: &egui::Context) {
        if self.is_culling_mode {
            let prev = self.culling_indices.iter().rev().find(|&&p| p < self.current_index).copied();
            if let Some(pos) = prev { self.navigate_to(pos, ctx); }
        } else if !self.image_files.is_empty() {
            self.navigate_to((self.current_index + self.image_files.len() - 1) % self.image_files.len(), ctx);
        }
    }

    fn log_cache_state(&self, action: &str) {
        let mem_mb: f64 = self.compare_images.values()
            .map(|d| (d.full_res_image.width() * d.full_res_image.height() * 4) as f64 / 1_048_576.0)
            .sum();
        log::info!(
            "[{}] idx={} compare={:?} cache_count={} cache_mem={:.1}MB pending={}",
            action,
            self.current_index,
            self.compare_set,
            self.compare_images.len(),
            mem_mb,
            self.prefetch_pending.len(),
        );
    }

    /// Navigate to `pos`, keeping the reference image pinned if one is set.
    fn navigate_to(&mut self, pos: usize, ctx: &egui::Context) {
        if let Some(ref_pos) = self.reference_image {
            if self.compare_set == vec![ref_pos] {
                self.compare_set.push(pos);
                self.compare_focus = 1;
                self.current_index = pos;
                self.ensure_image_loaded(pos, ctx);
                self.is_scaled_to_fit = true;
                self.prefetch_adjacent(pos);
                return;
            }
            if self.compare_set.len() > 1 {
                if self.compare_set[self.compare_focus] != ref_pos {
                    let slot = self.compare_focus;
                    let old = self.compare_set[slot];
                    self.compare_set[slot] = pos;
                    self.current_index = pos;
                    self.compare_images.remove(&old);
                    self.ensure_image_loaded(pos, ctx);
                    self.is_scaled_to_fit = true;
                    self.prefetch_adjacent(pos);
                    return;
                }
                if let Some(other_slot) = (0..self.compare_set.len()).find(|&s| self.compare_set[s] != ref_pos) {
                    let old = self.compare_set[other_slot];
                    self.compare_set[other_slot] = pos;
                    self.compare_focus = other_slot;
                    self.current_index = pos;
                    self.compare_images.remove(&old);
                    self.ensure_image_loaded(pos, ctx);
                    self.is_scaled_to_fit = true;
                    self.prefetch_adjacent(pos);
                    return;
                }
            }
        }
        self.load_image_at_index(pos, ctx); // already calls prefetch_adjacent
    }

    fn first_image(&mut self, ctx: &egui::Context) {
        if self.is_culling_mode {
            if let Some(&pos) = self.culling_indices.first() { self.load_image_at_index(pos, ctx); }
        } else if !self.image_files.is_empty() {
            self.load_image_at_index(0, ctx);
        }
    }


    fn last_image(&mut self, ctx: &egui::Context) {
        if self.is_culling_mode {
            if let Some(&pos) = self.culling_indices.last() { self.load_image_at_index(pos, ctx); }
        } else if !self.image_files.is_empty() {
            self.load_image_at_index(self.image_files.len() - 1, ctx);
        }
    }

    // --- Culling ---

    fn rebuild_culling_indices(&mut self) {
        let new_indices: Vec<usize> = (0..self.image_files.len())
            .filter(|&order_pos| {
                let path = &self.image_files[self.image_order[order_pos]];
                self.ratings.get(path).copied().unwrap_or(0) >= self.culling_min_stars
            })
            .collect();
        self.culling_indices = new_indices;
    }

    fn enter_culling_mode(&mut self, ctx: &egui::Context) {
        self.is_culling_mode = true;
        self.rebuild_culling_indices();
        if self.culling_indices.is_empty() { return; }
        // Remove any compare slots that don't pass the filter.
        self.apply_culling_to_compare_set(ctx);
        // In single-image view, jump to the first passing image if needed.
        if self.compare_set.len() <= 1 && !self.culling_indices.contains(&self.current_index) {
            let first = self.culling_indices[0];
            self.load_image_at_index(first, ctx);
        }
    }

    fn exit_culling_mode(&mut self) {
        self.is_culling_mode = false;
        self.culling_indices.clear();
        self.reference_image = None;
    }

    /// Remove compare_set slots that no longer pass the current culling filter,
    /// then clamp focus. If the set becomes empty, navigate to the first passing image.
    /// No-op when in single-image view (compare_set.len() <= 1).
    fn apply_culling_to_compare_set(&mut self, ctx: &egui::Context) {
        if self.compare_set.len() <= 1 { return; }
        let ci = self.culling_indices.clone(); // clone to sidestep borrow split
        let old_len = self.compare_set.len();
        self.compare_set.retain(|p| ci.contains(p));
        if self.compare_set.len() == old_len { return; } // nothing was removed

        if self.compare_focus >= self.compare_set.len().max(1) {
            self.compare_focus = self.compare_set.len().saturating_sub(1);
        }
        if self.compare_set.is_empty() {
            if let Some(&first) = self.culling_indices.first() {
                self.load_image_at_index(first, ctx);
            }
        } else {
            self.current_index = self.compare_set[self.compare_focus];
        }
    }
    
    // --- Prefetch ---

    /// HIGH-PRIORITY load: dedicated OS thread per job so it always starts immediately.
    /// Use for the current image and the next/prev 10 the user can reach right now.
    fn prefetch_images(&mut self, positions: &[usize]) {
        self.spawn_prefetch_jobs(positions, true);
    }

    /// LOW-PRIORITY load: uses the global rayon pool (bounded by num_cpus).
    /// Never competes with priority threads because priority uses separate OS threads.
    fn prefetch_background(&mut self, positions: &[usize]) {
        self.spawn_prefetch_jobs(positions, false);
    }

    /// Queue every image in the folder on the background rayon pool.
    /// Called once after the first image appears; warms the entire cache so
    /// navigation anywhere eventually becomes instant.
    fn prefetch_all_remaining(&mut self) {
        let all: Vec<usize> = (0..self.image_files.len()).collect();
        self.prefetch_background(&all);
    }

    fn spawn_prefetch_jobs(&mut self, positions: &[usize], priority: bool) {
        for &pos in positions {
            if self.compare_images.get(&pos).map(|d| !d.is_thumbnail).unwrap_or(false) { continue; }
            if self.prefetch_pending.contains(&pos) { continue; }
            let Some(&file_idx) = self.image_order.get(pos) else { continue };
            let Some(path) = self.image_files.get(file_idx).cloned() else { continue };
            self.prefetch_pending.insert(pos);
            if priority {
                // Dedicated OS thread — starts immediately, sends to the unbounded
                // priority channel so it never blocks waiting for the main thread.
                let tx = self.priority_tx.clone();
                std::thread::spawn(move || {
                    log::debug!("thread[{}]: starting load", pos);
                    match load_image(&path) {
                        Ok(LoadedImage::Static(img)) => {
                            log::debug!("thread[{}]: decoded {}×{}, sending", pos, img.width(), img.height());
                            match tx.send((pos, img)) {
                                Ok(()) => log::debug!("thread[{}]: sent OK", pos),
                                Err(_) => log::warn!("thread[{}]: send FAILED — receiver dropped", pos),
                            }
                        }
                        Err(e) => log::warn!("thread[{}]: load FAILED: {}", pos, e),
                    }
                });
            } else {
                // Rayon global pool — bounded by num_cpus threads. Sends to the
                // bounded sync_channel: when the channel is full (8 items) the
                // rayon thread blocks, preventing unbounded memory accumulation.
                let tx = self.background_tx.clone();
                rayon::spawn(move || {
                    log::debug!("bg_thread[{}]: starting load", pos);
                    match load_image(&path) {
                        Ok(LoadedImage::Static(img)) => {
                            log::debug!("bg_thread[{}]: decoded {}×{}, sending", pos, img.width(), img.height());
                            match tx.send((pos, img)) {
                                Ok(()) => log::debug!("bg_thread[{}]: sent OK", pos),
                                Err(_) => log::warn!("bg_thread[{}]: send FAILED — receiver dropped", pos),
                            }
                        }
                        Err(e) => log::warn!("bg_thread[{}]: load FAILED: {}", pos, e),
                    }
                });
            }
        }
    }

    /// Drain both prefetch channels and upload completed images to the GPU.
    /// Called every frame from the main (UI) thread.
    fn poll_prefetch(&mut self, ctx: &egui::Context) {
        let had_full_current = self.compare_images.get(&self.current_index)
            .map(|d| !d.is_thumbnail)
            .unwrap_or(false);
        let mut any_new = false;

        // ── Priority channel: drain everything ──────────────────────────────
        let mut priority_drained = 0u32;
        while let Ok((pos, color_image)) = self.priority_rx.try_recv() {
            priority_drained += 1;
            self.prefetch_pending.remove(&pos);
            if self.compare_images.get(&pos).map(|d| !d.is_thumbnail).unwrap_or(false) { continue; }
            any_new |= self.insert_decoded_image(pos, color_image, ctx);
        }

        // ── Background channel: at most 4 per frame ─────────────────────────
        let mut bg_processed = 0u32;
        while bg_processed < 4 {
            match self.background_rx.try_recv() {
                Ok((pos, color_image)) => {
                    bg_processed += 1;
                    self.prefetch_pending.remove(&pos);
                    if self.compare_images.get(&pos).map(|d| !d.is_thumbnail).unwrap_or(false) { continue; }
                    self.insert_decoded_image(pos, color_image, ctx);
                }
                Err(_) => break,
            }
        }
        if priority_drained > 0 || bg_processed > 0 {
            log::debug!("poll_prefetch: drained priority={} bg={} pending={}",
                priority_drained, bg_processed, self.prefetch_pending.len());
        }

        // Distance-based eviction every frame — keeps the cache window bounded
        // regardless of how many background results have arrived.
        self.evict_distant_images(self.current_index);

        // Keep the spinner animating while anything is still loading.
        if !self.prefetch_pending.is_empty() {
            ctx.request_repaint();
        }

        // When the current image's full preview first arrives:
        // 1. Expand priority prefetch (next 10 ahead/behind).
        // 2. Kick off full-folder background warming — rayon threads will block
        //    on the bounded channel when 8 results are already queued, so this
        //    never accumulates unbounded memory.
        if !had_full_current && any_new {
            let cur = self.current_index;
            self.prefetch_adjacent(cur);
            self.prefetch_all_remaining();
            ctx.request_repaint();
        }
    }

    /// Insert a decoded ColorImage into compare_images and return true if it
    /// was the current image (triggers a scale-to-fit reset).
    fn insert_decoded_image(&mut self, pos: usize, color_image: ColorImage, ctx: &egui::Context) -> bool {
        let Some(&file_idx) = self.image_order.get(pos) else { return false };
        let Some(path) = self.image_files.get(file_idx).cloned() else { return false };
        let max_texture_side = 2048;
        let needs_tiling = color_image.width() > max_texture_side || color_image.height() > max_texture_side;
        let preview_image = if needs_tiling {
            downscale_color_image(color_image.clone(), max_texture_side)
        } else {
            color_image.clone()
        };
        let preview_texture = ctx.load_texture(
            format!("{}_preview", path.display()),
            preview_image,
            Default::default(),
        );
        let bytes = color_image.width() * color_image.height() * 4;
        self.compare_images.insert(pos, DisplayableImage {
            full_res_image: color_image,
            preview_texture,
            tile_cache: HashMap::new(),
            needs_tiling,
            is_thumbnail: false,
        });
        log::debug!(
            "cache insert pos={} path={} size={}×{} mem={:.1}MB cache_count={}",
            pos,
            path.display(),
            self.compare_images.get(&pos).map(|d| d.full_res_image.width()).unwrap_or(0),
            self.compare_images.get(&pos).map(|d| d.full_res_image.height()).unwrap_or(0),
            bytes as f64 / 1_048_576.0,
            self.compare_images.len(),
        );
        if pos == self.current_index {
            self.is_scaled_to_fit = true;
            return true;
        }
        false
    }

    /// Priority-prefetch images the user can reach on their next action.
    ///
    /// In compare mode we look ahead from the RIGHTMOST slot (not just current_index)
    /// because SHIFT+Right adds images beyond the right edge of the visible window.
    /// We look behind from the LEFTMOST slot for the same reason.
    fn prefetch_adjacent(&mut self, _center: usize) {
        const LOOK_AHEAD:  usize = 10;
        const LOOK_BEHIND: usize = 3;

        let positions: Vec<usize> = if self.is_culling_mode {
            let ci = self.culling_indices.clone();
            // Forward edge: rightmost compare slot present in culling indices.
            let right_edge = self.compare_set.iter()
                .filter_map(|&p| ci.iter().position(|&c| c == p))
                .max()
                .unwrap_or(0);
            // Backward edge: leftmost compare slot present in culling indices.
            let left_edge  = self.compare_set.iter()
                .filter_map(|&p| ci.iter().position(|&c| c == p))
                .min()
                .unwrap_or(0);
            let start = left_edge.saturating_sub(LOOK_BEHIND);
            let end   = (right_edge + LOOK_AHEAD + 1).min(ci.len());
            let in_set: HashSet<usize> = self.compare_set.iter().copied().collect();
            ci[start..end].iter().copied().filter(|p| !in_set.contains(p)).collect()
        } else {
            let total     = self.image_files.len();
            let right_pos = self.compare_set.iter().copied().max().unwrap_or(_center);
            let left_pos  = self.compare_set.iter().copied().min().unwrap_or(_center);
            let start = left_pos.saturating_sub(LOOK_BEHIND);
            let end   = (right_pos + LOOK_AHEAD + 1).min(total);
            let in_set: HashSet<usize> = self.compare_set.iter().copied().collect();
            (start..end).filter(|p| !in_set.contains(p)).collect()
        };
        self.prefetch_images(&positions);
    }

    // --- Compare Mode ---

    /// SHIFT+Right: shrink from the left if images were added there, else extend right.
    fn compare_shift_right(&mut self, ctx: &egui::Context) {
        self.log_cache_state("compare_shift_right");
        if self.compare_focus > 0 {
            // Images were added to the left — remove leftmost (unless it is the reference)
            if self.reference_image == Some(self.compare_set[0]) { return; }
            let removed = self.compare_set.remove(0);
            self.compare_images.remove(&removed);
            self.compare_focus -= 1;
        } else {
            // Extend right (respects culling filter)
            let rightmost = *self.compare_set.last().unwrap_or(&self.current_index);
            let next = if self.is_culling_mode {
                self.culling_indices.iter().find(|&&p| p > rightmost).copied()
            } else {
                let n = rightmost + 1;
                if n < self.image_files.len() { Some(n) } else { None }
            };
            if let Some(pos) = next {
                self.compare_set.push(pos);
                self.ensure_image_loaded(pos, ctx);
                self.is_scaled_to_fit = true;
            }
        }
    }

    /// SHIFT+Left: shrink from the right if images were added there, else extend left.
    fn compare_shift_left(&mut self, ctx: &egui::Context) {
        self.log_cache_state("compare_shift_left");
        let n = self.compare_set.len();
        if self.compare_focus < n.saturating_sub(1) {
            // Images were added to the right — remove rightmost (unless it is the reference)
            if self.reference_image == self.compare_set.last().copied() { return; }
            let removed = self.compare_set.pop().unwrap();
            self.compare_images.remove(&removed);
        } else {
            // Extend left (respects culling filter)
            let leftmost = self.compare_set[0];
            let prev = if self.is_culling_mode {
                self.culling_indices.iter().rev().find(|&&p| p < leftmost).copied()
            } else {
                leftmost.checked_sub(1)
            };
            if let Some(pos) = prev {
                self.compare_set.insert(0, pos);
                self.compare_focus += 1;
                self.ensure_image_loaded(pos, ctx);
                self.is_scaled_to_fit = true;
            }
        }
    }

    /// Left/Right in compare mode: move selection within the compare set.
    fn compare_move_focus(&mut self, delta: i32) {
        let new_focus = (self.compare_focus as i32 + delta)
            .clamp(0, self.compare_set.len() as i32 - 1) as usize;
        if new_focus != self.compare_focus {
            self.compare_focus = new_focus;
            self.current_index = self.compare_set[new_focus];
        }
    }

    fn toggle_random_order(&mut self) {
        let cur = self.image_order[self.current_index];
        if self.is_randomized {
            #[allow(deprecated)]
            let mut rng = rand::rng();
            use rand::seq::SliceRandom;
            self.image_order.shuffle(&mut rng);
        } else {
            self.image_order = (0..self.image_files.len()).collect();
        }
        if let Some(p) = self.image_order.iter().position(|&i| i == cur) {
            self.current_index = p;
        }
    }

    /// Shift+Alt+Left/Right: swap the focused slot with its neighbour (reorder columns).
    fn compare_swap_slot(&mut self, forward: bool) {
        let n = self.compare_set.len();
        if n <= 1 { return; }
        let neighbour = if forward {
            if self.compare_focus + 1 >= n { return; }
            self.compare_focus + 1
        } else {
            if self.compare_focus == 0 { return; }
            self.compare_focus - 1
        };
        self.compare_set.swap(self.compare_focus, neighbour);
        self.compare_focus = neighbour;
        // current_index follows the focused image — it didn't change, only its column did.
    }

    /// Alt+Left/Right: shift the entire compare window by its width (next/prev batch).
    /// The reference slot (if any) is never moved.
    fn compare_window_shift(&mut self, forward: bool, ctx: &egui::Context) {
        let n = self.compare_set.len();
        if n == 0 { return; }

        let ref_pos  = self.reference_image;
        let ref_slot = ref_pos.and_then(|rp| self.compare_set.iter().position(|&p| p == rp));
        let move_count = n - if ref_slot.is_some() { 1 } else { 0 };
        if move_count == 0 { return; }

        let step = move_count;

        // Non-reference slots in their current display order.
        let non_ref: Vec<usize> = self.compare_set.iter()
            .copied().filter(|&p| Some(p) != ref_pos).collect();

        if self.is_culling_mode {
            let ci = self.culling_indices.clone();
            if ci.is_empty() { return; }

            let new_non_ref: Vec<usize> = if forward {
                // Keep the tail of non-ref slots (drop `step` from the front).
                let kept: Vec<usize> = non_ref[step.min(non_ref.len())..].to_vec();
                let anchor = *non_ref.last().unwrap();
                let new_part: Vec<usize> = ci.iter()
                    .copied()
                    .filter(|&p| p > anchor && Some(p) != ref_pos)
                    .take(step)
                    .collect();
                if new_part.len() < step { return; }
                [kept, new_part].concat()
            } else {
                // Keep the head of non-ref slots (drop `step` from the back).
                let kept: Vec<usize> = non_ref[..non_ref.len().saturating_sub(step)].to_vec();
                let anchor = *non_ref.first().unwrap();
                let mut new_part: Vec<usize> = ci.iter()
                    .rev()
                    .copied()
                    .filter(|&p| p < anchor && Some(p) != ref_pos)
                    .take(step)
                    .collect();
                if new_part.len() < step { return; }
                new_part.reverse();
                [new_part, kept].concat()
            };
            if new_non_ref.is_empty() { return; }

            // Rebuild compare_set: insert reference back at its original slot.
            let mut new_set = new_non_ref.clone();
            if let (Some(slot), Some(rp)) = (ref_slot, ref_pos) {
                new_set.insert(slot.min(new_set.len()), rp);
            }

            // Keep-zone: ±n images around the new window (plus reference).
            let new_first = *new_non_ref.first().unwrap();
            let new_last  = *new_non_ref.last().unwrap();
            let ci_first  = ci.iter().position(|&p| p == new_first).unwrap_or(0);
            let ci_last   = ci.iter().position(|&p| p == new_last).unwrap_or(0);
            let keep_ci_start = ci_first.saturating_sub(n);
            let keep_ci_end   = (ci_last + n + 1).min(ci.len());
            let mut keep: HashSet<usize> = ci[keep_ci_start..keep_ci_end].iter().copied().collect();
            if let Some(rp) = ref_pos { keep.insert(rp); }

            self.compare_set = new_set.clone();
            if self.compare_focus >= self.compare_set.len() { self.compare_focus = self.compare_set.len() - 1; }
            if ref_slot.is_some() && self.compare_set.get(self.compare_focus).copied() == ref_pos {
                if let Some(s) = (0..self.compare_set.len()).find(|&s| Some(self.compare_set[s]) != ref_pos) {
                    self.compare_focus = s;
                }
            }
            self.current_index = self.compare_set[self.compare_focus];

            for pos in new_set.iter().copied() { self.ensure_image_loaded(pos, ctx); }

            self.compare_images.retain(|&k, _| keep.contains(&k));
            self.prefetch_pending.retain(|&k| keep.contains(&k));

            let to_prefetch: Vec<usize> = ci[keep_ci_start..keep_ci_end].iter()
                .copied().filter(|p| !new_set.contains(p)).collect();
            self.prefetch_images(&to_prefetch);

        } else {
            // Standard contiguous window (non-culling).
            let new_non_ref: Vec<usize> = if forward {
                let kept: Vec<usize> = non_ref[step.min(non_ref.len())..].to_vec();
                let last = *non_ref.last().unwrap();
                let new_part: Vec<usize> = (last + 1..last + 1 + step)
                    .filter(|&p| Some(p) != ref_pos && p < self.image_files.len())
                    .collect();
                if new_part.len() < step { return; }
                [kept, new_part].concat()
            } else {
                let kept: Vec<usize> = non_ref[..non_ref.len().saturating_sub(step)].to_vec();
                let first = *non_ref.first().unwrap();
                if first < step { return; }
                let new_part: Vec<usize> = (first - step..first)
                    .filter(|&p| Some(p) != ref_pos)
                    .collect();
                if new_part.len() < step { return; }
                [new_part, kept].concat()
            };

            let mut new_set = new_non_ref.clone();
            if let (Some(slot), Some(rp)) = (ref_slot, ref_pos) {
                new_set.insert(slot.min(new_set.len()), rp);
            }

            let new_first = new_non_ref.first().copied().unwrap_or(0);
            let new_last  = new_non_ref.last().copied().unwrap_or(0);
            let keep_start = new_first.saturating_sub(n);
            let keep_end   = (new_last + n + 1).min(self.image_files.len());

            self.compare_set = new_set.clone();
            if self.compare_focus >= self.compare_set.len() { self.compare_focus = self.compare_set.len() - 1; }
            if ref_slot.is_some() && self.compare_set.get(self.compare_focus).copied() == ref_pos {
                if let Some(s) = (0..self.compare_set.len()).find(|&s| Some(self.compare_set[s]) != ref_pos) {
                    self.compare_focus = s;
                }
            }
            self.current_index = self.compare_set[self.compare_focus];

            for pos in new_set.iter().copied() { self.ensure_image_loaded(pos, ctx); }

            self.compare_images.retain(|&k, _| {
                (k >= keep_start && k < keep_end) || Some(k) == ref_pos
            });
            self.prefetch_pending.retain(|&k| k >= keep_start && k < keep_end);

            let next: Vec<usize> = (new_last + 1..keep_end).collect();
            let prev: Vec<usize> = (keep_start..new_first).collect();
            self.prefetch_images(&next);
            self.prefetch_images(&prev);
        }

        self.is_scaled_to_fit = true;
    }

    /// Read EXIF from the current image and cache it in `exif_data`.
    /// Groups fields by IFD, skips thumbnail IFD (In1), sorts alphabetically within each group.
    fn load_exif_for_current(&mut self) {
        let idx = self.current_index;
        if self.exif_for_index == Some(idx) { return; } // already cached
        self.exif_data.clear();
        self.exif_for_index = Some(idx);

        let Some(path) = self.image_files.get(self.image_order[idx]) else { return };
        let Ok(file) = std::fs::File::open(path) else { return };
        let mut buf = BufReader::new(file);
        let Ok(exif) = ExifReader::new().read_from_container(&mut buf) else { return };

        // Collect all fields, skipping thumbnail IFD.
        let mut rows: Vec<(String, String, String)> = exif
            .fields()
            .filter(|f| f.ifd_num != In::THUMBNAIL)
            .map(|f| (
                format!("{}", f.ifd_num),
                format!("{}", f.tag),
                f.display_value().with_unit(&exif).to_string(),
            ))
            .collect();

        // Sort: group by IFD label, then alphabetically by tag name.
        rows.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));
        self.exif_data = rows;
    }

    fn ratings_file_path() -> Option<PathBuf> {
        #[cfg(target_os = "windows")]
        let base = std::env::var("APPDATA").ok().map(PathBuf::from)?;
        #[cfg(not(target_os = "windows"))]
        let base = std::env::var("HOME").ok().map(|h| PathBuf::from(h).join(".config"))?;
        Some(base.join("lightningview").join("ratings.tsv"))
    }

    fn load_ratings(&mut self) {
        let Some(path) = Self::ratings_file_path() else { return };
        let Ok(contents) = fs::read_to_string(&path) else { return };
        for line in contents.lines() {
            let mut parts = line.splitn(2, '\t');
            if let (Some(img_path), Some(stars_str)) = (parts.next(), parts.next()) {
                if let Ok(stars) = stars_str.trim().parse::<u8>() {
                    self.ratings.insert(PathBuf::from(img_path), stars.min(5));
                }
            }
        }
    }

    fn save_ratings(&self) {
        let Some(path) = Self::ratings_file_path() else { return };
        if let Some(parent) = path.parent() {
            let _ = fs::create_dir_all(parent);
        }
        let contents: String = self.ratings
            .iter()
            .map(|(p, &s)| format!("{}\t{}\n", p.display(), s))
            .collect();
        let _ = fs::write(&path, contents);
    }

    /// Called when the user presses 'E'.
    ///
    /// First press on a new image: downscales to grayscale on the UI thread
    /// (~5 ms) then hands detection off to a background thread (~100–400 ms).
    /// The zoom is applied once the result arrives via `poll_eye_detection`.
    ///
    /// Subsequent presses (same image, result already cached) cycle through
    /// the detected eyes immediately ("reroll").
    fn cycle_eye_focus(&mut self) {
        let idx = self.current_index;
        log::info!("cycle_eye_focus: idx={} for={:?} pending={}", idx, self.eye_positions_for, self.eye_detection_pending);

        // Cached result for this image → just cycle.
        if self.eye_positions_for == Some(idx) {
            log::info!("cycle_eye_focus: cached {} eye(s), cycling", self.eye_positions.len());
            if !self.eye_positions.is_empty() {
                self.eye_index = (self.eye_index + 1) % self.eye_positions.len();
                self.apply_eye_zoom();
            }
            return;
        }

        // Detection already running for this image → ignore the keypress.
        if self.eye_detection_pending {
            log::info!("cycle_eye_focus: detection already pending, ignoring");
            return;
        }

        let Some(img) = self.compare_images.get(&idx) else {
            log::info!("cycle_eye_focus: image not in cache, cannot detect");
            return;
        };
        if img.is_thumbnail {
            log::info!("cycle_eye_focus: image is still a thumbnail, skipping");
            return;
        }

        // For RAW files extract the embedded JPEG directly (capped at 2 MP) so
        // detection never runs on a full 24 MP decode.  Fall back to full_res_image
        // for non-RAW formats or when no embedded JPEG is found.
        let path = self.image_files.get(self.image_order[idx]).cloned();
        let ext = path.as_ref()
            .and_then(|p| p.extension())
            .and_then(|s| s.to_str())
            .map(|s| s.to_lowercase())
            .unwrap_or_default();

        let jpeg_gray = if RAW_SUPPORTED_FORMATS.contains(&ext.as_str()) {
            path.as_ref()
                .and_then(|p| raw_preview::extract_preview_jpeg_capped(p, 2_000_000))
                .and_then(|jpeg| eye_focus::prepare_gray_from_jpeg(&jpeg))
        } else {
            None
        };
        let (gray, w, h, scale_inv) = jpeg_gray
            .unwrap_or_else(|| eye_focus::prepare_gray(&img.full_res_image));

        log::info!(
            "cycle_eye_focus: spawning detection thread for idx={} gray={}x{} source={}",
            idx, w, h, if RAW_SUPPORTED_FORMATS.contains(&ext.as_str()) { "embedded-jpeg" } else { "full_res_image" }
        );
        self.eye_no_face = false;

        let tx = self.eye_tx.clone();
        std::thread::spawn(move || {
            let positions = eye_focus::detect_from_gray(gray, w, h, scale_inv);
            log::info!("detection thread: found {} position(s) for idx={}", positions.len(), idx);
            let _ = tx.send((idx, positions));
        });
        self.eye_detection_pending = true;
    }

    /// Apply the current eye zoom using `self.eye_positions[self.eye_index]`.
    fn apply_eye_zoom(&mut self) {
        let Some(&eye_pos) = self.eye_positions.get(self.eye_index) else { return };
        self.zoom = eye_focus::EYE_ZOOM;
        self.offset = eye_focus::eye_zoom_offset(eye_pos, eye_focus::EYE_ZOOM, self.last_view_size);
        self.is_scaled_to_fit = false;
        self.velocity = Vec2::ZERO;
        log::info!(
            "Eye focus: eye {} of {} at ({:.0}, {:.0})",
            self.eye_index + 1,
            self.eye_positions.len(),
            eye_pos.x,
            eye_pos.y,
        );
    }

    /// Drain the eye detection channel. Called every frame from `logic()`.
    fn poll_eye_detection(&mut self, ctx: &egui::Context) {
        while let Ok((idx, positions)) = self.eye_rx.try_recv() {
            self.eye_detection_pending = false;
            // Discard stale results for images the user has already navigated away from.
            if idx != self.current_index {
                continue;
            }
            if positions.is_empty() {
                log::info!("Eye focus: no faces detected.");
                self.eye_no_face = true;
            } else {
                self.eye_no_face = false;
                self.eye_positions = positions;
                self.eye_positions_for = Some(idx);
                self.eye_index = 0;
                self.apply_eye_zoom();
            }
            ctx.request_repaint();
        }
        // Keep the event loop ticking while detection is in flight so the result
        // is applied as soon as it arrives — same pattern as poll_prefetch.
        if self.eye_detection_pending {
            ctx.request_repaint();
        }
    }

    fn handle_keyboard_input(&mut self, ctx: &egui::Context) {

        let events = ctx.input(|i| i.events.clone());
        // Iterate over all events that occurred this frame.
        for event in &events {
            match event {
                egui::Event::Copy => {
                    log::info!("Copying image to clipboard...");
                    self.copy_to_clipboard();
                }
                // '?' — toggle EXIF overlay.  Detected via Text event so it
                // works on any keyboard layout (German '?' is Shift+ß, not Shift+/).
                egui::Event::Text(t) if t == "?" => {
                    self.show_exif_overlay = !self.show_exif_overlay;
                    if self.show_exif_overlay {
                        self.exif_for_index = None; // force refresh
                        self.load_exif_for_current();
                    }
                }
                // 'e' — eye focus. Caught via Text event (same as '?') as a
                // robust fallback in case key_pressed misses it.
                egui::Event::Text(t) if t == "e" || t == "E" => {
                    log::info!("E via Text event");
                    self.cycle_eye_focus();
                }
                _ => {}
            }
        }

        let shift = ctx.input(|i| i.modifiers.shift);
        let alt   = ctx.input(|i| i.modifiers.alt);

        // Arrow keys — behaviour depends on modifiers and whether we're in compare mode
        let in_compare = self.compare_set.len() > 1;
        if ctx.input(|i| i.key_pressed(egui::Key::ArrowRight)) {
            if shift && alt {
                self.compare_swap_slot(true);
            } else if shift {
                self.compare_shift_right(ctx);
            } else if alt {
                self.compare_window_shift(true, ctx);
            } else if in_compare {
                self.compare_move_focus(1);
            } else {
                self.next_image(ctx);
            }
        }
        if ctx.input(|i| i.key_pressed(egui::Key::ArrowLeft)) {
            if shift && alt {
                self.compare_swap_slot(false);
            } else if shift {
                self.compare_shift_left(ctx);
            } else if alt {
                self.compare_window_shift(false, ctx);
            } else if in_compare {
                self.compare_move_focus(-1);
            } else {
                self.prev_image(ctx);
            }
        }
        if ctx.input(|i| i.key_pressed(egui::Key::Home)) {
            self.first_image(ctx);
        }
        if ctx.input(|i| i.key_pressed(egui::Key::End)) {
            self.last_image(ctx);
        }
        if ctx.input(|i| i.key_pressed(egui::Key::F1))
            || (ctx.input(|i| i.key_pressed(egui::Key::H)) && !shift && !alt)
        {
            self.show_help_overlay = !self.show_help_overlay;
        }
        if ctx.input(|i| i.key_pressed(egui::Key::Escape)) {
            if self.show_help_overlay {
                self.show_help_overlay = false;
            } else {
                ctx.send_viewport_cmd(egui::ViewportCommand::Close);
            }
        }
        if ctx.input(|i| i.key_pressed(egui::Key::F)) {
            self.is_fullscreen = !self.is_fullscreen;
        }
        if ctx.input(|i| i.key_pressed(egui::Key::Enter)) {
            self.is_scaled_to_fit = !self.is_scaled_to_fit;
        }
        if ctx.input(|i| i.key_pressed(egui::Key::Delete)) {
            self.show_delete_confirmation = true;
        }
        if ctx.input(|i| i.key_pressed(egui::Key::I)) {
            self.show_info_overlay = !self.show_info_overlay;
            if self.show_info_overlay {
                self.exif_for_index = None; // force refresh
                self.load_exif_for_current();
            }
        }
        // E: zoom to detected eye / cycle to next eye (reroll)
        if ctx.input(|i| i.key_pressed(egui::Key::E)) && !shift && !alt {
            log::info!("E key pressed");
            self.cycle_eye_focus();
        }
        // Invalidate EXIF cache whenever the displayed image changes
        if self.exif_for_index != Some(self.current_index) && (self.show_exif_overlay || self.show_info_overlay) {
            self.load_exif_for_current();
        }
        // R / Shift+R: toggle reference image (compare mode)
        if ctx.input(|i| i.key_pressed(egui::Key::R)) && !alt {
            let focused_pos = self.compare_set.get(self.compare_focus).copied();
            if let Some(pos) = focused_pos {
                if self.reference_image == Some(pos) {
                    self.reference_image = None;
                } else {
                    self.reference_image = Some(pos);
                }
            }
        }
        // C: toggle culling mode
        if ctx.input(|i| i.key_pressed(egui::Key::C)) && !shift && !alt {
            if self.is_culling_mode {
                self.exit_culling_mode();
            } else {
                self.enter_culling_mode(ctx);
            }
        }
        // Alt+Up/Down: adjust culling threshold (only while in culling mode)
        if self.is_culling_mode && alt && !shift {
            if ctx.input(|i| i.key_pressed(egui::Key::ArrowUp)) {
                if self.culling_min_stars < 5 {
                    self.culling_min_stars += 1;
                    self.rebuild_culling_indices();
                    self.apply_culling_to_compare_set(ctx);
                    if self.compare_set.len() <= 1
                        && !self.culling_indices.is_empty()
                        && !self.culling_indices.contains(&self.current_index)
                    {
                        let first = self.culling_indices[0];
                        self.load_image_at_index(first, ctx);
                    }
                }
            }
            if ctx.input(|i| i.key_pressed(egui::Key::ArrowDown)) {
                if self.culling_min_stars > 1 {
                    self.culling_min_stars -= 1;
                    self.rebuild_culling_indices();
                }
            }
        }
        // `.`: drop the focused image from the compare set
        if ctx.input(|i| i.key_pressed(egui::Key::Period)) && !shift && !alt {
            if self.compare_set.len() > 1 {
                let slot      = self.compare_focus;
                let order_pos = self.compare_set[slot];
                self.undo_stack.push(UndoAction::DropFromCompareSet { order_pos, slot });
                self.compare_set.remove(slot);
                self.compare_images.remove(&order_pos);
                if self.reference_image == Some(order_pos) {
                    self.reference_image = None;
                }
                if self.compare_focus >= self.compare_set.len() {
                    self.compare_focus = self.compare_set.len() - 1;
                }
                self.current_index = self.compare_set[self.compare_focus];
            }
        }
        // `u`: undo last action
        if ctx.input(|i| i.key_pressed(egui::Key::U)) && !shift && !alt {
            if let Some(action) = self.undo_stack.pop() {
                match action {
                    UndoAction::DropFromCompareSet { order_pos, slot } => {
                        let insert_at = slot.min(self.compare_set.len());
                        self.compare_set.insert(insert_at, order_pos);
                        self.compare_focus = insert_at;
                        self.current_index = order_pos;
                        self.ensure_image_loaded(order_pos, ctx);
                    }
                    UndoAction::SetRating { path, previous_stars } => {
                        match previous_stars {
                            Some(s) => { self.ratings.insert(path, s); }
                            None    => { self.ratings.remove(&path); }
                        }
                        self.save_ratings();
                        if self.is_culling_mode {
                            self.rebuild_culling_indices();
                        }
                    }
                }
            }
        }
        // Number keys 0-5 set the star rating only while the overlay is visible
        if self.show_info_overlay {
            let rating_keys = [
                (egui::Key::Num0, 0u8),
                (egui::Key::Num1, 1),
                (egui::Key::Num2, 2),
                (egui::Key::Num3, 3),
                (egui::Key::Num4, 4),
                (egui::Key::Num5, 5),
            ];
            for (key, stars) in rating_keys {
                if ctx.input(|i| i.key_pressed(key)) {
                    if let Some(path) = self.image_files.get(self.image_order[self.current_index]) {
                        let path = path.clone();
                        let previous_stars = self.ratings.get(&path).copied();
                        self.undo_stack.push(UndoAction::SetRating { path: path.clone(), previous_stars });
                        self.ratings.insert(path, stars);
                        self.save_ratings();
                        // In culling mode: rebuild filter and advance if image fell below threshold
                        if self.is_culling_mode {
                            self.rebuild_culling_indices();
                            if stars < self.culling_min_stars && !self.culling_indices.is_empty() {
                                let saved = self.current_index;
                                let next = self.culling_indices.iter()
                                    .find(|&&p| p > saved)
                                    .or_else(|| self.culling_indices.first())
                                    .copied();
                                if let Some(pos) = next {
                                    self.load_image_at_index(pos, ctx);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

impl eframe::App for ImageViewerApp {

    /// Called before every `ui` call, and also when the window is hidden but
    /// `request_repaint` fired (eframe 0.34 API).  Non-UI background work goes here.
    fn logic(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.poll_prefetch(ctx);
        self.poll_eye_detection(ctx);
    }

fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
    let ctx = ui.ctx().clone();
    let is_currently_fullscreen = ctx.input(|i| i.viewport().fullscreen.unwrap_or(false));
    if self.is_fullscreen != is_currently_fullscreen {
        ctx.send_viewport_cmd(egui::ViewportCommand::Fullscreen(self.is_fullscreen));
    }

    self.handle_keyboard_input(&ctx);

    egui::CentralPanel::default()
        .frame(egui::Frame::default().fill(Color32::from_rgb(20, 20, 20)))
        .show_inside(ui, |ui| {
            let available_rect = ui.available_rect_before_wrap();
            self.last_view_size = available_rect.size();

            if self.compare_set.len() > 1 {
                // ── COMPARE MODE ──────────────────────────────────────────
                let n = self.compare_set.len();
                let col_w = available_rect.width() / n as f32;
                let compare_set = self.compare_set.clone();

                // Enter resets zoom and all pan offsets
                if self.is_scaled_to_fit {
                    self.compare_zoom = 1.0;
                    self.compare_offsets.clear();
                    self.is_scaled_to_fit = false;
                }

                // Input: drag pans only the hovered column; scroll zooms all columns in sync
                let response = ui.allocate_rect(available_rect, egui::Sense::click_and_drag());

                if response.clicked_by(egui::PointerButton::Middle) {
                    self.compare_zoom = 1.0;
                    self.compare_offsets.clear();
                }

                if response.dragged_by(egui::PointerButton::Primary) {
                    if let Some(hover_pos) = response.hover_pos() {
                        let col_idx = ((hover_pos.x - available_rect.min.x) / col_w)
                            .floor().clamp(0.0, (n - 1) as f32) as usize;
                        if let Some(&order_pos) = compare_set.get(col_idx) {
                            *self.compare_offsets.entry(order_pos).or_insert(Vec2::ZERO) += response.drag_delta();
                        }
                    }
                }

                if let Some(hover_pos) = response.hover_pos() {
                    let scroll = ui.input(|i| i.smooth_scroll_delta.y);
                    if scroll != 0.0 {
                        let col_idx = ((hover_pos.x - available_rect.min.x) / col_w)
                            .floor().clamp(0.0, (n - 1) as f32) as usize;
                        let cursor_in_col = Vec2::new(
                            hover_pos.x - (available_rect.min.x + col_idx as f32 * col_w + col_w / 2.0),
                            hover_pos.y - available_rect.center().y,
                        );
                        let old_zoom = self.compare_zoom;
                        let speed = if scroll > 0.0 { 400.0 } else { 200.0 };
                        self.compare_zoom = (self.compare_zoom + (scroll / speed) * self.compare_zoom).max(0.8);
                        let zoom_ratio = self.compare_zoom / old_zoom;

                        // Zoom hovered column toward the cursor
                        if let Some(&order_pos) = compare_set.get(col_idx) {
                            let offset = self.compare_offsets.entry(order_pos).or_insert(Vec2::ZERO);
                            let image_pos = (cursor_in_col - *offset) / old_zoom;
                            *offset = cursor_in_col - image_pos * self.compare_zoom;
                        }
                        // Scale all other offsets proportionally to preserve their relative pan
                        for (i, &op) in compare_set.iter().enumerate() {
                            if i != col_idx {
                                *self.compare_offsets.entry(op).or_insert(Vec2::ZERO) *= zoom_ratio;
                            }
                        }
                        ctx.request_repaint();
                    }
                }

                for (slot, &order_pos) in compare_set.iter().enumerate() {
                    let col_rect = Rect::from_min_max(
                        Pos2::new(available_rect.min.x + slot as f32 * col_w, available_rect.min.y),
                        Pos2::new(available_rect.min.x + (slot + 1) as f32 * col_w, available_rect.max.y),
                    );
                    // Clip each column so zoomed/panned images don't bleed into neighbours
                    let painter = ui.painter().with_clip_rect(col_rect);

                    if let Some(image) = self.compare_images.get(&order_pos) {
                        let frs = Vec2::new(image.full_res_image.width() as f32, image.full_res_image.height() as f32);
                        let aspect = frs.x / frs.y;
                        let col_aspect = col_rect.width() / col_rect.height();
                        let fit_size = if aspect > col_aspect {
                            Vec2::new(col_rect.width(), col_rect.width() / aspect)
                        } else {
                            Vec2::new(col_rect.height() * aspect, col_rect.height())
                        };
                        let offset = self.compare_offsets.get(&order_pos).copied().unwrap_or(Vec2::ZERO);
                        let scaled_size = fit_size * self.compare_zoom;
                        let image_rect = Rect::from_center_size(
                            col_rect.center() + offset,
                            scaled_size,
                        );
                        painter.image(image.preview_texture.id(), image_rect,
                            Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0)), Color32::WHITE);
                    } else {
                        // Image is still loading — show a spinner in the column centre.
                        let spinner_rect = Rect::from_center_size(col_rect.center(), Vec2::splat(32.0));
                        ui.scope_builder(egui::UiBuilder::new().max_rect(spinner_rect), |ui| { ui.spinner(); });
                    }

                    // Column divider
                    if slot > 0 {
                        let x = col_rect.min.x;
                        ui.painter().line_segment(
                            [Pos2::new(x, available_rect.min.y), Pos2::new(x, available_rect.max.y)],
                            egui::Stroke::new(1.0, Color32::from_gray(50)),
                        );
                    }
                    // Focus highlight
                    if slot == self.compare_focus {
                        ui.painter().rect_stroke(col_rect, 0.0,
                            egui::Stroke::new(2.0, Color32::from_white_alpha(60)),
                            egui::StrokeKind::Inside);
                    }
                    // Reference image indicator — amber border + "REF" badge
                    if self.reference_image == Some(order_pos) {
                        ui.painter().rect_stroke(col_rect, 0.0,
                            egui::Stroke::new(2.0, Color32::from_rgb(255, 180, 0)),
                            egui::StrokeKind::Inside);
                        let hud_offset = if self.is_culling_mode { 34.0 } else { 0.0 };
                        let badge_pos = Pos2::new(col_rect.min.x + 8.0, col_rect.min.y + hud_offset + 6.0);
                        let badge_rect = Rect::from_min_size(badge_pos, Vec2::new(34.0, 18.0));
                        ui.painter().rect_filled(badge_rect, 3.0, Color32::from_rgb(255, 180, 0));
                        ui.painter().text(
                            badge_rect.center(), egui::Align2::CENTER_CENTER, "REF",
                            egui::FontId::proportional(11.0), Color32::BLACK);
                    }
                }

                response.context_menu(|ui| {
                    if ui.checkbox(&mut self.is_fullscreen, "Fullscreen (F)").clicked() { ui.close(); }
                    if ui.checkbox(&mut self.is_randomized, "Random order").clicked() {
                        self.toggle_random_order();
                        ui.close();
                    }
                });

            } else if let Some(image) = self.compare_images.get_mut(&self.current_index) {
                // ── SINGLE IMAGE MODE ──────────────────────────────────────
                let response = ui.allocate_rect(available_rect, egui::Sense::click_and_drag());
                let full_res_size = Vec2::new(image.full_res_image.width() as f32, image.full_res_image.height() as f32);

                // Handle Scale to Fit
                if self.is_scaled_to_fit {
                    let aspect_ratio = full_res_size.x / full_res_size.y;
                    let available_aspect = available_rect.width() / available_rect.height();
                    let mut fit_size = available_rect.size();
                    if aspect_ratio > available_aspect {
                        fit_size.y = fit_size.x / aspect_ratio;
                    } else {
                        fit_size.x = fit_size.y * aspect_ratio;
                    }
                    self.zoom = fit_size.x / full_res_size.x;
                    self.offset = (available_rect.size() - fit_size) / 2.0;
                    self.velocity = Vec2::ZERO;
                }

                let mut is_interacting = false;

                // Middle click resets zoom and pan
                if response.clicked_by(egui::PointerButton::Middle) {
                    self.is_scaled_to_fit = true;
                    self.velocity = Vec2::ZERO;
                }

                // Handle Dragging & Inertia
                if response.dragged_by(egui::PointerButton::Primary) {
                    let delta = response.drag_delta();
                    self.offset += delta;
                    self.velocity = delta;
                    self.is_scaled_to_fit = false;
                    is_interacting = true;
                } else {
                    self.offset += self.velocity;
                }

                // Handle Zooming
                if let Some(hover_pos) = response.hover_pos() {
                    let scroll = ui.input(|i| i.smooth_scroll_delta.y);
                    if scroll != 0.0 {
                        let old_zoom = self.zoom;
                        // Zoom in is slower (divisor 400) than zoom out (divisor 200)
                        let speed = if scroll > 0.0 { 400.0 } else { 200.0 };
                        let zoom_delta = (scroll / speed) * self.zoom;
                        // Floor at 80% of the fit-to-screen zoom so the image never
                        // shrinks to a tiny dot when scrolling out
                        let fit_zoom = (available_rect.width() / full_res_size.x)
                            .min(available_rect.height() / full_res_size.y);
                        self.zoom = (self.zoom + zoom_delta).max(fit_zoom * 0.8);
                        let image_coords = (hover_pos - available_rect.min - self.offset) / old_zoom;
                        self.offset -= image_coords * (self.zoom - old_zoom);
                        self.is_scaled_to_fit = false;
                        self.velocity = Vec2::ZERO;
                        is_interacting = true;
                    }
                }

                // Bouncing & Constraints
                if !self.is_scaled_to_fit && !is_interacting {
                    let screen_size = available_rect.size();
                    let scaled_image_size = full_res_size * self.zoom;
                    let friction = 0.92;
                    let tension = 0.06;
                    let damping = 0.65;
                    let handle_axis = |offset: &mut f32, velocity: &mut f32, view_dim: f32, img_dim: f32| {
                        let target_pos;
                        let is_out_of_bounds;
                        if img_dim <= view_dim {
                            target_pos = (view_dim - img_dim) / 2.0;
                            is_out_of_bounds = (*offset - target_pos).abs() > 0.5;
                        } else {
                            let min = view_dim - img_dim;
                            let max = 0.0;
                            if *offset > max { target_pos = max; is_out_of_bounds = true; }
                            else if *offset < min { target_pos = min; is_out_of_bounds = true; }
                            else { target_pos = *offset; is_out_of_bounds = false; }
                        }
                        if is_out_of_bounds {
                            let displacement = target_pos - *offset;
                            *velocity += displacement * tension;
                            *velocity *= damping;
                        } else {
                            *velocity *= friction;
                        }
                    };
                    handle_axis(&mut self.offset.x, &mut self.velocity.x, screen_size.x, scaled_image_size.x);
                    handle_axis(&mut self.offset.y, &mut self.velocity.y, screen_size.y, scaled_image_size.y);
                    if self.velocity.length_sq() > 0.01 { ctx.request_repaint(); }
                    else { self.velocity = Vec2::ZERO; }
                }
                if self.velocity.length_sq() > 0.1 { ctx.request_repaint(); }
                else { self.velocity = Vec2::ZERO; }

                let preview_size = image.preview_texture.size_vec2();
                let preview_scale = preview_size.x / full_res_size.x;
                let show_tiles = image.needs_tiling && self.zoom > preview_scale;

                if !show_tiles {
                    if !image.tile_cache.is_empty() {
                        log::debug!("Zoomed out, clearing tile cache of {} textures.", image.tile_cache.len());
                        image.tile_cache.clear();
                    }
                    let scaled_size = full_res_size * self.zoom;
                    let image_rect = Rect::from_min_size(available_rect.min + self.offset, scaled_size);
                    ui.painter().image(image.preview_texture.id(), image_rect,
                        Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0)), Color32::WHITE);
                } else {
                    let screen_offset_in_image_pixels = (available_rect.min - (available_rect.min + self.offset)) / self.zoom;
                    let screen_size_in_image_pixels = available_rect.size() / self.zoom;
                    let visible_image_rect = Rect::from_min_size(
                        Pos2::new(screen_offset_in_image_pixels.x, screen_offset_in_image_pixels.y),
                        screen_size_in_image_pixels,
                    );
                    let min_col_f = visible_image_rect.min.x / TILE_SIZE as f32;
                    let max_col_f = visible_image_rect.max.x / TILE_SIZE as f32;
                    let min_row_f = visible_image_rect.min.y / TILE_SIZE as f32;
                    let max_row_f = visible_image_rect.max.y / TILE_SIZE as f32;
                    let num_cols = (image.full_res_image.width() + TILE_SIZE - 1) / TILE_SIZE;
                    let num_rows = (image.full_res_image.height() + TILE_SIZE - 1) / TILE_SIZE;
                    let row_start = (min_row_f.floor() as i32).max(0) as usize;
                    let row_end   = (max_row_f.ceil()  as i32).max(0) as usize;
                    let col_start = (min_col_f.floor() as i32).max(0) as usize;
                    let col_end   = (max_col_f.ceil()  as i32).max(0) as usize;
                    for row in row_start..row_end.min(num_rows) {
                        for col in col_start..col_end.min(num_cols) {
                            let tile_key = (row, col);
                            let (texture_id, tile_dims) = if let Some((texture, dims)) = image.tile_cache.get(&tile_key) {
                                (texture.id(), *dims)
                            } else {
                                let x_start = col * TILE_SIZE;
                                let y_start = row * TILE_SIZE;
                                let tile_w = (x_start + TILE_SIZE).min(image.full_res_image.width()) - x_start;
                                let tile_h = (y_start + TILE_SIZE).min(image.full_res_image.height()) - y_start;
                                if tile_w == 0 || tile_h == 0 { continue; }
                                let mut tile_pixels = Vec::with_capacity(tile_w * tile_h);
                                for y in 0..tile_h {
                                    let src_y = y_start + y;
                                    let row_start_index = src_y * image.full_res_image.width();
                                    let row_slice_start = row_start_index + x_start;
                                    tile_pixels.extend_from_slice(&image.full_res_image.pixels[row_slice_start..row_slice_start + tile_w]);
                                }
                                let tile_image = ColorImage { size: [tile_w, tile_h], pixels: tile_pixels, source_size: Vec2::new(tile_w as f32, tile_h as f32) };
                                let texture = ctx.load_texture(format!("tile_{}_{}", row, col), tile_image, Default::default());
                                let id = texture.id();
                                let dims = [tile_w, tile_h];
                                image.tile_cache.insert(tile_key, (texture, dims));
                                (id, dims)
                            };
                            let tile_min_in_image_pixels = Pos2::new((col * TILE_SIZE) as f32, (row * TILE_SIZE) as f32);
                            let tile_min_on_screen = available_rect.min + self.offset + tile_min_in_image_pixels.to_vec2() * self.zoom;
                            let tile_dims_vec = Vec2::new(tile_dims[0] as f32, tile_dims[1] as f32);
                            let tile_screen_rect = Rect::from_min_size(tile_min_on_screen, tile_dims_vec * self.zoom);
                            if available_rect.intersects(tile_screen_rect) {
                                ui.painter().image(texture_id, tile_screen_rect,
                                    Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0)), Color32::WHITE);
                            }
                        }
                    }
                }

                let scaled_size = full_res_size * self.zoom;
                let image_screen_rect = Rect::from_min_size(available_rect.min + self.offset, scaled_size);
                if ui.clip_rect().intersects(image_screen_rect) {
                    ui.painter().add(Shape::Rect(RectShape::stroke(image_screen_rect, 0.0,
                        (1.0, Color32::from_gray(80)), egui::StrokeKind::Outside)));
                }

                response.context_menu(|ui| {
                    if ui.checkbox(&mut self.is_fullscreen, "Fullscreen (F)").clicked() { ui.close(); }
                    if ui.checkbox(&mut self.is_scaled_to_fit, "Scale to fit (Enter)").clicked() { ui.close(); }
                    if ui.checkbox(&mut self.is_randomized, "Random order").clicked() {
                        self.toggle_random_order();
                        ui.close();
                    }
                });

            } else if let Some(err) = &self.last_error {
                ui.centered_and_justified(|ui| {
                    ui.label(egui::RichText::new(err).color(Color32::RED).size(18.0));
                });
            } else if !self.image_files.is_empty() {
                // Image is queued in the background — show a spinner until it arrives.
                ui.centered_and_justified(|ui| {
                    ui.spinner();
                });
            }

            // Reference badge in single-image mode (compare mode draws its own badge per column)
            if self.reference_image == Some(self.current_index) && self.compare_set.len() <= 1 {
                let hud_offset = if self.is_culling_mode { 34.0 } else { 0.0 };
                let badge_pos = Pos2::new(available_rect.min.x + 8.0, available_rect.min.y + hud_offset + 6.0);
                let badge_rect = Rect::from_min_size(badge_pos, Vec2::new(34.0, 18.0));
                ui.painter().rect_filled(badge_rect, 3.0, Color32::from_rgb(255, 180, 0));
                ui.painter().text(badge_rect.center(), egui::Align2::CENTER_CENTER, "REF",
                    egui::FontId::proportional(11.0), Color32::BLACK);
            }

            // Eye-detection status badge — top-right corner
            let eye_badge: Option<(&str, Color32)> = if self.eye_detection_pending {
                Some(("EYE...", Color32::from_rgb(100, 180, 255)))
            } else if self.eye_no_face {
                Some(("NO FACE", Color32::from_rgb(220, 80, 80)))
            } else {
                None
            };
            if let Some((label, color)) = eye_badge {
                let hud_offset = if self.is_culling_mode { 34.0 } else { 0.0 };
                let badge_w = 56.0;
                let badge_pos = Pos2::new(available_rect.max.x - badge_w - 8.0, available_rect.min.y + hud_offset + 6.0);
                let badge_rect = Rect::from_min_size(badge_pos, Vec2::new(badge_w, 18.0));
                ui.painter().rect_filled(badge_rect, 3.0, color);
                ui.painter().text(badge_rect.center(), egui::Align2::CENTER_CENTER, label,
                    egui::FontId::proportional(11.0), Color32::BLACK);
            }

            // Rating + EXIF info overlay — shown in both single and compare mode
            if self.show_info_overlay && !self.image_files.is_empty() {
                let current_path = &self.image_files[self.image_order[self.current_index]];
                let rating = *self.ratings.get(current_path).unwrap_or(&0);
                let stars: String = (1u8..=5).map(|i| if i <= rating { '★' } else { '☆' }).collect();

                // Compact EXIF line: tag name + optional prefix override
                const EXIF_INFO_TAGS: &[(&str, &str)] = &[
                    ("ExposureTime",     ""),
                    ("FNumber",          "f/"),
                    ("FocalLength",      ""),
                    ("LensSpecification",""),
                    ("ColorSpace",       ""),
                    ("WhiteBalance",     ""),
                ];
                let exif_line = EXIF_INFO_TAGS.iter()
                    .filter_map(|&(tag, prefix)| {
                        self.exif_data.iter()
                            .find(|(_, t, _)| t == tag)
                            .map(|(_, _, v)| format!("{prefix}{v}"))
                    })
                    .collect::<Vec<_>>()
                    .join("  ·  ");

                let stars_h = 48.0;
                let exif_h  = if exif_line.is_empty() { 0.0 } else { 26.0 };
                let overlay_height = stars_h + exif_h;
                let overlay_rect = Rect::from_min_max(
                    Pos2::new(available_rect.min.x, available_rect.max.y - overlay_height),
                    available_rect.max,
                );
                ui.painter().rect_filled(overlay_rect, 0.0, Color32::from_black_alpha(180));

                let stars_center = Pos2::new(
                    overlay_rect.center().x,
                    overlay_rect.min.y + stars_h / 2.0,
                );
                ui.painter().text(stars_center, egui::Align2::CENTER_CENTER,
                    &stars, egui::FontId::proportional(28.0), Color32::from_gray(220));

                if !exif_line.is_empty() {
                    let exif_center = Pos2::new(
                        overlay_rect.center().x,
                        overlay_rect.max.y - exif_h / 2.0,
                    );
                    ui.painter().text(exif_center, egui::Align2::CENTER_CENTER,
                        &exif_line, egui::FontId::proportional(13.0), Color32::from_gray(190));
                }
            }

            // Culling mode HUD — top bar
            if self.is_culling_mode {
                let hud_height = 34.0;
                let hud_rect = Rect::from_min_max(
                    available_rect.min,
                    Pos2::new(available_rect.max.x, available_rect.min.y + hud_height),
                );
                ui.painter().rect_filled(hud_rect, 0.0, Color32::from_black_alpha(160));
                let min_stars = self.culling_min_stars;
                let threshold: String = (1u8..=5).map(|i| if i <= min_stars { '★' } else { '☆' }).collect();
                let ref_suffix = if self.reference_image.is_some() { "  · REF" } else { "" };
                let label = format!(
                    "CULLING  {}  ·  {} / {} images{}",
                    threshold,
                    self.culling_indices.len(),
                    self.image_files.len(),
                    ref_suffix,
                );
                ui.painter().text(
                    hud_rect.center(),
                    egui::Align2::CENTER_CENTER,
                    &label,
                    egui::FontId::proportional(15.0),
                    Color32::from_gray(200),
                );
            }
        });
        
    // ── EXIF side panel ───────────────────────────────────────────────────────
    if self.show_exif_overlay {
        let path_str = self.image_files
            .get(self.image_order[self.current_index])
            .map(|p| p.file_name().unwrap_or_default().to_string_lossy().into_owned())
            .unwrap_or_default();

        let screen_rect = ctx.content_rect();
        let panel_w = 280.0;
        let panel_top_left = Pos2::new(screen_rect.max.x - panel_w, screen_rect.min.y);
        let panel_rect = Rect::from_min_max(panel_top_left, screen_rect.max);

        // Semi-transparent dark background drawn behind the content area
        ctx.layer_painter(egui::LayerId::new(egui::Order::Foreground, egui::Id::new("exif_bg")))
            .rect_filled(panel_rect, 0.0, Color32::from_rgba_unmultiplied(22, 22, 27, 215));

        egui::Area::new(egui::Id::new("exif_panel"))
            .fixed_pos(panel_top_left)
            .order(egui::Order::Foreground)
            .interactable(true)
            .show(&ctx, |ui| {
                egui::Frame::default()
                    .fill(Color32::TRANSPARENT)
                    .stroke(egui::Stroke::NONE)
                    .inner_margin(egui::Margin::same(14))
                    .show(ui, |ui| {
                        ui.set_width(panel_w - 28.0);
                        ui.set_min_height(screen_rect.height() - 28.0);
                        egui::ScrollArea::vertical().auto_shrink([false; 2]).show(ui, |ui| {
                            if self.exif_data.is_empty() {
                                ui.colored_label(Color32::from_gray(130), "No EXIF data found.");
                                return;
                            }
                            ui.label(egui::RichText::new(&path_str)
                                .color(Color32::from_gray(160))
                                .size(12.0));
                            ui.add_space(6.0);
                            let mut prev_ifd = "";
                            egui::Grid::new("exif_grid")
                                .num_columns(2)
                                .striped(true)
                                .min_col_width(80.0)
                                .show(ui, |ui| {
                                    for (ifd, tag, value) in &self.exif_data {
                                        if ifd.as_str() != prev_ifd {
                                            ui.end_row();
                                            ui.colored_label(
                                                Color32::from_gray(130),
                                                egui::RichText::new(ifd).strong().size(11.0),
                                            );
                                            ui.label("");
                                            ui.end_row();
                                            prev_ifd = ifd.as_str();
                                        }
                                        ui.label(egui::RichText::new(tag).color(Color32::from_gray(150)));
                                        ui.label(egui::RichText::new(value).color(Color32::from_gray(200)));
                                        ui.end_row();
                                    }
                                });
                        });
                    });
            });
    }

    // ── Keyboard help overlay (F1) ────────────────────────────────────────────
    if self.show_help_overlay {
        let mut open = self.show_help_overlay;
        egui::Window::new("Keyboard Shortcuts")
            .open(&mut open)
            .collapsible(false)
            .resizable(false)
            .anchor(egui::Align2::CENTER_CENTER, Vec2::ZERO)
            .show(&ctx, |ui| {
                // Each entry: (key label, description)
                // An entry with an empty key is rendered as a section header.
                const ENTRIES: &[(&str, &str)] = &[
                    ("Navigation",           ""),
                    ("← / →",               "Previous / next image"),
                    ("Home / End",           "First / last image"),
                    ("",                     ""),
                    ("View",                 ""),
                    ("Scroll",               "Zoom in / out"),
                    ("Drag",                 "Pan image"),
                    ("Middle click",         "Reset zoom & pan"),
                    ("Enter",                "Scale to fit"),
                    ("F",                    "Toggle fullscreen"),
                    ("E",                    "Zoom to detected eye / next eye"),
                    ("Ctrl+C",               "Copy image to clipboard"),
                    ("",                     ""),
                    ("Overlays",             ""),
                    ("I",                    "Rating & EXIF info bar"),
                    ("0–5",                  "Set star rating  (info bar visible)"),
                    ("?",                    "EXIF side panel"),
                    ("H / F1",               "This help screen"),
                    ("Escape",               "Close overlay / quit"),
                    ("",                     ""),
                    ("Compare mode",         ""),
                    ("← / →",               "Move focus between images"),
                    ("Shift+← / →",         "Extend compare set left / right"),
                    ("Alt+← / →",           "Shift compare window left / right"),
                    ("Shift+Alt+← / →",     "Reorder columns left / right"),
                    ("Scroll",               "Zoom all images in sync"),
                    ("Drag",                 "Pan focused column independently"),
                    ("Middle click",         "Reset zoom & pan for all columns"),
                    ("Enter",                "Reset zoom & pan for all columns"),
                    ("R",                    "Pin / unpin reference image"),
                    (".",                    "Drop focused image from compare set"),
                    ("U",                    "Undo"),
                    ("",                     ""),
                    ("Culling mode",         ""),
                    ("C",                    "Toggle culling mode"),
                    ("Alt+↑ / ↓",           "Raise / lower star threshold"),
                    ("",                     ""),
                    ("File",                 ""),
                    ("Delete",               "Delete current file"),
                ];

                egui::Grid::new("help_grid")
                    .num_columns(2)
                    .min_col_width(160.0)
                    .spacing([24.0, 4.0])
                    .show(ui, |ui| {
                        for &(key, desc) in ENTRIES {
                            if desc.is_empty() {
                                // Section header or blank spacer
                                if key.is_empty() {
                                    ui.label("");
                                    ui.label("");
                                } else {
                                    ui.label(egui::RichText::new(key)
                                        .strong()
                                        .color(Color32::from_gray(180)));
                                    ui.label("");
                                }
                            } else {
                                ui.label(egui::RichText::new(key)
                                    .color(Color32::from_gray(220))
                                    .monospace());
                                ui.label(egui::RichText::new(desc)
                                    .color(Color32::from_gray(160)));
                            }
                            ui.end_row();
                        }
                    });
            });
        self.show_help_overlay = open;
    }

    if self.show_delete_confirmation {
            let path = self.image_files.get(self.image_order[self.current_index]).cloned();
        egui::Window::new("Delete File")
            .collapsible(false)
            .resizable(false)
            .anchor(egui::Align2::CENTER_CENTER, Vec2::ZERO)
            .show(&ctx, |ui| {
                if let Some(path) = &path {
                ui.label(format!("Are you sure you want to delete '{}'?", path.display()));
                ui.add_space(10.0);
                ui.horizontal(|ui| {
                    if ui.button("Cancel").clicked() {
                        self.show_delete_confirmation = false;
                    }
                    if ui.button(egui::RichText::new("Delete").color(Color32::RED)).clicked() {
                        if let Err(e) = fs::remove_file(path) {
                            self.last_error = Some(format!("Failed to delete file: {}", e));
                        } else {
                            log::info!("Deleted file: {}", path.display());
                            let removed_order_index = self.image_order.remove(self.current_index);
                            self.image_files.remove(removed_order_index);
                            for order_idx in self.image_order.iter_mut() {
                                if *order_idx > removed_order_index {
                                    *order_idx -= 1;
                                }
                            }
                            if self.image_files.is_empty() {
                                ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                            } else {
                                self.current_index %= self.image_files.len();
                                self.load_image_at_index(self.current_index, &ctx);
                            }
                        }
                        self.show_delete_confirmation = false;
                    }
                });
                }
            });
        }
    }
}

// --- Image Loading & Helper Functions ---

/// Read the EXIF Orientation tag from `path` (returns None if absent or unreadable).
fn read_exif_orientation(path: &Path) -> Option<u32> {
    use exif::Tag;
    let file = fs::File::open(path).ok()?;
    let mut reader = BufReader::new(file);
    let exif = ExifReader::new().read_from_container(&mut reader).ok()?;
    exif.get_field(Tag::Orientation, In::PRIMARY)
        .and_then(|f| f.value.get_uint(0))
}

/// Rotate / flip `img` so that its pixels match the EXIF orientation stored in `path`.
fn apply_exif_orientation(img: DynamicImage, path: &Path) -> DynamicImage {
    match read_exif_orientation(path).unwrap_or(1) {
        2 => img.fliph(),
        3 => img.rotate180(),
        4 => img.flipv(),
        5 => img.rotate90().fliph(),
        6 => img.rotate90(),
        7 => img.rotate270().fliph(),
        8 => img.rotate270(),
        _ => img, // 1 = normal, anything else — leave as-is
    }
}

fn load_image(path: &Path) -> Result<LoadedImage, String> {
    let path_str = path.to_string_lossy();
    let extension = path.extension().and_then(|s| s.to_str()).unwrap_or("").to_lowercase();
    
    if ANIM_SUPPORTED_FORMATS.contains(&extension.as_str()) {
        let frames = load_animated_gif_frames(&path_str)?;
        let first_frame = frames.into_iter().next().map(|(img, _)| img).ok_or_else(|| "GIF has no frames".to_string())?;
        return Ok(LoadedImage::Static(first_frame));
    }

    log::info!("Loading image: {}", path_str);
    log::info!("Detected format based on extension: {}", extension);

    let dynamic_image = if RAW_SUPPORTED_FORMATS.contains(&extension.as_str()) {
        load_raw(&path_str)
    } else if FITS_SUPPORTED_FORMATS.contains(&extension.as_str()) {
        load_fits(&path_str)
    } else if JXL_SUPPORTED_FORMATS.contains(&extension.as_str()) {
        load_jxl(&path_str)
    } else {
        load_with_image_crate(&path_str)
    }?;

    // Apply EXIF orientation for formats that can carry EXIF data.
    // Skip FITS (no EXIF) and FITS-like scientific formats.
    let oriented = if !FITS_SUPPORTED_FORMATS.contains(&extension.as_str()) {
        apply_exif_orientation(dynamic_image, path)
    } else {
        dynamic_image
    };

    Ok(LoadedImage::Static(to_egui_color_image(oriented)))
}

fn load_jxl(path: &str) -> Result<DynamicImage, String> {
    log::info!("Loading JXL: {}", path);
    let file = fs::File::open(path).map_err(|e| format!("Failed to open JXL: {}", e))?;
    let reader = BufReader::new(file);
    let decoder = JxlDecoder::new(reader).map_err(|e| format!("Failed to create JXL decoder: {}", e))?;
    let dynamic_image: DynamicImage = DynamicImage::from_decoder(decoder).map_err(|e| format!("Failed to decode JXL: {}", e))?;
    log::info!("Loading image data: {}x{}", dynamic_image.width(), dynamic_image.height());

    Ok(dynamic_image)
}

fn load_animated_gif_frames(path: &str) -> Result<Vec<(ColorImage, Duration)>, String> {
    let file = fs::File::open(path).map_err(|e| format!("Failed to open GIF: {}", e))?;
    let reader = BufReader::new(file);
    let decoder = GifDecoder::new(reader).map_err(|e| format!("Failed to create GIF decoder: {}", e))?;
    let frames = decoder.into_frames().collect_frames().map_err(|e| format!("Failed to decode GIF frames: {}", e))?;

    Ok(frames
        .into_iter()
        .map(|frame| {
            let delay = Duration::from(frame.delay());
            let image_buffer = frame.into_buffer();
            let dims = image_buffer.dimensions();
            let color_image = ColorImage::from_rgba_unmultiplied([dims.0 as _, dims.1 as _], image_buffer.as_raw());
            (color_image, delay)
        })
        .collect())
}

fn to_egui_color_image(img: DynamicImage) -> ColorImage {
    let rgba = img.into_rgba8();
    let dims = rgba.dimensions();
    ColorImage::from_rgba_unmultiplied([dims.0 as _, dims.1 as _], rgba.as_raw())
}

fn load_with_image_crate(path: &str) -> Result<DynamicImage, String> {
    log::debug!("Loading with image-rs: {}", path);
    ImageReader::open(path)
        .map_err(|e| format!("Failed to open {}: {}", path, e))?
        .decode()
        .map_err(|e| format!("Failed to decode {}: {}", path, e))
}

/// Returns true when the JPEG byte slice carries an EXIF ColorSpace tag
/// with value 2 (AdobeRGB).  Reads from the already-loaded bytes — no I/O.
fn jpeg_is_adobe_rgb(data: &[u8]) -> bool {
    use exif::Tag;
    let mut cursor = Cursor::new(data);
    let Ok(exif) = ExifReader::new().read_from_container(&mut cursor) else { return false };
    exif.get_field(Tag::ColorSpace, In::PRIMARY)
        .and_then(|f| f.value.get_uint(0))
        .map(|v| v == 2)
        .unwrap_or(false)
}

/// Convert every pixel in `img` from AdobeRGB to sRGB.
///
/// Uses two precomputed LUTs (built once per call, ~200 µs):
///   • `de_gamma[256]`  — AdobeRGB γ=2.2 decode: u8 → linear f32
///   • `srgb_lut[4097]` — sRGB encode: linear f32 (scaled ×4096) → u8
///
/// Hot loop: 3 LUT lookups + 4 muls + 2 adds + 3 LUT lookups per pixel.
/// At 1080×1620 that is ≈1.75 M pixels, typically < 5 ms on a single core.
fn adobe_rgb_to_srgb(img: DynamicImage) -> DynamicImage {
    // AdobeRGB γ=2.2 decode: u8 → linear f32
    let mut de_gamma = [0.0f32; 256];
    for (i, v) in de_gamma.iter_mut().enumerate() {
        *v = (i as f32 / 255.0).powf(2.2);
    }

    // sRGB piecewise encode: linear f32 (× 4096, clamped to [0, 4096]) → u8
    let mut srgb_lut = [0u8; 4097];
    for (i, out) in srgb_lut.iter_mut().enumerate() {
        let v = i as f32 / 4096.0;
        let enc = if v <= 0.003_130_8 {
            v * 12.92
        } else {
            1.055 * v.powf(1.0 / 2.4) - 0.055
        };
        *out = (enc.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
    }

    // AdobeRGB linear → sRGB linear matrix (D65, via XYZ).
    // Rows 0 and 2 have two near-zero terms that are omitted.
    //   R_sRGB =  1.397715·R − 0.398186·G
    //   G_sRGB =  G                          (row 1 is identity)
    //   B_sRGB = −0.042918·G + 1.042939·B
    const R_R: f32 =  1.397_715;
    const R_G: f32 = -0.398_186;
    const B_G: f32 = -0.042_918;
    const B_B: f32 =  1.042_939;

    let encode = |v: f32| -> u8 {
        srgb_lut[(v * 4096.0).clamp(0.0, 4096.0) as usize]
    };

    let mut rgb = img.into_rgb8();
    for pixel in rgb.pixels_mut() {
        let [r, g, b] = pixel.0;
        let rl = de_gamma[r as usize];
        let gl = de_gamma[g as usize];
        let bl = de_gamma[b as usize];
        pixel.0 = [
            encode(R_R * rl + R_G * gl),
            encode(gl),
            encode(B_G * gl + B_B * bl),
        ];
    }
    DynamicImage::ImageRgb8(rgb)
}

fn load_raw(path: &str) -> Result<DynamicImage, String> {
    log::debug!("Loading RAW: {}", path);
    let p = Path::new(path);

    // Fast path: extract an embedded JPEG preview from the RAW container.
    // Cap at 4 MP (≈ 2688×1512) — enough for any display while keeping each
    // decoded ColorImage at ~10 MB instead of the ~45 MB a full Z7 II preview
    // would require.  Falls back to the smallest available preview if all
    // candidates exceed 4 MP.
    if let Some(jpeg_bytes) = raw_preview::extract_preview_jpeg_capped(p, 4_000_000) {
        log::debug!("Using embedded JPEG preview ({} bytes)", jpeg_bytes.len());
        if let Ok(img) = image::load_from_memory_with_format(&jpeg_bytes, image::ImageFormat::Jpeg) {
            if jpeg_is_adobe_rgb(&jpeg_bytes) {
                log::debug!("AdobeRGB embedded JPEG — converting to sRGB");
                return Ok(adobe_rgb_to_srgb(img));
            }
            return Ok(img);
        }
        log::warn!("Embedded JPEG decode failed — falling back to full RAW decode");
    }

    // Slow path: full RAW decode via imagepipe (demosaicing + colour processing).
    log::debug!("Full RAW decode for {}", path);
    let mut pipeline = imagepipe::Pipeline::new_from_file(path)
        .map_err(|e| format!("Failed to load RAW: {}", e))?;
    let decoded = pipeline.output_8bit(None)
        .map_err(|e| format!("Failed to process RAW: {}", e))?;
    image::RgbImage::from_raw(decoded.width as u32, decoded.height as u32, decoded.data)
        .map(DynamicImage::ImageRgb8)
        .ok_or_else(|| "Failed to create image from RAW data".to_string())
}

fn load_fits(path: &str) -> Result<DynamicImage, String> {
    log::debug!("Loading FITS: {}", path);
    let mut fits = rsf::Fits::open(Path::new(path)).map_err(|e| format!("FITS open error: {}", e))?;
    let hdu = fits.remove_hdu(0).ok_or_else(|| "FITS HDU error: failed to remove HDU".to_string())?;
    let data = hdu.to_parts().1.ok_or("No data in FITS HDU")?;

    let array = match data {
        rsf::Extension::Image(img) => rgb_to_grayscale(img.as_owned_f32_array()),
        _ => Err("No image data found in FITS".into()),
    }
    .map_err(|e| format!("FITS data conversion error: {}", e))?;

    let (height, width) = (array.shape()[0], array.shape()[1]);
    #[allow(deprecated)]
    let mut data_f32: Vec<f32> = array.into_raw_vec();

    let (min_val, max_val) = data_f32
        .par_iter()
        .fold(|| (f32::MAX, f32::MIN), |(min, max), &x| (min.min(x), max.max(x)))
        .reduce(|| (f32::MAX, f32::MIN), |(a_min, a_max), (b_min, b_max)| (a_min.min(b_min), a_max.max(b_max)));
    let scale = 255.0 / (max_val - min_val).max(1e-5);
    data_f32.par_iter_mut().for_each(|x| *x = (*x - min_val) * scale);

    let log_factor = 3000.0;
    let gamma = 1.5;
    let buffer: Vec<u8> = data_f32
        .par_iter()
        .map(|&x| {
            let log_scaled = 255.0 * (1.0 + log_factor * (x.clamp(0.0, 255.0) / 255.0)).ln() / (1.0 + log_factor).ln();
            ((log_scaled / 255.0).powf(gamma) * 255.0) as u8
        })
        .collect();

    image::ImageBuffer::<Luma<u8>, Vec<u8>>::from_raw(width as u32, height as u32, buffer)
        .map(DynamicImage::ImageLuma8)
        .ok_or_else(|| "Failed to create image from FITS data".to_string())
}

fn rgb_to_grayscale(rgb_image: Result<Array<f32, IxDyn>, Box<dyn Error>>) -> Result<Array2<f32>, Box<dyn Error>> {
    let rgb_array = rgb_image?;
    let shape = rgb_array.shape();
    if shape.len() != 3 || shape[2] != 3 {
        return Err("Invalid shape: Expected (H, W, 3)".into());
    }
    Ok(&rgb_array.slice(s![.., .., 0]) * 0.2989 + &rgb_array.slice(s![.., .., 1]) * 0.5870 + &rgb_array.slice(s![.., .., 2]) * 0.1140)
}

fn get_absolute_path(filename: &str) -> Result<PathBuf, String> {
    let path = Path::new(filename);
    if path.is_absolute() {
        Ok(path.to_path_buf())
    } else {
        env::current_dir()
            .map(|mut dir| {
                dir.push(path);
                dir
            })
            .map_err(|e| format!("Failed to get current dir: {}", e))
    }
}

fn downscale_color_image(image: ColorImage, max_size: usize) -> ColorImage {
    let size = image.size;
    let rgba_image = image::RgbaImage::from_raw(size[0] as u32, size[1] as u32, image.pixels.iter().flat_map(|c| c.to_array()).collect()).unwrap();
    let (width, height) = (rgba_image.width(), rgba_image.height());
    let new_dims = if width > max_size as u32 || height > max_size as u32 {
        let aspect_ratio = width as f32 / height as f32;
        if width > height { (max_size as u32, (max_size as f32 / aspect_ratio) as u32) } 
        else { ((max_size as f32 * aspect_ratio) as u32, max_size as u32) }
    } else { (width, height) };
    let resized_img = imageops::resize(&rgba_image, new_dims.0, new_dims.1, imageops::FilterType::Lanczos3);
    ColorImage::from_rgba_unmultiplied([resized_img.width() as _, resized_img.height() as _], resized_img.as_raw())
}

// --- Main Entry Point ---
fn main() -> Result<(), Box<dyn Error>> {
    env_logger::init();
    log::info!("LightningView {} starting", env!("CARGO_PKG_VERSION"));
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        println!("Usage: {} [/windowed] <imagefile>", args[0]);
        return Ok(());
    }
    
    let mut is_fullscreen = true;
    let mut image_file_arg = &args[1];

    if args[1].eq_ignore_ascii_case("/windowed") {
        if args.len() > 2 {
            is_fullscreen = false;
            image_file_arg = &args[2];
        } else {
            println!("Missing image file after /windowed");
            return Ok(());
        }
    }

    #[cfg(target_os = "windows")]
    {
        if image_file_arg.eq_ignore_ascii_case("/register") {
            return match register_urlhandler() {
                Ok(_) => {
                    println!("Success! Registered as image viewer.");
                    Ok(())
                }
                Err(err) => {
                    println!("Failed to register: {}", err);
                    Ok(())
                }
            };
        } else if image_file_arg.eq_ignore_ascii_case("/unregister") {
            unregister_urlhandler();
            println!("Unregistered as image viewer.");
            return Ok(());
        }
    }

    let initial_path = get_absolute_path(image_file_arg)?;

    let native_options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1280.0, 720.0])
            .with_min_inner_size([300.0, 200.0])
	    .with_app_id("lightningview"),
        ..Default::default()
    };

    eframe::run_native(
        "Lightning View (egui)",
        native_options,
        Box::new(|cc| Ok(Box::new(ImageViewerApp::new(cc, Some(initial_path), is_fullscreen)))),
    )?;

    Ok(())
}
