/// Ultra-fast embedded-JPEG extractor for TIFF-based RAW files
/// (NEF, CR2, ARW, DNG, ORF, PEF, RW2, SRW …) and Fuji RAF.
///
/// File-access strategy — memory-mapped I/O:
///   1. Open file and memory-map it (one syscall; pages loaded on demand by OS).
///   2. Read TIFF header from the map — pure in-memory arithmetic, no seeks.
///   3. Walk every IFD via pointer arithmetic over the mmap slice.
///   4. Collect ALL valid JPEG candidates (offset, length, score).
///   5. Caller picks smallest (thumbnail) or largest (preview).
///   6. Verify JPEG magic with a bounds-checked slice index, return the byte range.
///
/// The full RAW pixel data is never touched or allocated.
use std::{
    collections::HashSet,
    fs::File,
    path::Path,
};
use memmap2::Mmap;

// ── Public API ────────────────────────────────────────────────────────────────

/// Smallest embedded JPEG in the file — typically a 160×120 thumbnail.
/// Decodes in ~1 ms; use for instant first-paint while the full preview loads.
pub fn extract_thumbnail_jpeg(path: &Path) -> Option<Vec<u8>> {
    let mmap = mmap_file(path)?;
    let mut candidates = scan_tiff_candidates(&mmap)?;
    candidates.sort_by_key(|c| c.score);
    let best = candidates.into_iter().next()?; // smallest score
    slice_jpeg(&mmap, best.offset, best.length).map(|s| s.to_vec())
}

/// Largest embedded JPEG preview whose actual decoded pixel count ≤ `max_pixels`.
///
/// Reads the SOF (Start-Of-Frame) marker from each candidate JPEG to get real
/// dimensions, then picks the largest image that fits in budget.
///
/// Fallback: if every candidate exceeds `max_pixels`, returns the smallest one.
pub fn extract_preview_jpeg_capped(path: &Path, max_pixels: u64) -> Option<Vec<u8>> {
    let mmap = mmap_file(path)?;

    if let Some(mut candidates) = scan_tiff_candidates(&mmap) {
        candidates.sort_by_key(|c| c.score); // smallest first

        for candidate in candidates.iter().rev() {
            if let Some(slice) = slice_jpeg(&mmap, candidate.offset, candidate.length) {
                match jpeg_pixel_count(slice) {
                    Some(pixels) if pixels <= max_pixels => return Some(slice.to_vec()),
                    Some(_) => {} // over budget — try next smaller candidate
                    None    => {} // SOF unreadable — skip
                }
            }
        }
        // All parseable candidates exceeded max_pixels — fall back to smallest.
        if let Some(first) = candidates.first() {
            return slice_jpeg(&mmap, first.offset, first.length).map(|s| s.to_vec());
        }
    }

    // Fuji RAF — no size metadata at container level; always return it.
    find_jpeg_raf(&mmap)
}

/// Parse the actual image dimensions from a JPEG byte slice by finding the
/// SOF (Start-Of-Frame) segment.  Returns `Some(width × height)` on success,
/// `None` if the SOF could not be located.
fn jpeg_pixel_count(data: &[u8]) -> Option<u64> {
    let mut i = 0;
    while i + 1 < data.len() {
        if data[i] != 0xFF { i += 1; continue; }
        let marker = data[i + 1];
        i += 2;
        match marker {
            0xD8 | 0xD9 | 0x01 => {} // SOI, EOI, TEM — no length field
            0xC0 | 0xC1 | 0xC2 | 0xC3 | 0xC5 | 0xC6 | 0xC7
            | 0xC9 | 0xCA | 0xCB | 0xCD | 0xCE | 0xCF => {
                if i + 7 <= data.len() {
                    let h = u16::from_be_bytes([data[i + 3], data[i + 4]]) as u64;
                    let w = u16::from_be_bytes([data[i + 5], data[i + 6]]) as u64;
                    return Some(h * w);
                }
                return None;
            }
            0xFF => { i -= 1; } // padding byte — back up one
            _ => {
                if i + 2 > data.len() { break; }
                let seg_len = u16::from_be_bytes([data[i], data[i + 1]]) as usize;
                if seg_len < 2 { break; }
                i += seg_len;
            }
        }
    }
    None
}

/// Largest embedded JPEG preview — full-resolution preview for quality display.
/// Returns None if no JPEG preview exists in a recognised RAW container.
pub fn extract_embedded_jpeg_preview(path: &Path) -> Option<Vec<u8>> {
    let mmap = mmap_file(path)?;

    if let Some(mut candidates) = scan_tiff_candidates(&mmap) {
        candidates.sort_by_key(|c| c.score);
        if let Some(best) = candidates.into_iter().last() {
            return slice_jpeg(&mmap, best.offset, best.length).map(|s| s.to_vec());
        }
    }

    // Fuji RAF
    find_jpeg_raf(&mmap)
}

// ── Internal types ────────────────────────────────────────────────────────────

struct JpegCandidate {
    offset: u64,
    length: u64,
    score:  u64, // img_w × img_h, or byte-length if dimensions unknown
}

// ── mmap helper ───────────────────────────────────────────────────────────────

/// Open a file and memory-map it for read-only access.
///
/// # Safety
/// If the underlying file is truncated by another process while the map is
/// live, accessing the affected pages produces SIGBUS.  For read-only image
/// viewing of camera files this risk is negligible.
fn mmap_file(path: &Path) -> Option<Mmap> {
    let f = File::open(path).ok()?;
    // SAFETY: we only read; file is not expected to be modified concurrently.
    unsafe { Mmap::map(&f).ok() }
}

// ── Low-level byte helpers ────────────────────────────────────────────────────

#[inline] fn u16_le_be(b: &[u8], le: bool) -> u16 {
    if le { u16::from_le_bytes([b[0], b[1]]) } else { u16::from_be_bytes([b[0], b[1]]) }
}
#[inline] fn u32_le_be(b: &[u8], le: bool) -> u32 {
    if le { u32::from_le_bytes([b[0], b[1], b[2], b[3]]) } else { u32::from_be_bytes([b[0], b[1], b[2], b[3]]) }
}

/// Verify JPEG magic and return the byte slice at [offset, offset+length).
/// Returns None on bounds violation or missing FF D8 SOI.
fn slice_jpeg(data: &[u8], offset: u64, length: u64) -> Option<&[u8]> {
    if length < 2 { return None; }
    let start = offset as usize;
    let end   = start.checked_add(length as usize)?;
    if end > data.len() { return None; }
    let slice = &data[start..end];
    if slice[0] == 0xFF && slice[1] == 0xD8 { Some(slice) } else { None }
}

/// Parse one IFD from the mmap slice.
/// Returns (n_entries, entry_block, next_ifd_offset) or None on error.
fn parse_ifd(data: &[u8], ifd_off: usize, le: bool) -> Option<(usize, &[u8], usize)> {
    let cnt_end = ifd_off.checked_add(2)?;
    if cnt_end > data.len() { return None; }
    let n = u16_le_be(&data[ifd_off..], le) as usize;
    if n == 0 || n > 512 { return None; }
    let entries_start = ifd_off + 2;
    let entries_end   = entries_start.checked_add(n * 12)?;
    let next_ptr_end  = entries_end.checked_add(4)?;
    if next_ptr_end > data.len() { return None; }
    let entries = &data[entries_start..entries_end];
    let next    = u32_le_be(&data[entries_end..], le) as usize;
    Some((n, entries, next))
}

// ── TIFF IFD BFS scanner ──────────────────────────────────────────────────────

/// Walk every reachable IFD in the mmap slice and return all valid JPEG candidates.
/// Returns None if the data is not a recognised TIFF-based RAW.
fn scan_tiff_candidates(data: &[u8]) -> Option<Vec<JpegCandidate>> {
    if data.len() < 8 { return None; }

    let le = match &data[0..2] {
        b"II" => true,
        b"MM" => false,
        _ => return None,
    };
    match u16_le_be(&data[2..], le) {
        42 | 0x4F52 | 0x5352 | 0x0055 => {}
        _ => return None,
    }
    let first_ifd = u32_le_be(&data[4..], le) as usize;

    let mut queue:    Vec<usize>         = vec![first_ifd];
    let mut visited:  HashSet<usize>     = HashSet::new();
    let mut candidates: Vec<JpegCandidate> = Vec::new();

    while let Some(ifd_off) = queue.pop() {
        if ifd_off == 0 || !visited.insert(ifd_off) { continue; }

        let (n, entries, next_ifd) = match parse_ifd(data, ifd_off, le) {
            Some(v) => v,
            None    => continue,
        };

        let mut jpeg_off:    Option<u64> = None;
        let mut jpeg_len:    Option<u64> = None;
        let mut compression: u16         = 0;
        let mut strip_off:   Option<u64> = None;
        let mut strip_len:   Option<u64> = None;
        let mut img_w: u32 = 0;
        let mut img_h: u32 = 0;

        for i in 0..n {
            let e = i * 12;
            let tag = u16_le_be(&entries[e..],     le);
            let typ = u16_le_be(&entries[e + 2..], le);
            let cnt = u32_le_be(&entries[e + 4..], le) as usize;
            let val = &entries[e + 8..e + 12];

            let inline_u32 = |typ: u16, v: &[u8]| -> Option<u32> {
                match typ {
                    3 => Some(u16_le_be(v, le) as u32),
                    4 => Some(u32_le_be(v, le)),
                    _ => None,
                }
            };
            let val_u32 = u32_le_be(val, le);

            match tag {
                0x0100 => img_w       = inline_u32(typ, val).unwrap_or(0),
                0x0101 => img_h       = inline_u32(typ, val).unwrap_or(0),
                0x0103 => compression = u16_le_be(val, le),

                0x0111 if cnt == 1 => strip_off = inline_u32(typ, val).map(|v| v as u64),
                0x0117 if cnt == 1 => strip_len = inline_u32(typ, val).map(|v| v as u64),

                0x0201 => jpeg_off = Some(val_u32 as u64),
                0x0202 => jpeg_len = Some(val_u32 as u64),

                // SubIFD — Nikon preview JPEG lives here
                0x014A => {
                    if cnt == 1 {
                        queue.push(val_u32 as usize);
                    } else {
                        let list_off = val_u32 as usize;
                        let n_sub    = cnt.min(32);
                        let list_end = list_off + n_sub * 4;
                        if list_end <= data.len() {
                            for j in 0..n_sub {
                                queue.push(u32_le_be(&data[list_off + j * 4..], le) as usize);
                            }
                        }
                    }
                }

                0x8769 | 0x8825 | 0xA005 => queue.push(val_u32 as usize),
                _ => {}
            }
        }

        if next_ifd > 0 { queue.push(next_ifd); }

        let candidate = if let (Some(o), Some(l)) = (jpeg_off, jpeg_len) {
            Some((o, l))
        } else if matches!(compression, 6 | 7) {
            strip_off.zip(strip_len)
        } else {
            None
        };

        if let Some((off, len)) = candidate {
            if len < 2 { continue; }
            // Verify JPEG magic — pure slice index, no I/O
            let start = off as usize;
            if data.get(start..start + 2) == Some(&[0xFF, 0xD8]) {
                let score = if img_w > 0 && img_h > 0 {
                    img_w as u64 * img_h as u64
                } else {
                    len
                };
                candidates.push(JpegCandidate { offset: off, length: len, score });
            }
        }
    }

    if candidates.is_empty() { None } else { Some(candidates) }
}

// ── Fuji RAF ──────────────────────────────────────────────────────────────────

/// RAF header is 160 bytes; preview offset/length are at fixed positions.
/// Reads only the first 0x5C bytes, then slices directly to the preview data.
fn find_jpeg_raf(data: &[u8]) -> Option<Vec<u8>> {
    if data.len() < 0x5C { return None; }
    if !data.starts_with(b"FUJIFILM") { return None; }

    let off = u32::from_be_bytes([data[0x54], data[0x55], data[0x56], data[0x57]]) as u64;
    let len = u32::from_be_bytes([data[0x58], data[0x59], data[0x5A], data[0x5B]]) as u64;
    if off == 0 || len == 0 { return None; }

    slice_jpeg(data, off, len).map(|s| s.to_vec())
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    // ── helpers ──────────────────────────────────────────────────────────────

    /// Write `bytes` to a named temp file and return the path + temp-file handle
    /// (keep the handle alive so the file isn't deleted before the test ends).
    fn write_temp(bytes: &[u8]) -> (tempfile::NamedTempFile, std::path::PathBuf) {
        let mut f = tempfile::NamedTempFile::new().expect("temp file");
        f.write_all(bytes).expect("write");
        let path = f.path().to_path_buf();
        (f, path)
    }

    /// A 4-byte JPEG that passes the FF D8 magic check.
    /// SOI (FF D8) + EOI (FF D9) — the smallest possible syntactically valid JPEG.
    const MINIMAL_JPEG: &[u8] = &[0xFF, 0xD8, 0xFF, 0xD9];

    /// Build a little-endian TIFF with one IFD containing `n_entries` entries.
    /// Returns the full byte vector; JPEG data is appended at the end.
    ///
    /// IFD layout:
    ///   Offset 0 : "II" + 42 + 8       — 8-byte TIFF header (IFD at offset 8)
    ///   Offset 8 : u16 entry count      — n_entries
    ///   Offset 10: n_entries × 12 bytes — IFD entries (caller supplies them)
    ///   Offset 10 + n*12: u32 next IFD  — 0
    ///   Offset 10 + n*12 + 4: JPEG data
    fn build_tiff_le(entries: &[[u8; 12]], jpeg: &[u8]) -> Vec<u8> {
        let n = entries.len();
        let jpeg_offset: u32 = (8 + 2 + n * 12 + 4) as u32;
        let jpeg_len:   u32 = jpeg.len() as u32;

        let mut out = Vec::new();
        // Header
        out.extend_from_slice(b"II");
        out.extend_from_slice(&42u16.to_le_bytes());
        out.extend_from_slice(&8u32.to_le_bytes()); // first IFD at byte 8

        // IFD
        out.extend_from_slice(&(n as u16).to_le_bytes());
        for e in entries {
            out.extend_from_slice(e);
        }
        out.extend_from_slice(&0u32.to_le_bytes()); // next IFD = 0

        out.extend_from_slice(jpeg);

        // Fix up JPEGInterchangeFormat / JPEGInterchangeFormatLength values
        // if the caller left them as zero placeholders.
        let ifd_start = 10usize;
        for i in 0..n {
            let base = ifd_start + i * 12;
            let tag = u16::from_le_bytes([out[base], out[base + 1]]);
            if tag == 0x0201 {
                out[base + 8..base + 12].copy_from_slice(&jpeg_offset.to_le_bytes());
            }
            if tag == 0x0202 {
                out[base + 8..base + 12].copy_from_slice(&jpeg_len.to_le_bytes());
            }
        }

        out
    }

    /// One 12-byte IFD entry (little-endian TIFF).
    fn entry(tag: u16, typ: u16, count: u32, value: u32) -> [u8; 12] {
        let mut e = [0u8; 12];
        e[0..2].copy_from_slice(&tag.to_le_bytes());
        e[2..4].copy_from_slice(&typ.to_le_bytes());
        e[4..8].copy_from_slice(&count.to_le_bytes());
        e[8..12].copy_from_slice(&value.to_le_bytes());
        e
    }

    // ── raw_preview unit tests ────────────────────────────────────────────────

    #[test]
    fn extract_preview_from_tiff_jpeg_tags() {
        // TIFF with ImageWidth=640, ImageHeight=480, JPEGInterchangeFormat + Length.
        // Values for 0x0201/0x0202 are patched by build_tiff_le.
        let tiff = build_tiff_le(
            &[
                entry(0x0100, 3, 1, 640),  // ImageWidth  SHORT 640
                entry(0x0101, 3, 1, 480),  // ImageLength SHORT 480
                entry(0x0201, 4, 1, 0),    // JPEGInterchangeFormat (patched)
                entry(0x0202, 4, 1, 0),    // JPEGInterchangeFormatLength (patched)
            ],
            MINIMAL_JPEG,
        );
        let (_tmp, path) = write_temp(&tiff);
        let result = extract_embedded_jpeg_preview(&path);
        assert!(result.is_some(), "expected JPEG preview from TIFF");
        let jpeg = result.unwrap();
        assert_eq!(&jpeg[..2], &[0xFF, 0xD8], "result must begin with JPEG SOI");
        assert_eq!(jpeg.as_slice(), MINIMAL_JPEG);
    }

    #[test]
    fn extract_thumbnail_returns_smallest_candidate() {
        // Build a TIFF whose IFD chain has two JPEGs: one small (score = 160×120)
        // and one large (score = 640×480).  Thumbnail extractor picks the smaller one.

        let small_jpeg: &[u8] = MINIMAL_JPEG;
        let large_jpeg: &[u8] = &[0xFF, 0xD8, 0xFF, 0xD9, 0xFF, 0xD8, 0xFF, 0xD9];

        let ifd0_start: u32 = 8;
        let ifd0_size: u32 = 2 + 4 * 12 + 4; // 54 bytes
        let small_jpeg_offset: u32 = ifd0_start + ifd0_size;          // 62
        let ifd1_start: u32 = small_jpeg_offset + small_jpeg.len() as u32; // 66
        let ifd1_size: u32 = 2 + 4 * 12 + 4;
        let large_jpeg_offset: u32 = ifd1_start + ifd1_size;          // 120

        let mut out: Vec<u8> = Vec::new();
        // TIFF header
        out.extend_from_slice(b"II");
        out.extend_from_slice(&42u16.to_le_bytes());
        out.extend_from_slice(&ifd0_start.to_le_bytes());

        // IFD0: small image
        out.extend_from_slice(&4u16.to_le_bytes());
        out.extend_from_slice(&entry(0x0100, 3, 1, 160));
        out.extend_from_slice(&entry(0x0101, 3, 1, 120));
        out.extend_from_slice(&entry(0x0201, 4, 1, small_jpeg_offset));
        out.extend_from_slice(&entry(0x0202, 4, 1, small_jpeg.len() as u32));
        out.extend_from_slice(&ifd1_start.to_le_bytes()); // next IFD

        // small JPEG
        out.extend_from_slice(small_jpeg);

        // IFD1: large image
        out.extend_from_slice(&4u16.to_le_bytes());
        out.extend_from_slice(&entry(0x0100, 3, 1, 640));
        out.extend_from_slice(&entry(0x0101, 3, 1, 480));
        out.extend_from_slice(&entry(0x0201, 4, 1, large_jpeg_offset));
        out.extend_from_slice(&entry(0x0202, 4, 1, large_jpeg.len() as u32));
        out.extend_from_slice(&0u32.to_le_bytes()); // no next IFD

        // large JPEG
        out.extend_from_slice(&large_jpeg);

        let (_tmp, path) = write_temp(&out);

        let thumb = extract_thumbnail_jpeg(&path);
        assert!(thumb.is_some(), "thumbnail must be found");
        assert_eq!(thumb.unwrap().len(), small_jpeg.len(), "thumbnail must be the smaller JPEG");

        let preview = extract_embedded_jpeg_preview(&path);
        assert!(preview.is_some(), "preview must be found");
        assert_eq!(preview.unwrap().len(), large_jpeg.len(), "preview must be the larger JPEG");
    }

    #[test]
    fn non_tiff_file_returns_none() {
        let png_magic: &[u8] = &[0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A];
        let (_tmp, path) = write_temp(png_magic);
        assert!(extract_embedded_jpeg_preview(&path).is_none());
        assert!(extract_thumbnail_jpeg(&path).is_none());
    }

    #[test]
    fn empty_file_returns_none() {
        let (_tmp, path) = write_temp(&[]);
        assert!(extract_embedded_jpeg_preview(&path).is_none());
        assert!(extract_thumbnail_jpeg(&path).is_none());
    }

    #[test]
    fn corrupted_jpeg_offset_returns_none() {
        let tiff = build_tiff_le(
            &[
                entry(0x0201, 4, 1, 0xFFFF_FF00),
                entry(0x0202, 4, 1, 4),
            ],
            &[],
        );
        let (_tmp, path) = write_temp(&tiff);
        assert!(extract_embedded_jpeg_preview(&path).is_none());
    }

    #[test]
    fn orf_magic_accepted() {
        let jpeg = MINIMAL_JPEG;
        let jpeg_offset: u32 = 8 + 2 + 2 * 12 + 4;

        let mut out: Vec<u8> = Vec::new();
        out.extend_from_slice(b"II");
        out.extend_from_slice(&0x4F52u16.to_le_bytes()); // ORF magic
        out.extend_from_slice(&8u32.to_le_bytes());
        out.extend_from_slice(&2u16.to_le_bytes());
        out.extend_from_slice(&entry(0x0201, 4, 1, jpeg_offset));
        out.extend_from_slice(&entry(0x0202, 4, 1, jpeg.len() as u32));
        out.extend_from_slice(&0u32.to_le_bytes());
        out.extend_from_slice(jpeg);

        let (_tmp, path) = write_temp(&out);
        let result = extract_embedded_jpeg_preview(&path);
        assert!(result.is_some(), "ORF (magic 0x4F52) must be recognised");
    }

    #[test]
    fn raf_preview_extracted() {
        let jpeg = MINIMAL_JPEG;
        let preview_offset: u32 = 0x5C;
        let preview_len:    u32 = jpeg.len() as u32;

        let mut hdr = vec![0u8; 0x5C];
        hdr[..8].copy_from_slice(b"FUJIFILM");
        hdr[0x54..0x58].copy_from_slice(&preview_offset.to_be_bytes());
        hdr[0x58..0x5C].copy_from_slice(&preview_len.to_be_bytes());

        let mut data = hdr;
        data.extend_from_slice(jpeg);

        let (_tmp, path) = write_temp(&data);
        let result = extract_embedded_jpeg_preview(&path);
        assert!(result.is_some(), "RAF preview must be extracted");
        assert_eq!(&result.unwrap()[..2], &[0xFF, 0xD8]);
    }

    #[test]
    fn jpeg_pixel_count_parses_sof() {
        let mut jpeg: Vec<u8> = vec![0xFF, 0xD8];
        jpeg.extend_from_slice(&[0xFF, 0xE0, 0x00, 0x10]);
        jpeg.extend_from_slice(&[0x4A, 0x46, 0x49, 0x46, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x00, 0x01, 0x00, 0x00]);
        jpeg.extend_from_slice(&[0xFF, 0xC0, 0x00, 0x11, 0x08, 0x01, 0xE0, 0x02, 0x80]);

        let pixels = jpeg_pixel_count(&jpeg);
        assert_eq!(pixels, Some(640 * 480), "SOF0 must parse 640×480 = 307200 pixels");
    }

    #[test]
    fn capped_extractor_rejects_oversized_candidate() {
        let mut jpeg: Vec<u8> = vec![0xFF, 0xD8];
        jpeg.extend_from_slice(&[0xFF, 0xE0, 0x00, 0x10]);
        jpeg.extend_from_slice(&[0; 14]);
        jpeg.extend_from_slice(&[0xFF, 0xC0, 0x00, 0x11, 0x08, 0x04, 0x38, 0x07, 0x80]);
        jpeg.extend_from_slice(&[0xFF, 0xD9]);

        let jpeg_offset: u32 = 8 + 2 + 2 * 12 + 4;
        let mut tiff: Vec<u8> = Vec::new();
        tiff.extend_from_slice(b"II");
        tiff.extend_from_slice(&42u16.to_le_bytes());
        tiff.extend_from_slice(&8u32.to_le_bytes());
        tiff.extend_from_slice(&2u16.to_le_bytes());
        tiff.extend_from_slice(&entry(0x0201, 4, 1, jpeg_offset));
        tiff.extend_from_slice(&entry(0x0202, 4, 1, jpeg.len() as u32));
        tiff.extend_from_slice(&0u32.to_le_bytes());
        tiff.extend_from_slice(&jpeg);

        let (_tmp, path) = write_temp(&tiff);

        let result = extract_preview_jpeg_capped(&path, 3_000_000);
        assert!(result.is_some(), "under-cap: should return the only candidate");

        let fallback = extract_preview_jpeg_capped(&path, 1_000_000);
        assert!(fallback.is_some(), "over-cap: fallback must still return something");
    }

    #[test]
    fn big_endian_tiff_accepted() {
        let jpeg = MINIMAL_JPEG;
        let jpeg_offset: u32 = 8 + 2 + 2 * 12 + 4;

        let mut out: Vec<u8> = Vec::new();
        out.extend_from_slice(b"MM");
        out.extend_from_slice(&42u16.to_be_bytes());
        out.extend_from_slice(&8u32.to_be_bytes());
        out.extend_from_slice(&2u16.to_be_bytes());

        let mut e0 = [0u8; 12];
        e0[0..2].copy_from_slice(&0x0201u16.to_be_bytes());
        e0[2..4].copy_from_slice(&4u16.to_be_bytes());
        e0[4..8].copy_from_slice(&1u32.to_be_bytes());
        e0[8..12].copy_from_slice(&jpeg_offset.to_be_bytes());
        out.extend_from_slice(&e0);

        let mut e1 = [0u8; 12];
        e1[0..2].copy_from_slice(&0x0202u16.to_be_bytes());
        e1[2..4].copy_from_slice(&4u16.to_be_bytes());
        e1[4..8].copy_from_slice(&1u32.to_be_bytes());
        e1[8..12].copy_from_slice(&(jpeg.len() as u32).to_be_bytes());
        out.extend_from_slice(&e1);

        out.extend_from_slice(&0u32.to_be_bytes());
        out.extend_from_slice(jpeg);

        let (_tmp, path) = write_temp(&out);
        let result = extract_embedded_jpeg_preview(&path);
        assert!(result.is_some(), "big-endian TIFF (MM) must be recognised");
    }
}
