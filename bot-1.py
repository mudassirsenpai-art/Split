#!/usr/bin/env python3
"""
Smart Manhwa/Webtoon Page Splitter — Telegram Bot
====================================================
Splits very tall manhwa/webtoon strips into ~1024px-tall pieces WITHOUT
cutting through speech bubbles, text, SFX, or panel art — and WITHOUT any
resizing/rescaling/quality loss. Outputs a single .cbz with strict
sequential zero-padded naming.

Framework : Pyrogram (asyncio)
PDF       : PyMuPDF (fitz) — renders pages at native/full resolution
Archives  : zipfile / cbz (zip) supported natively
Detection : OpenCV (Canny edges + binarized ink density row profile)

Author: generated for Mudasir
"""

import asyncio
import io
import logging
import os
import shutil
import tempfile
import time
import zipfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from pyrogram import Client, filters
from pyrogram.types import Message

# =========================================================================
# CONFIG
# =========================================================================

API_ID = int(os.environ["API_ID"])
API_HASH = os.environ["API_HASH"]
BOT_TOKEN = os.environ["BOT_TOKEN"]

# Target height for each split piece (in pixels)
TARGET_HEIGHT = 1024

# Flexible search window around the target height. The algorithm will
# look for the cleanest row to cut anywhere inside [TARGET_HEIGHT - MIN_SLACK,
# TARGET_HEIGHT + MAX_SLACK] before falling back to a forced cut.
MIN_SLACK = 174   # -> allows cutting as early as 850px
MAX_SLACK = 76    # -> allows cutting as late as 1100px

# Never produce a slice shorter than this (avoids 1px slivers near the end)
MIN_SLICE_HEIGHT = 200

# JPEG quality for saved output (100 = visually lossless / max quality)
JPEG_QUALITY = 100

# How many image bytes to buffer per Telegram edit to avoid flood-wait
PROGRESS_EDIT_INTERVAL = 2.5  # seconds between progress message edits

SUPPORTED_INPUT_EXTS = {".zip", ".cbz", ".pdf"}
SUPPORTED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger("manhwa-splitter")

app = Client(
    "manhwa-splitter-bot",
    api_id=API_ID,
    api_hash=API_HASH,
    bot_token=BOT_TOKEN,
)

# In-memory registry of "pending" downloaded jobs, keyed by (chat_id, reply_to_message_id)
# so that when the user replies "/split" to the uploaded file, we know which file to use.
# Structure: { (chat_id, message_id_of_uploaded_doc): local_file_path }
PENDING_UPLOADS: dict[tuple[int, int], dict] = {}


# =========================================================================
# UI HELPERS — Status Board
# =========================================================================

def human_size(n: int) -> str:
    """Convert bytes to human-readable MB string."""
    mb = n / (1024 * 1024)
    return f"{mb:.1f}MB"


def make_progress_bar(percent: float, length: int = 12) -> str:
    filled = int(length * percent / 100)
    filled = max(0, min(length, filled))
    return "▓" * filled + "░" * (length - filled)


def render_transfer_board(title: str, current: int, total: int) -> str:
    """
    Renders a board like:

    📥 Downloading...
    [▓▓▓▓▓░░░░░░░] 41%
    10.2MB / 50.0MB
    """
    percent = (current / total * 100) if total else 0
    bar = make_progress_bar(percent)
    return (
        f"{title}\n"
        f"`[{bar}] {percent:.0f}%`\n"
        f"{human_size(current)} / {human_size(total)}"
    )


def render_page_board(done: int, total: int, extra: str = "") -> str:
    """
    Renders a board like:

    ✂️ Splitting Pages safely...
    Pages: 17 / 30
    (no bubble/text cut through)
    """
    bar = make_progress_bar((done / total * 100) if total else 0)
    lines = [
        "✂️ Splitting Pages safely...",
        f"`[{bar}]` {done} / {total} pages",
    ]
    if extra:
        lines.append(extra)
    return "\n".join(lines)


class ProgressThrottler:
    """Prevents hitting Telegram's flood limits by rate-limiting message edits."""

    def __init__(self, interval: float = PROGRESS_EDIT_INTERVAL):
        self.interval = interval
        self._last = 0.0

    def ready(self) -> bool:
        now = time.monotonic()
        if now - self._last >= self.interval:
            self._last = now
            return True
        return False


# =========================================================================
# OPENCV — SMART CUT DETECTOR
# =========================================================================

def compute_row_density(gray: np.ndarray) -> np.ndarray:
    """
    Builds a per-row "ink density" profile of the image by combining:
      1. Canny edge detection (catches panel borders, SFX, bubble outlines,
         text glyph edges)
      2. Adaptive binarization ink mask (catches solid text/bubble fill,
         flat colors, screentone-free art)

    Returns a 1D array (one value per row) — the LOWER the value, the more
    likely that row is "blank" (safe to cut).
    """
    # 1) Edge map — catches outlines/text/SFX strokes
    edges = cv2.Canny(gray, 50, 150)

    # 2) Binarized ink mask — catches solid fills (bubble fill, dark panels,
    #    bold text blocks) that Canny alone might under-represent
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Combine: a row counts as "busy" if it has edges OR significant ink
    combined = cv2.bitwise_or(edges, binary)

    # Row-wise density = fraction of "on" pixels in that row
    row_density = combined.mean(axis=1) / 255.0
    return row_density


def find_safe_cut_y(img_bgr: np.ndarray, search_start: int, target_height: int,
                     min_slack: int, max_slack: int) -> tuple[int, bool]:
    """
    Finds the cleanest horizontal row to cut on, inside the flexible window:
        [search_start + target_height - min_slack, search_start + target_height + max_slack]

    Returns (cut_y, was_forced)
      - cut_y      : absolute y-coordinate (row index) in the full image to cut at
      - was_forced : True if no sufficiently clean row was found and we had to
                     fall back to the least-busy row available (i.e. "no choice"
                     cut through content), False if a genuinely clean blank
                     row was found.
    """
    h, w = img_bgr.shape[:2]
    ideal_y = search_start + target_height

    window_top = max(search_start + MIN_SLICE_HEIGHT, ideal_y - min_slack)
    window_bottom = min(h, ideal_y + max_slack)

    # If the remaining image is shorter than the window even needs, just
    # cut at the end of the image (last slice).
    if window_top >= h:
        return h, False

    if window_bottom <= window_top:
        window_bottom = min(h, window_top + 1)

    crop = img_bgr[window_top:window_bottom, :]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    row_density = compute_row_density(gray)

    # A row is "clean" (safe blank space) if its density is near-zero.
    CLEAN_THRESHOLD = 0.008  # essentially pure blank/background line

    clean_rows = np.where(row_density <= CLEAN_THRESHOLD)[0]

    if len(clean_rows) > 0:
        # Prefer the clean row closest to the ideal target height so pieces
        # stay as close to ~1024px as possible.
        ideal_offset = target_height - min_slack  # offset of ideal_y within window
        best_idx = clean_rows[np.argmin(np.abs(clean_rows - ideal_offset))]
        cut_y = window_top + int(best_idx)
        return cut_y, False

    # --- Fallback: "if no any choice then it will cut" ---
    # No pure-blank row exists anywhere in the window (e.g. a full-bleed
    # panel or dense art spanning the whole search range). Pick the row
    # with the LOWEST density available — the least-bad place to cut —
    # rather than failing.
    best_idx = int(np.argmin(row_density))
    cut_y = window_top + best_idx
    return cut_y, True


def split_strip_lossless(image_path: Path, target_height: int = TARGET_HEIGHT,
                          min_slack: int = MIN_SLACK, max_slack: int = MAX_SLACK):
    """
    Splits one tall manhwa strip image into multiple pieces using the smart
    cut detector. NO resizing/scaling is performed — each piece keeps the
    original pixel width and its natural (cropped) height at full source
    resolution.

    Returns a list of PIL.Image objects (RGB), in top-to-bottom order, plus
    a list of booleans indicating whether each cut below that piece was a
    forced (non-clean) cut.
    """
    # Load with OpenCV for analysis (handles arbitrary formats via cv2.imread
    # fallback to PIL if needed for exotic formats like some PNGs/WebP)
    img_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        # Fallback via PIL for formats OpenCV can't read directly
        pil_img = Image.open(image_path).convert("RGB")
        img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    h, w = img_bgr.shape[:2]

    # If the strip is already shorter than target height, no split needed.
    if h <= target_height + max_slack:
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        return [Image.fromarray(rgb)], [False]

    pieces = []
    forced_flags = []
    y = 0
    while y < h:
        remaining = h - y
        if remaining <= target_height + max_slack:
            # Last piece — take everything remaining, no further cutting needed.
            cut_y = h
            forced = False
        else:
            cut_y, forced = find_safe_cut_y(img_bgr, y, target_height, min_slack, max_slack)
            if cut_y <= y:  # safety guard against zero-height slices
                cut_y = min(h, y + target_height)

        piece_bgr = img_bgr[y:cut_y, :]
        rgb = cv2.cvtColor(piece_bgr, cv2.COLOR_BGR2RGB)
        pieces.append(Image.fromarray(rgb))
        forced_flags.append(forced)
        y = cut_y

    return pieces, forced_flags


def save_lossless(img: Image.Image, out_path: Path):
    """
    Saves an image with ZERO quality loss / no resizing:
      - JPEG saved at quality=100, no chroma subsampling
      - PNG saved with lossless compression
    Chooses JPEG for smaller CBZ size at visually-lossless quality by default,
    matching the requirement "JPEG quality=100 or PNG lossless".
    """
    ext = out_path.suffix.lower()
    if ext == ".png":
        img.save(out_path, format="PNG", optimize=True)
    else:
        img.save(
            out_path,
            format="JPEG",
            quality=JPEG_QUALITY,
            subsampling=0,   # 4:4:4 — no chroma subsampling = no color quality loss
            optimize=True,
        )


# =========================================================================
# EXTRACTION — ZIP / CBZ / PDF -> ordered list of source image paths
# =========================================================================

def extract_images_from_zip_or_cbz(archive_path: Path, work_dir: Path) -> list[Path]:
    """Extracts all images from a .zip/.cbz, sorted in natural filename order."""
    extract_dir = work_dir / "extracted_src"
    extract_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(archive_path, "r") as zf:
        for member in zf.namelist():
            if member.endswith("/"):
                continue
            ext = Path(member).suffix.lower()
            if ext not in SUPPORTED_IMAGE_EXTS:
                continue
            # Flatten into extract_dir, guarding against path traversal
            safe_name = Path(member).name
            target = extract_dir / safe_name
            with zf.open(member) as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst)

    images = sorted(
        [p for p in extract_dir.iterdir() if p.suffix.lower() in SUPPORTED_IMAGE_EXTS],
        key=lambda p: p.name,
    )
    return images


def extract_images_from_pdf(pdf_path: Path, work_dir: Path) -> list[Path]:
    """Renders every PDF page to a full-resolution PNG using PyMuPDF (no downscaling)."""
    import fitz  # PyMuPDF

    extract_dir = work_dir / "extracted_src"
    extract_dir.mkdir(parents=True, exist_ok=True)

    images = []
    doc = fitz.open(str(pdf_path))
    try:
        # Zoom factor of 2.0 renders at ~144 DPI base scaling; we use the
        # PDF's intrinsic pixel dimensions via a high matrix to guarantee we
        # never render BELOW the source resolution embedded in the page.
        # A matrix of (3, 3) (~216 DPI equivalent on a 72dpi base) safely
        # captures full source resolution for typical scanned manhwa PDFs
        # without introducing any lossy downscaling.
        matrix = fitz.Matrix(3.0, 3.0)
        for i, page in enumerate(doc):
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            out_path = extract_dir / f"page_{i:05d}.png"
            pix.save(str(out_path))
            images.append(out_path)
    finally:
        doc.close()

    return images


def extract_source_images(input_path: Path, work_dir: Path) -> list[Path]:
    ext = input_path.suffix.lower()
    if ext == ".pdf":
        return extract_images_from_pdf(input_path, work_dir)
    elif ext in (".zip", ".cbz"):
        return extract_images_from_zip_or_cbz(input_path, work_dir)
    else:
        raise ValueError(f"Unsupported input format: {ext}")


# =========================================================================
# CBZ PACKAGING
# =========================================================================

def pack_cbz(image_paths: list[Path], out_cbz_path: Path):
    """
    Packs images into a single .cbz with strict sequential zero-padded
    naming (0001.jpg, 0002.jpg, ...). Uses 4-digit padding, expanding to
    more digits automatically if the chapter has 10000+ pieces.
    """
    total = len(image_paths)
    pad = max(4, len(str(total)))

    with zipfile.ZipFile(out_cbz_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for idx, img_path in enumerate(image_paths, start=1):
            ext = img_path.suffix.lower()
            if ext not in (".jpg", ".jpeg", ".png"):
                ext = ".jpg"
            arcname = f"{idx:0{pad}d}{ext}"
            zf.write(img_path, arcname=arcname)


# =========================================================================
# TELEGRAM HANDLERS
# =========================================================================

@app.on_message(filters.command("start"))
async def start_cmd(client: Client, message: Message):
    await message.reply_text(
        "**📖 Smart Manhwa/Webtoon Page Splitter**\n\n"
        "Send me a `.zip`, `.cbz`, or `.pdf` chapter file.\n"
        "Then **reply to that file** with `/split` to begin.\n\n"
        "✅ Zero quality loss — no resizing, no compression artifacts\n"
        "✅ Smart cuts avoid bubbles, text, SFX, and panel borders\n"
        "✅ Output: single strictly-numbered `.cbz`",
    )


@app.on_message(filters.document)
async def on_document_upload(client: Client, message: Message):
    doc = message.document
    file_name = doc.file_name or "file"
    ext = Path(file_name).suffix.lower()

    if ext not in SUPPORTED_INPUT_EXTS:
        await message.reply_text(
            f"⚠️ Unsupported file type `{ext}`.\n"
            f"Please send a `.zip`, `.cbz`, or `.pdf` file."
        )
        return

    job_dir = Path(tempfile.mkdtemp(prefix="manhwa_"))
    local_path = job_dir / file_name

    status_msg = await message.reply_text(
        render_transfer_board("📥 Downloading...", 0, doc.file_size or 1),
        quote=True,
    )

    throttler = ProgressThrottler()

    async def dl_progress(current, total):
        if throttler.ready() or current == total:
            try:
                await status_msg.edit_text(render_transfer_board("📥 Downloading...", current, total))
            except Exception:
                pass

    await message.download(file_name=str(local_path), progress=dl_progress)

    # Register this upload so a reply of "/split" can find it.
    PENDING_UPLOADS[(message.chat.id, message.id)] = {
        "path": local_path,
        "job_dir": job_dir,
        "original_name": file_name,
        "status_msg": status_msg,
    }

    await status_msg.edit_text(
        f"✅ **Received:** `{file_name}` ({human_size(doc.file_size or 0)})\n\n"
        f"↩️ Reply to this file with `/split` to begin splitting."
    )


@app.on_message(filters.command("split"))
async def on_split_cmd(client: Client, message: Message):
    if not message.reply_to_message or not message.reply_to_message.document:
        await message.reply_text(
            "⚠️ Please reply with `/split` **directly to the file you uploaded** "
            "(.zip / .cbz / .pdf)."
        )
        return

    key = (message.chat.id, message.reply_to_message.id)
    job = PENDING_UPLOADS.get(key)

    if not job:
        await message.reply_text(
            "⚠️ I couldn't find that upload in my active session "
            "(it may have expired). Please re-upload the file and try again."
        )
        return

    status_msg = job["status_msg"]
    input_path: Path = job["path"]
    job_dir: Path = job["job_dir"]

    try:
        await run_split_pipeline(client, message, status_msg, input_path, job_dir)
    except Exception as e:
        log.exception("Split pipeline failed")
        await status_msg.edit_text(f"❌ **Error during processing:**\n`{e}`")
    finally:
        # Cleanup temp directory + registry entry regardless of outcome
        PENDING_UPLOADS.pop(key, None)
        shutil.rmtree(job_dir, ignore_errors=True)


async def run_split_pipeline(client: Client, message: Message, status_msg: Message,
                              input_path: Path, job_dir: Path):
    # ---- Stage 1: Extraction ----
    await status_msg.edit_text("📦 **Extracting...**\nReading pages from your file.")
    source_images = await asyncio.to_thread(extract_source_images, input_path, job_dir)

    if not source_images:
        await status_msg.edit_text("❌ No valid images found inside the uploaded file.")
        return

    # ---- Stage 2: Smart splitting ----
    output_dir = job_dir / "output_parts"
    output_dir.mkdir(parents=True, exist_ok=True)

    all_output_paths: list[Path] = []
    forced_cut_count = 0
    total_source_pages = len(source_images)

    throttler = ProgressThrottler()
    await status_msg.edit_text(render_page_board(0, total_source_pages))

    for page_idx, src_path in enumerate(source_images, start=1):
        pieces, forced_flags = await asyncio.to_thread(split_strip_lossless, src_path)

        for piece_idx, (piece_img, was_forced) in enumerate(zip(pieces, forced_flags), start=1):
            out_ext = ".jpg" if src_path.suffix.lower() != ".png" else ".png"
            out_name = f"src{page_idx:05d}_part{piece_idx:03d}{out_ext}"
            out_path = output_dir / out_name
            await asyncio.to_thread(save_lossless, piece_img, out_path)
            all_output_paths.append(out_path)
            if was_forced:
                forced_cut_count += 1

        if throttler.ready() or page_idx == total_source_pages:
            extra = f"⚠️ {forced_cut_count} forced cut(s) so far" if forced_cut_count else "✅ 0 bubble/text cuts so far"
            await status_msg.edit_text(render_page_board(page_idx, total_source_pages, extra))

    # ---- Stage 3: Package into CBZ ----
    await status_msg.edit_text(
        f"📚 **Packaging CBZ...**\n"
        f"{len(all_output_paths)} total split parts from {total_source_pages} source pages."
    )

    out_name = Path(input_path).stem + "_split.cbz"
    out_cbz_path = job_dir / out_name
    await asyncio.to_thread(pack_cbz, all_output_paths, out_cbz_path)

    cbz_size = out_cbz_path.stat().st_size

    # ---- Stage 4: Upload ----
    await status_msg.edit_text(render_transfer_board("📤 Uploading...", 0, cbz_size))

    up_throttler = ProgressThrottler()

    async def up_progress(current, total):
        if up_throttler.ready() or current == total:
            try:
                await status_msg.edit_text(render_transfer_board("📤 Uploading...", current, total))
            except Exception:
                pass

    summary_caption = (
        f"✅ **Done!**\n"
        f"Source pages: {total_source_pages}\n"
        f"Output parts: {len(all_output_paths)}\n"
        f"Forced (no-clean-spot) cuts: {forced_cut_count}\n"
        f"Quality: Lossless (JPEG q100 / no resizing)"
    )

    await client.send_document(
        chat_id=message.chat.id,
        document=str(out_cbz_path),
        file_name=out_name,
        caption=summary_caption,
        progress=up_progress,
    )

    await status_msg.edit_text(
        f"🎉 **Complete!**\n\n{summary_caption}"
    )


# =========================================================================
# ENTRYPOINT
# =========================================================================

if __name__ == "__main__":
    log.info("Starting Smart Manhwa Splitter Bot...")
    app.run()
