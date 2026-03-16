"""
Generates 40×40 px carrier-medium icons for Aurik 9.
Output: Aurik910/resources/carrier_icons/<medium>.png

Design language: dark/transparent background, white + accent-color shapes,
matching Aurik's dark purple-blue UI theme.
"""
from __future__ import annotations
import math
from pathlib import Path
from PIL import Image, ImageDraw

OUT = Path(__file__).parent.parent / "Aurik910" / "resources" / "carrier_icons"
OUT.mkdir(parents=True, exist_ok=True)

SIZE = 40
BG = (0, 0, 0, 0)          # transparent
WHITE = (255, 255, 255, 255)
GREY = (180, 190, 210, 255)
ACCENT = (130, 100, 240, 255)   # Aurik violet
GOLD = (240, 190, 60, 255)
CYAN = (80, 210, 230, 255)
RED = (220, 70, 70, 255)
GREEN = (80, 200, 120, 255)


def new() -> tuple[Image.Image, ImageDraw.ImageDraw]:
    img = Image.new("RGBA", (SIZE, SIZE), BG)
    return img, ImageDraw.Draw(img)


def save(img: Image.Image, name: str) -> None:
    path = OUT / f"{name}.png"
    img.save(path, "PNG")
    print(f"  ✓ {path.name}")


# ── vinyl ─────────────────────────────────────────────────────────────────
def make_vinyl() -> None:
    img, d = new()
    cx, cy = 20, 20
    # Outer disc
    d.ellipse([2, 2, 37, 37], fill=(30, 30, 30, 255), outline=WHITE)
    # Grooves
    for r in [14, 11, 8]:
        d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=(120, 120, 120, 255))
    # Label in center
    d.ellipse([cx - 6, cy - 6, cx + 6, cy + 6], fill=ACCENT)
    # Center hole
    d.ellipse([cx - 2, cy - 2, cx + 2, cy + 2], fill=(0, 0, 0, 200))
    save(img, "vinyl")


# ── shellac ───────────────────────────────────────────────────────────────
def make_shellac() -> None:
    img, d = new()
    cx, cy = 20, 20
    d.ellipse([2, 2, 37, 37], fill=(50, 35, 20, 255), outline=GOLD)
    for r in [15, 11, 7]:
        d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=(160, 130, 60, 180))
    d.ellipse([cx - 5, cy - 5, cx + 5, cy + 5], fill=GOLD)
    d.ellipse([cx - 2, cy - 2, cx + 2, cy + 2], fill=(0, 0, 0, 200))
    save(img, "shellac")


# ── tape / cassette (two reels) ───────────────────────────────────────────
def _draw_cassette(name: str, color: tuple) -> None:
    img, d = new()
    # Housing
    d.rounded_rectangle([2, 8, 37, 33], radius=4, fill=(40, 40, 55, 255), outline=color)
    # Left reel
    d.ellipse([5, 11, 19, 25], outline=color)
    d.ellipse([10, 16, 14, 20], fill=color)
    # Right reel
    d.ellipse([21, 11, 35, 25], outline=color)
    d.ellipse([26, 16, 30, 20], fill=color)
    # Tape window line
    d.line([12, 27, 28, 27], fill=color, width=1)
    save(img, name)


def make_tape() -> None:
    _draw_cassette("tape", WHITE)


def make_cassette() -> None:
    _draw_cassette("cassette", CYAN)


# ── reel_tape ─────────────────────────────────────────────────────────────
def make_reel_tape() -> None:
    img, d = new()
    # Left large reel
    cx1, cy1 = 11, 21
    d.ellipse([cx1 - 9, cy1 - 9, cx1 + 9, cy1 + 9], outline=WHITE, width=2)
    # Spokes
    for angle_deg in [0, 120, 240]:
        a = math.radians(angle_deg)
        x2 = cx1 + 7 * math.cos(a)
        y2 = cy1 + 7 * math.sin(a)
        d.line([cx1, cy1, x2, y2], fill=WHITE, width=1)
    d.ellipse([cx1 - 2, cy1 - 2, cx1 + 2, cy1 + 2], fill=WHITE)
    # Right large reel
    cx2, cy2 = 29, 21
    d.ellipse([cx2 - 9, cy2 - 9, cx2 + 9, cy2 + 9], outline=ACCENT, width=2)
    for angle_deg in [60, 180, 300]:
        a = math.radians(angle_deg)
        x2 = cx2 + 7 * math.cos(a)
        y2 = cy2 + 7 * math.sin(a)
        d.line([cx2, cy2, x2, y2], fill=ACCENT, width=1)
    d.ellipse([cx2 - 2, cy2 - 2, cx2 + 2, cy2 + 2], fill=ACCENT)
    # Tape strip between reels
    d.line([cx1 + 9, cy1 - 4, cx2 - 9, cy2 - 4], fill=GREY, width=2)
    save(img, "reel_tape")


# ── wax_cylinder ──────────────────────────────────────────────────────────
def make_wax_cylinder() -> None:
    img, d = new()
    # Cylinder body
    d.rectangle([8, 10, 32, 32], fill=(80, 55, 20, 255), outline=GOLD)
    # Top ellipse
    d.ellipse([8, 6, 32, 14], fill=(100, 70, 25, 255), outline=GOLD)
    # Groove lines
    for y in [16, 20, 24, 28]:
        d.line([8, y, 32, y], fill=(160, 130, 60, 120), width=1)
    save(img, "wax_cylinder")


# ── lacquer_disc ──────────────────────────────────────────────────────────
def make_lacquer_disc() -> None:
    img, d = new()
    cx, cy = 20, 20
    d.ellipse([2, 2, 37, 37], fill=(20, 20, 30, 255), outline=CYAN)
    # Cut marks (radial scratches)
    for angle_deg in range(0, 360, 30):
        a = math.radians(angle_deg)
        x1 = cx + 10 * math.cos(a)
        y1 = cy + 10 * math.sin(a)
        x2 = cx + 17 * math.cos(a)
        y2 = cy + 17 * math.sin(a)
        d.line([x1, y1, x2, y2], fill=(100, 200, 200, 150), width=1)
    d.ellipse([cx - 5, cy - 5, cx + 5, cy + 5], fill=CYAN)
    d.ellipse([cx - 2, cy - 2, cx + 2, cy + 2], fill=BG)
    save(img, "lacquer_disc")


# ── wire_recording ────────────────────────────────────────────────────────
def make_wire_recording() -> None:
    img, d = new()
    # Spool left
    d.ellipse([2, 12, 16, 28], outline=WHITE, width=2)
    d.ellipse([7, 17, 11, 23], fill=WHITE)
    # Spool right
    d.ellipse([24, 12, 38, 28], outline=WHITE, width=2)
    d.ellipse([29, 17, 33, 23], fill=WHITE)
    # Wire as wavy line
    pts = []
    for i in range(17, 24):
        x = i
        y = 20 + int(3 * math.sin((i - 17) * math.pi / 3))
        pts.append((x, y))
    d.line(pts, fill=GREY, width=2)
    save(img, "wire_recording")


# ── cd_digital ────────────────────────────────────────────────────────────
def make_cd() -> None:
    img, d = new()
    cx, cy = 20, 20
    # Disc with rainbow-like gradient via concentric rings
    ring_colors = [
        (200, 180, 255, 220),
        (180, 220, 255, 200),
        (255, 220, 180, 180),
        (200, 255, 200, 160),
    ]
    for i, col in enumerate(ring_colors):
        r = 17 - i * 2
        d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=col, width=2)
    # Outer rim
    d.ellipse([3, 3, 36, 36], outline=WHITE, width=1)
    # Center hub
    d.ellipse([cx - 5, cy - 5, cx + 5, cy + 5], fill=(60, 60, 80, 255), outline=WHITE)
    d.ellipse([cx - 2, cy - 2, cx + 2, cy + 2], fill=BG)
    save(img, "cd_digital")
    save(img, "cd")


# ── dat ───────────────────────────────────────────────────────────────────
def make_dat() -> None:
    img, d = new()
    # Small cassette-like housing
    d.rounded_rectangle([4, 10, 36, 30], radius=3, fill=(30, 40, 60, 255), outline=CYAN)
    # Two small reels
    d.ellipse([8, 13, 18, 23], outline=CYAN)
    d.ellipse([22, 13, 32, 23], outline=CYAN)
    # DAT label
    d.rectangle([10, 25, 30, 28], fill=CYAN)
    save(img, "dat")


# ── minidisc ──────────────────────────────────────────────────────────────
def make_minidisc() -> None:
    img, d = new()
    # Housing – square with rounded corners
    d.rounded_rectangle([4, 4, 36, 36], radius=6, fill=(30, 30, 50, 255), outline=WHITE)
    # Disc
    d.ellipse([8, 8, 32, 32], outline=ACCENT, width=2)
    # Shutter (rectangular cutout at bottom)
    d.rectangle([13, 28, 27, 35], fill=(30, 30, 50, 255), outline=WHITE)
    # Center hole
    cx, cy = 20, 20
    d.ellipse([cx - 3, cy - 3, cx + 3, cy + 3], fill=(50, 50, 70, 255), outline=WHITE)
    save(img, "minidisc")


# ── mp3_low ───────────────────────────────────────────────────────────────
def make_mp3_low() -> None:
    img, d = new()
    # Waveform bars (low quality = short + broken)
    heights = [8, 14, 6, 16, 4, 10, 6, 4]
    x = 3
    for h in heights:
        y_top = 20 - h // 2
        d.rectangle([x, y_top, x + 3, y_top + h], fill=RED)
        x += 5
    # Compression artifact lines
    d.line([3, 32, 37, 32], fill=(200, 80, 80, 150), width=1)
    save(img, "mp3_low")


# ── mp3_high ──────────────────────────────────────────────────────────────
def make_mp3_high() -> None:
    img, d = new()
    heights = [10, 18, 24, 28, 22, 16, 10, 6]
    x = 3
    for h in heights:
        y_top = 20 - h // 2
        d.rectangle([x, y_top, x + 3, y_top + h], fill=GREEN)
        x += 5
    save(img, "mp3_high")


# ── damaged_mp3 ───────────────────────────────────────────────────────────
def make_damaged_mp3() -> None:
    img, d = new()
    heights = [10, 18, 24, 28, 22, 16, 10, 6]
    x = 3
    for i, h in enumerate(heights):
        y_top = 20 - h // 2
        color = RED if i in (2, 5) else GREY
        d.rectangle([x, y_top, x + 3, y_top + h], fill=color)
        x += 5
    # X overlay
    d.line([6, 6, 34, 34], fill=RED, width=2)
    d.line([34, 6, 6, 34], fill=RED, width=2)
    save(img, "damaged_mp3")


# ── aac ───────────────────────────────────────────────────────────────────
def make_aac() -> None:
    img, d = new()
    heights = [12, 20, 26, 30, 24, 18, 12, 8]
    x = 3
    for h in heights:
        y_top = 20 - h // 2
        d.rectangle([x, y_top, x + 3, y_top + h], fill=CYAN)
        x += 5
    save(img, "aac")


# ── streaming ─────────────────────────────────────────────────────────────
def make_streaming() -> None:
    img, d = new()
    cx, cy = 20, 28
    # Radio arcs
    for r, alpha in [(16, 180), (11, 210), (6, 240)]:
        d.arc([cx - r, cy - r, cx + r, cy + r], start=200, end=340,
              fill=(255, 255, 255, alpha), width=2)
    # Dot at center
    d.ellipse([cx - 2, cy - 2, cx + 2, cy + 2], fill=WHITE)
    save(img, "streaming")


# ── unknown ───────────────────────────────────────────────────────────────
def make_unknown() -> None:
    img, d = new()
    cx, cy = 20, 20
    d.ellipse([3, 3, 36, 36], outline=GREY, width=2)
    # Question mark (manual segments for portability)
    # Arc top of Q
    d.arc([12, 8, 28, 22], start=200, end=350, fill=GREY, width=3)
    # Stem
    d.line([20, 19, 20, 27], fill=GREY, width=3)
    # Dot
    d.ellipse([18, 29, 22, 33], fill=GREY)
    save(img, "unknown")


if __name__ == "__main__":
    print("Generating carrier icons →", OUT)
    make_vinyl()
    make_shellac()
    make_tape()
    make_cassette()
    make_reel_tape()
    make_wax_cylinder()
    make_lacquer_disc()
    make_wire_recording()
    make_cd()
    make_dat()
    make_minidisc()
    make_mp3_low()
    make_mp3_high()
    make_damaged_mp3()
    make_aac()
    make_streaming()
    make_unknown()
    print("Done.")
