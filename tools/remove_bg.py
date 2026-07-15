"""Batch background remover for Class Highlight cutouts (LOCAL use — rembg is not
installed on Render on purpose; its inference memory would OOM the instance).

Usage:
    py tools/remove_bg.py <input_dir> [output_dir]

Every jpg/jpeg/png/webp in <input_dir> is stripped to a transparent-background PNG.
Default output_dir = data/class_highlight_headshots (files named like the input, so
name them <player_id_64a>.jpg to have the page pick them up automatically).
"""
import sys
from pathlib import Path

from PIL import Image
from rembg import remove

APP_DIR = Path(__file__).resolve().parent.parent


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    src = Path(sys.argv[1])
    dst = Path(sys.argv[2]) if len(sys.argv) > 2 else APP_DIR / 'data' / 'class_highlight_headshots'
    dst.mkdir(parents=True, exist_ok=True)
    files = [p for p in src.iterdir() if p.suffix.lower() in ('.jpg', '.jpeg', '.png', '.webp')]
    if not files:
        print(f'no images in {src}')
        return
    for p in files:
        out = dst / (p.stem + '.png')
        im = Image.open(p).convert('RGBA')
        remove(im).save(out, 'PNG')
        print(f'  {p.name} -> {out}')
    print(f'done: {len(files)} cutout(s) in {dst}')


if __name__ == '__main__':
    main()
