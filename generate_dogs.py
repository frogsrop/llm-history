"""Generate pixel art dog sprites for the neuron module backprop card.

Inspired by Shutterstock pixel art reference: cute brown dog, 4 poses.
Colors matched to dark theme background.

Output: static/dogs/ directory with 4 transparent PNGs.
"""

from PIL import Image
from pathlib import Path

# Color palette — warm browns on transparent bg
C = {
    '.': (0, 0, 0, 0),         # transparent
    'd': (92, 51, 16, 255),     # dark brown outline
    'b': (212, 167, 106, 255),  # body tan
    'l': (236, 200, 139, 255),  # light belly / chest
    'e': (178, 130, 72, 255),   # ear inner / shading
    'w': (255, 255, 255, 255),  # eye white
    'n': (26, 26, 26, 255),     # nose / pupil
    'p': (166, 120, 64, 255),   # far-side limbs (shadow)
}

# ── Standing (facing right, tail raised on left) ──
STAND = [
    '..............dd.dd.',
    '.............dbbdebd',
    '.............dbbbbbd',
    '.............dbwbnd.',
    '..dd.........dbbbd..',
    '.dbd.dbbbbbbbddd....',
    '..bd.dbbbbbbbd......',
    '.....dbbbllbbd......',
    '.....dbbbllbbd......',
    '.....dbbbbbbbd......',
    '......dbbbbbd.......',
    '......dp..dpd.......',
    '......dp..dpd.......',
    '......dd..ddd.......',
]

# ── Crouching (body lower, legs shorter) ──
CROUCH = [
    '..............dd.dd.',
    '.............dbbdebd',
    '.............dbbbbbd',
    '..d..........dbwbnd.',
    '..dd.dbbbbbbbdbbbd..',
    '...d.dbbbbbbdddd....',
    '.....dbbblllbbd.....',
    '.....dbbblllbbd.....',
    '.....dbbbbbbbbd.....',
    '......dbbbbbd.......',
    '......dp.dpd........',
    '......dd.ddd........',
]

# ── Sitting (vertical body, head high, haunches on ground) ──
SIT = [
    '....dd.dd.',
    '...dbbdebd',
    '...dbbbbbd',
    '...dbwbnd.',
    '...dbbbd..',
    '..dbbbbbd.',
    '.dbbbbbbd.',
    '.dbbllbbd.',
    '.dbbllbbd.',
    'dbbbbbbbbd',
    'dbbbbbbd..',
    'dp..dbbd..',
    'dp..dppd..',
    'dd..dddd..',
]

# ── Lying (flat, curled, head resting forward) ──
LIE = [
    '..............dd.dd.',
    '.............dbbdebd',
    '.ddddbbbbbbbbbbbbbd.',
    'dbbbbbbbbbbbbbwbbnd.',
    'dbbbllllllbbbbbbbbd.',
    'dbbbllllllbbbbdddd..',
    '.dbbbbbbbbbbbd......',
    '..ddddddddddddd....',
]


def render(rows, scale=1):
    """Render pixel art rows into a PIL Image."""
    h = len(rows)
    w = max(len(r) for r in rows)
    img = Image.new('RGBA', (w * scale, h * scale), (0, 0, 0, 0))
    for y, row in enumerate(rows):
        for x, ch in enumerate(row):
            color = C.get(ch, C['.'])
            if color[3] == 0:
                continue
            for dy in range(scale):
                for dx in range(scale):
                    img.putpixel((x * scale + dx, y * scale + dy), color)
    return img


def main():
    out = Path('static/dogs')
    out.mkdir(parents=True, exist_ok=True)

    poses = {
        'stand': STAND,
        'crouch': CROUCH,
        'sit': SIT,
        'lie': LIE,
    }

    for name, data in poses.items():
        img = render(data, scale=1)
        path = out / f'{name}.png'
        img.save(path)
        print(f'{path}: {img.size[0]}x{img.size[1]}')


if __name__ == '__main__':
    main()
