"""Draw the Setsuna fcitx5 themes (setsuna, setsuna-dark) and tray icons.

The PNGs and theme.conf files next to this script are its output and are
checked in, so building the addon needs no Python. Regenerate with

    uv run --no-project --with pillow python crates/linux/fcitx5/themes/generate.py

Everything is drawn at 4x and downsampled for smooth edges.
"""

import math
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw, ImageFilter

HERE = Path(__file__).resolve().parent
SS = 4  # supersampling

# The panels are translucent but not blurred (see EnableBlur below), so the
# text behind shows through: keep the alpha high enough to read over it.

# Panel geometry (1x pixels). The background image is a 9-patch: the shadow
# lives inside the image, outside the rounded panel.
BG = 128
RADIUS = 14
# Kept small: fcitx5-gtk (GTK apps on Wayland) places the panel without
# subtracting ShadowMargin, so the shadow is distance from the caret there.
SHADOW = dict(left=10, right=10, top=4, bottom=14)
STRIP = 5  # accent strip under the top edge

# The brand gradient: amber (the TSF highlight #D77800) through coral to violet.
ACCENT = [(255, 176, 59), (255, 112, 102), (171, 120, 255)]

THEMES = {
    "setsuna": dict(
        name="Setsuna",
        name_ja="Setsuna",
        desc="Setsuna: light panel with an amber-to-violet accent",
        top=(255, 255, 255, 212),
        bottom=(246, 243, 255, 222),
        border=(120, 104, 170, 70),
        inner=(255, 255, 255, 200),
        shadow=(40, 20, 80, 70),
        text="#2b2838",
        highlight_text="#ffffff",
        preedit_bg="#ffb03b",
        preedit_text="#2b2838",
        hl_top=(255, 186, 82),
        hl_bottom=(236, 112, 18),
        hl_border=(196, 92, 0, 160),
        button=(120, 104, 170),
        sparkle_alpha=110,
        separator="#dcd6ee",
    ),
    "setsuna-dark": dict(
        name="Setsuna Dark",
        name_ja="Setsuna ダーク",
        desc="Setsuna: dark panel with an amber-to-violet accent",
        top=(40, 42, 62, 200),
        bottom=(26, 27, 42, 212),
        border=(255, 255, 255, 34),
        inner=(255, 255, 255, 22),
        shadow=(0, 0, 0, 150),
        text="#e8e6f2",
        highlight_text="#ffffff",
        preedit_bg="#ff9f43",
        preedit_text="#1a1b2a",
        hl_top=(255, 166, 72),
        hl_bottom=(226, 86, 40),
        hl_border=(255, 210, 160, 90),
        button=(200, 186, 255),
        sparkle_alpha=140,
        separator="#3a3c55",
    ),
}


def lerp(a, b, t):
    return tuple(round(x + (y - x) * t) for x, y in zip(a, b))


def gradient(stops, t):
    """Colour at `t` (0..1) along evenly spaced `stops`."""
    if t <= 0:
        return stops[0]
    if t >= 1:
        return stops[-1]
    seg = t * (len(stops) - 1)
    i = int(seg)
    return lerp(stops[i], stops[i + 1], seg - i)


def rounded_mask(size, box, radius):
    m = Image.new("L", size, 0)
    ImageDraw.Draw(m).rounded_rectangle(box, radius=radius, fill=255)
    return m


def vertical_fill(size, top, bottom):
    w, h = size
    col = Image.new("RGBA", (1, h))
    for y in range(h):
        col.putpixel((0, y), lerp(top, bottom, y / max(h - 1, 1)))
    return col.resize((w, h))


def horizontal_fill(size, stops):
    w, h = size
    row = Image.new("RGBA", (w, 1))
    for x in range(w):
        row.putpixel((x, 0), gradient(stops, x / max(w - 1, 1)) + (255,))
    return row.resize((w, h))


def down(img):
    return img.resize((img.width // SS, img.height // SS), Image.LANCZOS)


def panel_box():
    s = SHADOW
    return (s["left"] * SS, s["top"] * SS, (BG - s["right"]) * SS - 1, (BG - s["bottom"]) * SS - 1)


def background(t):
    size = (BG * SS, BG * SS)
    box = panel_box()
    r = RADIUS * SS
    out = Image.new("RGBA", size, (0, 0, 0, 0))

    # soft drop shadow, a little below the panel
    sh = Image.new("RGBA", size, t["shadow"][:3] + (0,))
    sh_box = (box[0], box[1] + 3 * SS, box[2], box[3] + 3 * SS)
    sh.putalpha(rounded_mask(size, sh_box, r).point(lambda v: v * t["shadow"][3] // 255))
    out = Image.alpha_composite(out, sh.filter(ImageFilter.GaussianBlur(5 * SS)))

    mask = rounded_mask(size, box, r)
    body = vertical_fill(size, t["top"], t["bottom"])
    body.putalpha(ImageChops.multiply(body.getchannel("A"), mask))
    out = Image.alpha_composite(out, body)

    # accent strip hugging the top edge, clipped to the rounded panel
    strip = horizontal_fill(size, ACCENT)
    strip_mask = Image.new("L", size, 0)
    ImageDraw.Draw(strip_mask).rectangle((box[0], box[1], box[2], box[1] + STRIP * SS), fill=255)
    strip.putalpha(ImageChops.multiply(strip_mask, mask))
    out = Image.alpha_composite(out, strip)

    d = ImageDraw.Draw(out)
    # inner light line below the strip, then the border
    inset = (box[0] + SS, box[1] + (STRIP + 1) * SS, box[2] - SS, box[3] - SS)
    d.rounded_rectangle(inset, radius=r - SS, outline=t["inner"], width=SS)
    d.rounded_rectangle(box, radius=r, outline=t["border"], width=SS)
    return down(out)


def highlight(t, w=48, h=36, radius=7):
    size = (w * SS, h * SS)
    box = (SS, SS, w * SS - SS - 1, h * SS - SS - 1)
    mask = rounded_mask(size, box, radius * SS)
    fill = vertical_fill(size, t["hl_top"] + (255,), t["hl_bottom"] + (255,))
    fill.putalpha(mask)
    out = Image.new("RGBA", size, (0, 0, 0, 0))
    out = Image.alpha_composite(out, fill)
    d = ImageDraw.Draw(out)
    # glossy upper half
    gloss = Image.new("RGBA", size, (255, 255, 255, 0))
    gd = ImageDraw.Draw(gloss)
    gd.rounded_rectangle(
        (box[0] + SS, box[1] + SS, box[2] - SS, box[1] + (box[3] - box[1]) // 2),
        radius=(radius - 1) * SS,
        fill=(255, 255, 255, 46),
    )
    out = Image.alpha_composite(out, gloss)
    d = ImageDraw.Draw(out)
    d.rounded_rectangle(box, radius=radius * SS, outline=t["hl_border"], width=SS)
    return down(out)


def page_button(t, direction, size=22):
    s = size * SS
    out = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    d = ImageDraw.Draw(out)
    c = t["button"]
    d.ellipse((SS, SS, s - SS - 1, s - SS - 1), fill=c + (40,), outline=c + (150,), width=SS)
    m = s // 2
    a = 4 * SS
    if direction == "prev":
        pts = [(m + a // 2, m - a), (m - a // 2, m), (m + a // 2, m + a)]
    else:
        pts = [(m - a // 2, m - a), (m + a // 2, m), (m - a // 2, m + a)]
    d.line(pts, fill=c + (255,), width=int(2.2 * SS), joint="curve")
    return down(out)


def sparkle(t, size=30):
    """A four-point star in the accent gradient, with two satellite dots."""
    s = size * SS
    shape = Image.new("L", (s, s), 0)
    d = ImageDraw.Draw(shape)

    def star(cx, cy, r, waist):
        pts = []
        for k in range(8):
            ang = k * 3.14159265 / 4 - 3.14159265 / 2
            rr = r if k % 2 == 0 else waist
            pts.append((cx + rr * math.cos(ang), cy + rr * math.sin(ang)))
        d.polygon(pts, fill=255)

    star(s * 0.58, s * 0.42, s * 0.36, s * 0.09)
    d.ellipse((s * 0.12, s * 0.70, s * 0.24, s * 0.82), fill=255)
    d.ellipse((s * 0.80, s * 0.80, s * 0.88, s * 0.88), fill=255)
    fill = Image.new("RGBA", (s, s))
    for y in range(s):
        for x in range(0, s, SS):
            col = gradient(ACCENT, (x + y) / (2 * s)) + (255,)
            for dx in range(SS):
                fill.putpixel((x + dx, y), col)
    fill.putalpha(shape.point(lambda v: v * t["sparkle_alpha"] // 255))
    return down(fill)


def menu_highlight(t):
    return highlight(t, w=40, h=30, radius=6)


def theme_conf(t):
    s = SHADOW
    content = dict(left=s["left"] + 10, right=s["right"] + 10, top=s["top"] + STRIP + 9, bottom=s["bottom"] + 8)
    nine = dict(
        left=s["left"] + RADIUS + 2,
        right=s["right"] + RADIUS + 2,
        top=s["top"] + RADIUS + STRIP,
        bottom=s["bottom"] + RADIUS + 2,
    )

    def margin(section, m):
        return (
            f"[{section}]\nLeft={m['left']}\nRight={m['right']}\nTop={m['top']}\nBottom={m['bottom']}\n\n"
        )

    out = f"""# Generated by generate.py; edit that instead.
[Metadata]
Name={t['name']}
Name[ja]={t['name_ja']}
Version=1
Author=Setsuna
Description={t['desc']}
ScaleWithDPI=True

[InputPanel]
NormalColor={t['text']}
HighlightCandidateColor={t['highlight_text']}
HighlightColor={t['preedit_text']}
HighlightBackgroundColor={t['preedit_bg']}
# No blur: with KWin's background-effect blur the Wayland input panel stays
# clipped to its first size (the candidates are cut off).
EnableBlur=False
FullWidthHighlight=True
Spacing=2
PageButtonAlignment=Bottom

"""
    out += margin("InputPanel/ShadowMargin", s)
    out += margin("InputPanel/TextMargin", dict(left=10, right=10, top=6, bottom=6))
    out += margin("InputPanel/ContentMargin", content)
    out += """[InputPanel/Background]
Image=panel.png
Overlay=sparkle.png
Gravity=Top Right
OverlayOffsetX=16
OverlayOffsetY=10
HideOverlayIfOversize=True

"""
    out += margin("InputPanel/Background/Margin", nine)
    out += margin("InputPanel/Background/OverlayClipMargin", s)
    out += "[InputPanel/Highlight]\nImage=highlight.png\n\n"
    out += margin("InputPanel/Highlight/Margin", dict(left=10, right=10, top=8, bottom=8))
    out += margin("InputPanel/Highlight/HighlightClickMargin", dict(left=0, right=0, top=0, bottom=0))
    out += "[InputPanel/PrevPage]\nImage=prev.png\n\n"
    out += margin("InputPanel/PrevPage/ClickMargin", dict(left=4, right=4, top=4, bottom=4))
    out += "[InputPanel/NextPage]\nImage=next.png\n\n"
    out += margin("InputPanel/NextPage/ClickMargin", dict(left=4, right=4, top=4, bottom=4))
    out += "[Menu]\nNormalColor={0}\nHighlightCandidateColor={1}\nSpacing=0\n\n".format(
        t["text"], t["highlight_text"]
    )
    out += "[Menu/Background]\nImage=panel.png\n\n"
    out += margin("Menu/Background/Margin", nine)
    out += margin("Menu/ContentMargin", content)
    out += margin("Menu/TextMargin", dict(left=10, right=10, top=6, bottom=6))
    out += "[Menu/Highlight]\nImage=menu-highlight.png\n\n"
    out += margin("Menu/Highlight/Margin", dict(left=8, right=8, top=7, bottom=7))
    out += f"[Menu/Separator]\nColor={t['separator']}\n\n"
    return out


ICON = """<svg xmlns="http://www.w3.org/2000/svg" width="64" height="64" viewBox="0 0 64 64">
  <defs>
    <linearGradient id="g" x1="0" y1="0" x2="1" y2="1">
{stops}
    </linearGradient>
    <linearGradient id="gloss" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0" stop-color="#ffffff" stop-opacity="0.35"/>
      <stop offset="0.5" stop-color="#ffffff" stop-opacity="0"/>
    </linearGradient>
  </defs>
  <rect x="3" y="3" width="58" height="58" rx="15" fill="url(#g)"/>
  <rect x="3" y="3" width="58" height="58" rx="15" fill="url(#gloss)"/>
  <text x="32" y="45" text-anchor="middle" font-family="IBM Plex Sans JP, Noto Sans CJK JP, sans-serif"
        font-size="{size}" font-weight="700" fill="#ffffff">{label}</text>
</svg>
"""


def icon(label, colors, size):
    stops = "\n".join(
        f'      <stop offset="{i / (len(colors) - 1):.2f}" stop-color="{c}"/>' for i, c in enumerate(colors)
    )
    return ICON.format(stops=stops, label=label, size=size)


def main():
    for key, t in THEMES.items():
        d = HERE / key
        d.mkdir(exist_ok=True)
        background(t).save(d / "panel.png")
        highlight(t).save(d / "highlight.png")
        menu_highlight(t).save(d / "menu-highlight.png")
        page_button(t, "prev").save(d / "prev.png")
        page_button(t, "next").save(d / "next.png")
        sparkle(t).save(d / "sparkle.png")
        (d / "theme.conf").write_text(theme_conf(t))

    icons = HERE.parent / "icons"
    icons.mkdir(exist_ok=True)
    accent = ["#ffb03b", "#ff7066", "#ab78ff"]
    (icons / "setsuna.svg").write_text(icon("あ", accent, 34))
    (icons / "setsuna-hiragana.svg").write_text(icon("あ", accent, 34))
    (icons / "setsuna-direct.svg").write_text(icon("A", ["#7a8199", "#4b5068"], 36))


if __name__ == "__main__":
    main()
