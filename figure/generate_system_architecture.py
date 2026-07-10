"""Generate Figure 1: overall system architecture (SVG + PNG + PDF)."""

from __future__ import annotations

import math
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
W, H = 2100, 1150
BG, INK, MUTED = "#F8FAFC", "#1E293B", "#475569"
BLUE, BLUE_L = "#2563EB", "#DBEAFE"
GREEN, GREEN_L = "#059669", "#D1FAE5"
AMBER, AMBER_L = "#D97706", "#FEF3C7"
PURPLE, PURPLE_L = "#7C3AED", "#EDE9FE"
WHITE = "#FFFFFF"


def _font(size: int, bold: bool = False):
    for p in [
        Path(f"C:/Windows/Fonts/{'arialbd.ttf' if bold else 'arial.ttf'}"),
        Path(f"C:/Windows/Fonts/{'msyhbd.ttc' if bold else 'msyh.ttc'}"),
    ]:
        if p.exists():
            return ImageFont.truetype(str(p), size)
    return ImageFont.load_default()


F_H, F, F_S, F_T = _font(24, True), _font(20), _font(17), _font(15)


def wrap(text: str, n: int) -> list[str]:
    out: list[str] = []
    for line in text.split("\n"):
        out.extend(textwrap.wrap(line, width=n, break_long_words=False) or [""])
    return out


def esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;")


def svg_box(x, y, w, h, title, body="", fill=WHITE, stroke=BLUE):
    lines = wrap(title, max(8, w // 11))
    body_y = y + 36 + (len(lines) - 1) * 22 + 26
    parts = [
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="16" fill="{fill}" stroke="{stroke}" stroke-width="2"/>',
        f'<text x="{x+w//2}" y="{y+32}" text-anchor="middle" font-family="Arial,sans-serif" font-size="18" font-weight="700" fill="{INK}">',
    ]
    for i, ln in enumerate(lines):
        parts.append(f'<tspan x="{x+w//2}" dy="{0 if i==0 else 22}">{esc(ln)}</tspan>')
    parts.append("</text>")
    if body:
        bl = wrap(body, max(10, w // 9))
        parts.append(
            f'<text x="{x+w//2}" y="{body_y}" text-anchor="middle" font-family="Arial,sans-serif" font-size="14" fill="{MUTED}">'
        )
        for i, ln in enumerate(bl):
            parts.append(f'<tspan x="{x+w//2}" dy="{0 if i==0 else 18}">{esc(ln)}</tspan>')
        parts.append("</text>")
    return "\n".join(parts)


def svg_arrow(x1, y1, x2, y2, color=BLUE, dashed=False):
    d = ' stroke-dasharray="7 7"' if dashed else ""
    return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="2.5" marker-end="url(#m{color[1:]})"{d}/>'


def build_svg() -> str:
    dy = 0
    deep = 590 + dy
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
        "<defs>",
        f'<marker id="m{BLUE[1:]}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="{BLUE}"/></marker>',
        f'<marker id="m{GREEN[1:]}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="{GREEN}"/></marker>',
        f'<marker id="m{AMBER[1:]}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="{AMBER}"/></marker>',
        f'<marker id="m{PURPLE[1:]}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="{PURPLE}"/></marker>',
        "</defs>",
        f'<rect width="{W}" height="{H}" fill="{BG}"/>',
        svg_box(60, 70+dy, 210, 85, "User Question", "maintenance query"),
        svg_box(340, 65+dy, 290, 110, "Complexity-aware Router",
                "routine: alarms, measurements, lookup, charts\ncomplex: root-cause, cross-document reasoning", BLUE_L),
        svg_box(60, 280+dy, 270, 130, "Structured Operating Data",
                "alarms, telemetry, diagnostic tables, time series", GREEN_L, GREEN),
        svg_box(370, 280+dy, 270, 130, "Private Maintenance KB",
                "manuals, standards, failure cases, image chunks", AMBER_L, AMBER),
        f'<rect x="700" y="{40+dy}" width="1280" height="400" rx="22" fill="#EFF6FF" stroke="{BLUE}" stroke-width="2"/>',
        f'<text x="730" y="{72+dy}" font-family="Arial,sans-serif" font-size="22" font-weight="700" fill="{BLUE}">Fast Path: Single-Agent Tool Routing</text>',
        f'<rect x="700" y="{560+dy}" width="1280" height="380" rx="22" fill="#F5F3FF" stroke="{PURPLE}" stroke-width="2"/>',
        f'<text x="730" y="{592+dy}" font-family="Arial,sans-serif" font-size="22" font-weight="700" fill="{PURPLE}">Deep-Research Path: Multi-Agent Diagnostic Reasoning</text>',
        svg_arrow(270, 112+dy, 340, 112+dy),
        svg_arrow(630, 100+dy, 700, 100+dy),
        svg_arrow(630, 150+dy, 700, deep+60, PURPLE),
        f'<text x="655" y="88+{dy}" font-size="15" font-weight="700" fill="{BLUE}">routine</text>',
        f'<text x="645" y="370+{dy}" font-size="15" font-weight="700" fill="{PURPLE}">complex</text>',
        svg_box(750, 150+dy, 240, 95, "Single-agent Supervisor", "selects next action"),
        svg_box(1050, 105+dy, 180, 75, "DIRECT", "general response"),
        svg_box(1280, 105+dy, 180, 75, "RETRIEVE", "knowledge lookup"),
        svg_box(1050, 230+dy, 180, 80, "DATABASE", "schema-constrained SQL"),
        svg_box(1280, 230+dy, 180, 80, "CHART", "visualization"),
        svg_box(1620, 165+dy, 230, 100, "Answer Synthesis", "response, SQL result, or chart", BLUE_L),
        svg_box(750, deep+55, 190, 80, "Pre-retrieval", "initial evidence", WHITE, PURPLE),
        svg_box(990, deep+55, 190, 80, "Draft generation", "diagnostic sketch", WHITE, PURPLE),
        svg_box(1230, deep+40, 260, 110, "Supervisor-researcher coordination", "iterative evidence seeking", WHITE, PURPLE),
        svg_box(1540, deep+55, 210, 80, "Finding compression", "merge key findings", WHITE, PURPLE),
        svg_box(1320, deep+210, 260, 95, "Final report synthesis", "diagnosis, evidence, actions", PURPLE_L, PURPLE),
        svg_box(1750, 820+dy, 220, 115, "Final Answer / Report", "answer, chart, or report", WHITE, GREEN),
    ]
    for x2, y2 in [(1050, 142+dy), (1280, 142+dy), (1050, 268+dy), (1280, 268+dy)]:
        parts.append(svg_arrow(990, 195+dy, x2, y2))
    for x1, y1, y2 in [(1230, 142+dy, 210+dy), (1460, 142+dy, 210+dy), (1240, 268+dy, 240+dy), (1460, 268+dy, 240+dy)]:
        parts.append(svg_arrow(x1, y1, 1620, y2))
    for x1, x2 in [(940, 990), (1180, 1230), (1490, 1540)]:
        parts.append(svg_arrow(x1, deep+95, x2, deep+95, PURPLE))
    parts.append(svg_arrow(1750, deep+135, 1450, deep+210, PURPLE))
    parts.append(f'<text x="1580" y="{deep+198}" font-size="14" font-weight="700" fill="{PURPLE}">compressed evidence</text>')
    parts.append(svg_arrow(330, 345+dy, 1080, 310+dy, GREEN, True))
    parts.append(svg_arrow(640, 345+dy, 1380, 175+dy, AMBER, True))
    parts.append(svg_arrow(195, 410+dy, 845, deep+55, GREEN, True))
    parts.append(svg_arrow(505, 410+dy, 845, deep+55, AMBER, True))
    parts.append(svg_arrow(1735, 240+dy, 1860, 820+dy, GREEN))
    parts.append(svg_arrow(1450, deep+260, 1750, 870+dy, GREEN))
    parts.append(
        f'<text x="{W//2}" y="1100" text-anchor="middle" font-family="Arial,sans-serif" font-size="16" fill="{MUTED}">'
        "Both paths are grounded by structured operating data and the private maintenance knowledge base.</text>"
    )
    parts.append("</svg>")
    return "\n".join(parts)


def center(draw, xy, text, fnt, fill=INK):
    x, y = xy
    if hasattr(draw, "textbbox"):
        bb = draw.textbbox((0, 0), text, font=fnt)
        w = bb[2] - bb[0]
    else:
        w = draw.textsize(text, font=fnt)[0]
    draw.text((x - w / 2, y), text, font=fnt, fill=fill)


def rrect(draw, box, r, fill, outline, w=2):
    if hasattr(draw, "rounded_rectangle"):
        draw.rounded_rectangle(box, radius=r, fill=fill, outline=outline, width=w)
    else:
        draw.rectangle(box, fill=fill, outline=outline)


def draw_box(draw, x, y, w, h, title, body="", fill=WHITE, outline=BLUE):
    rrect(draw, (x, y, x + w, y + h), 16, fill, outline)
    for i, ln in enumerate(wrap(title, max(8, w // 11))[:2]):
        center(draw, (x + w // 2, y + 18 + i * 22), ln, F_S if i else F_H)
    if body:
        for i, ln in enumerate(wrap(body, max(10, w // 9))[:3]):
            center(draw, (x + w // 2, y + 52 + i * 18), ln, F_T, MUTED)


def draw_arrow(draw, x1, y1, x2, y2, color=BLUE, dashed=False):
    if dashed:
        dx, dy, dist = x2 - x1, y2 - y1, math.hypot(x2 - x1, y2 - y1)
        if not dist:
            return
        ux, uy = dx / dist, dy / dist
        t = 0
        while t < dist - 10:
            s, e = t, min(t + 8, dist - 10)
            draw.line((x1 + ux * s, y1 + uy * s, x1 + ux * e, y1 + uy * e), fill=color, width=3)
            t += 16
    else:
        draw.line((x1, y1, x2, y2), fill=color, width=3)
    ang = math.atan2(y2 - y1, x2 - x1)
    s = 14
    draw.polygon([(x2, y2), (x2 - s * math.cos(ang - 0.5), y2 - s * math.sin(ang - 0.5)),
                  (x2 - s * math.cos(ang + 0.5), y2 - s * math.sin(ang + 0.5))], fill=color)


def build_png() -> Image.Image:
    dy = 0
    deep = 590 + dy
    im = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(im)
    draw_box(d, 60, 70 + dy, 210, 85, "User Question", "maintenance query")
    draw_box(d, 340, 65 + dy, 290, 110, "Complexity-aware Router",
             "routine: alarms, measurements, lookup, charts\ncomplex: root-cause, cross-document reasoning", BLUE_L)
    draw_box(d, 60, 280 + dy, 270, 130, "Structured Operating Data",
             "alarms, telemetry, diagnostic tables, time series", GREEN_L, GREEN)
    draw_box(d, 370, 280 + dy, 270, 130, "Private Maintenance KB",
             "manuals, standards, failure cases, image chunks", AMBER_L, AMBER)
    rrect(d, (700, 40 + dy, 1980, 440 + dy), 22, "#EFF6FF", BLUE)
    d.text((730, 52 + dy), "Fast Path: Single-Agent Tool Routing", font=F_H, fill=BLUE)
    rrect(d, (700, 560 + dy, 1980, 940 + dy), 22, "#F5F3FF", PURPLE)
    d.text((730, 572 + dy), "Deep-Research Path: Multi-Agent Diagnostic Reasoning", font=F_H, fill=PURPLE)
    draw_arrow(d, 270, 112 + dy, 340, 112 + dy)
    draw_arrow(d, 630, 100 + dy, 700, 100 + dy)
    draw_arrow(d, 630, 150 + dy, 700, deep + 60, PURPLE)
    d.text((655, 72 + dy), "routine", font=F_S, fill=BLUE)
    d.text((645, 352 + dy), "complex", font=F_S, fill=PURPLE)
    draw_box(d, 750, 150 + dy, 240, 95, "Single-agent Supervisor", "selects next action")
    draw_box(d, 1050, 105 + dy, 180, 75, "DIRECT", "general response")
    draw_box(d, 1280, 105 + dy, 180, 75, "RETRIEVE", "knowledge lookup")
    draw_box(d, 1050, 230 + dy, 180, 80, "DATABASE", "schema-constrained SQL")
    draw_box(d, 1280, 230 + dy, 180, 80, "CHART", "visualization")
    draw_box(d, 1620, 165 + dy, 230, 100, "Answer Synthesis", "response, SQL result, or chart", BLUE_L)
    draw_box(d, 750, deep + 55, 190, 80, "Pre-retrieval", "initial evidence", WHITE, PURPLE)
    draw_box(d, 990, deep + 55, 190, 80, "Draft generation", "diagnostic sketch", WHITE, PURPLE)
    draw_box(d, 1230, deep + 40, 260, 110, "Supervisor-researcher coordination", "iterative evidence seeking", WHITE, PURPLE)
    draw_box(d, 1540, deep + 55, 210, 80, "Finding compression", "merge key findings", WHITE, PURPLE)
    draw_box(d, 1320, deep + 210, 260, 95, "Final report synthesis", "diagnosis, evidence, actions", PURPLE_L, PURPLE)
    draw_box(d, 1750, 820 + dy, 220, 115, "Final Answer / Report", "answer, chart, or report", WHITE, GREEN)
    for x2, y2 in [(1050, 142 + dy), (1280, 142 + dy), (1050, 268 + dy), (1280, 268 + dy)]:
        draw_arrow(d, 990, 195 + dy, x2, y2)
    for x1, y1, y2 in [(1230, 142 + dy, 210 + dy), (1460, 142 + dy, 210 + dy), (1240, 268 + dy, 240 + dy), (1460, 268 + dy, 240 + dy)]:
        draw_arrow(d, x1, y1, 1620, y2)
    for x1, x2 in [(940, 990), (1180, 1230), (1490, 1540)]:
        draw_arrow(d, x1, deep + 95, x2, deep + 95, PURPLE)
    draw_arrow(d, 1750, deep + 135, 1450, deep + 210, PURPLE)
    center(d, (1580, deep + 188), "compressed evidence", F_S, PURPLE)
    draw_arrow(d, 330, 345 + dy, 1080, 310 + dy, GREEN, True)
    draw_arrow(d, 640, 345 + dy, 1380, 175 + dy, AMBER, True)
    draw_arrow(d, 195, 410 + dy, 845, deep + 55, GREEN, True)
    draw_arrow(d, 505, 410 + dy, 845, deep + 55, AMBER, True)
    draw_arrow(d, 1735, 240 + dy, 1860, 820 + dy, GREEN)
    draw_arrow(d, 1450, deep + 260, 1750, 870 + dy, GREEN)
    center(d, (W // 2, 1080), "Both paths are grounded by structured operating data and the private maintenance knowledge base.", F, MUTED)
    return im


def main() -> int:
    svg = HERE / "system_architecture.svg"
    png = HERE / "system_architecture.png"
    pdf = HERE / "system_architecture.pdf"
    svg.write_text(build_svg(), encoding="utf-8")
    im = build_png()
    im.save(png, "PNG", dpi=(300, 300))
    im.save(pdf, "PDF", resolution=300.0)
    print(f"Wrote {svg}")
    print(f"Wrote {png}")
    print(f"Wrote {pdf}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
