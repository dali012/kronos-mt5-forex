"""Rendered Telegram image cards for rich companion-bot replies."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any

try:  # Pillow is an optional runtime dependency for rich Telegram cards.
    from PIL import Image, ImageChops, ImageDraw, ImageFont
except ImportError:  # pragma: no cover - exercised only on minimal installs.
    Image = None
    ImageChops = None
    ImageDraw = None
    ImageFont = None


ASSETS = Path(__file__).with_name("assets")
COIN_ASSETS = ASSETS / "coins"

WIDTH = 760
PAD = 34
ROW_H = 66
BG = "#101821"
PANEL = "#162333"
PANEL_LINE = "#26394c"
TEXT = "#eef6ff"
SOFT = "#c7d6e4"
MUTED = "#87a0b6"
GOOD = "#5dffb0"
BAD = "#ff7676"
ZERO = "#eef6ff"


def render_positions_card(snap: dict[str, Any]) -> bytes:
    """Render the current open positions as a Telegram-ready PNG."""
    if Image is None or ImageChops is None or ImageDraw is None or ImageFont is None:
        raise RuntimeError("Pillow is required to render Telegram image cards")

    positions = snap.get("positions", []) or []
    row_count = max(1, len(positions))
    height = PAD * 2 + 134 + row_count * ROW_H + 26

    img = Image.new("RGBA", (WIDTH, height), BG)
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle((0, 0, WIDTH - 1, height - 1), radius=28, fill=PANEL)
    draw.rounded_rectangle((0, 0, WIDTH - 1, height - 1), radius=28, outline=PANEL_LINE, width=2)
    draw.rectangle((0, 0, 6, height), fill=GOOD)

    title_font = _font(34, bold=True)
    meta_font = _font(21)
    label_font = _font(18, bold=True)
    coin_font = _font(24, bold=True)
    value_font = _font(22, mono=True)
    pnl_font = _font(24, bold=True)

    open_count = len(positions)
    total_upnl = sum(_float(p.get("unrealized")) for p in positions)
    side = _basket_side(positions)

    y = PAD
    draw.text((PAD, y), "Positions", font=title_font, fill=TEXT)
    open_text = f"{open_count} open"
    open_w = _text_w(draw, open_text, label_font)
    pill_x = WIDTH - PAD - open_w - 28
    draw.rounded_rectangle(
        (pill_x, y + 4, WIDTH - PAD, y + 38),
        radius=17,
        fill="#213247",
        outline="#31516a",
        width=2,
    )
    draw.text((pill_x + 14, y + 9), open_text, font=label_font, fill="#9ed1ff")

    y += 52
    summary = f"{side} | uPnL"
    draw.text((PAD, y), summary, font=meta_font, fill=SOFT)
    upnl_text = _signed(total_upnl)
    draw.text(
        (PAD + _text_w(draw, summary, meta_font) + 10, y),
        upnl_text,
        font=meta_font,
        fill=_pnl_color(total_upnl),
    )

    y += 58
    draw.text((PAD, y), "Coin", font=label_font, fill=MUTED)
    draw.text((PAD + 88, y), "Size", font=label_font, fill=MUTED)
    pnl_header = "uPnL"
    draw.text(
        (WIDTH - PAD - _text_w(draw, pnl_header, label_font), y),
        pnl_header,
        font=label_font,
        fill=MUTED,
    )

    y += 30
    draw.line((PAD, y, WIDTH - PAD, y), fill=PANEL_LINE, width=2)
    y += 12

    if not positions:
        draw.text((PAD, y + 14), "Flat - no open positions", font=meta_font, fill=SOFT)
    for pos in positions:
        _draw_position_row(draw, img, pos, y, coin_font, value_font, pnl_font)
        y += ROW_H

    out = BytesIO()
    img.convert("RGB").save(out, format="PNG", optimize=True)
    return out.getvalue()


def _draw_position_row(
    draw: Any,
    img: Any,
    pos: dict[str, Any],
    y: int,
    coin_font: Any,
    value_font: Any,
    pnl_font: Any,
) -> None:
    coin = _coin(pos.get("symbol", ""))
    qty = _float(pos.get("qty"))
    upnl = _float(pos.get("unrealized"))

    logo_x = PAD
    logo_y = y + 7
    _paste_coin_logo(img, draw, coin, logo_x, logo_y, 42)

    text_x = PAD + 58
    draw.text((text_x, y + 3), coin, font=coin_font, fill=TEXT)
    draw.text((text_x, y + 34), _qty(qty), font=value_font, fill=SOFT)

    pnl = _signed(upnl)
    pnl_w = _text_w(draw, pnl, pnl_font)
    draw.text((WIDTH - PAD - pnl_w, y + 17), pnl, font=pnl_font, fill=_pnl_color(upnl))

    draw.line((text_x, y + ROW_H - 2, WIDTH - PAD, y + ROW_H - 2), fill="#203145", width=1)


def _paste_coin_logo(img: Any, draw: Any, coin: str, x: int, y: int, size: int) -> None:
    logo_path = COIN_ASSETS / f"{coin.lower()}.png"
    if logo_path.exists():
        logo = Image.open(logo_path).convert("RGBA").resize((size, size), Image.Resampling.LANCZOS)
        mask = Image.new("L", (size, size), 0)
        mask_draw = ImageDraw.Draw(mask)
        mask_draw.ellipse((0, 0, size - 1, size - 1), fill=255)
        mask = ImageChops.multiply(mask, logo.getchannel("A"))
        img.paste(logo, (x, y), mask)
        return

    draw.ellipse((x, y, x + size, y + size), fill="#26384a", outline="#41566d", width=2)
    fallback_font = _font(15, bold=True)
    initials = coin[:2] or "?"
    w = _text_w(draw, initials, fallback_font)
    draw.text((x + (size - w) / 2, y + 12), initials, font=fallback_font, fill="#9ed1ff")


def _font(size: int, *, bold: bool = False, mono: bool = False) -> Any:
    candidates = []
    if mono:
        candidates.extend(
            [
                "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
                "/usr/share/fonts/dejavu/DejaVuSansMono.ttf",
            ]
        )
    elif bold:
        candidates.extend(
            [
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
            ]
        )
    candidates.extend(
        [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/dejavu/DejaVuSans.ttf",
        ]
    )
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def _text_w(draw: Any, text: str, font: Any) -> int:
    box = draw.textbbox((0, 0), text, font=font)
    return box[2] - box[0]


def _coin(symbol: str) -> str:
    return str(symbol).split("USDT")[0].replace("-PERP", "").upper()


def _basket_side(positions: list[dict[str, Any]]) -> str:
    qtys = [_float(p.get("qty")) for p in positions]
    if qtys and all(q < 0 for q in qtys):
        return "Short basket"
    if qtys and all(q > 0 for q in qtys):
        return "Long basket"
    return "Mixed basket"


def _float(value: Any) -> float:
    try:
        return float(value or 0)
    except (TypeError, ValueError):
        return 0.0


def _qty(value: float) -> str:
    return f"{value:.4f}"


def _signed(value: float) -> str:
    sign = "+" if value >= 0 else "-"
    return f"{sign}${abs(value):,.0f}"


def _pnl_color(value: float) -> str:
    if value > 0:
        return GOOD
    if value < 0:
        return BAD
    return ZERO
