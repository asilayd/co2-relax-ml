"""Единый стиль рисунков для отчёта.

Палитра проверена на различимость при дальтонизме (дейтан/протан/тритан)
и на контраст с фоном, поэтому цвет в рисунках несёт смысл и при печати
в оттенках серого дублируется маркером или подписью.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "reports" / "figures"

# --- палитра -----------------------------------------------------------
BLUE, ORANGE, PURPLE, GREEN = "#3b6ea5", "#c4622d", "#8a5fa8", "#4f9153"
INK, MUTED, GRID = "#1b1b1b", "#5c5c5c", "#d8d8d4"

# канал -> цвет и маркер (маркер дублирует цвет для ч/б печати)
CHANNEL = {
    "R_VT2_12":   dict(c=BLUE,   m="o", label="VT2"),
    "R_VV23_12":  dict(c=ORANGE, m="s", label="VV23"),
    "R_VV123_12": dict(c=PURPLE, m="^", label="VV123"),
}

MODEL = {
    "сеть с физикой":  dict(c=BLUE,   m="o"),
    "сплайн по R":     dict(c=ORANGE, m="s"),
    "сеть без физики": dict(c=PURPLE, m="^"),
    "сплайн по A":     dict(c=GREEN,  m="D"),
}

# последовательная шкала (модуль величины): один тон, светлый -> тёмный
SEQ = LinearSegmentedColormap.from_list("seq", ["#f2f6fa", BLUE, "#1d3d5c"])
# расходящаяся шкала (знак величины): два тона через нейтральный серый
DIV = LinearSegmentedColormap.from_list("div", [ORANGE, "#f0efec", BLUE])


def use_style() -> None:
    mpl.rcParams.update({
        "figure.dpi": 130,
        "savefig.dpi": 200,
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 8.5,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "lines.linewidth": 1.6,
        "lines.markersize": 4.5,
        "legend.frameon": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })


def set_out(path: Path | None) -> None:
    """Меняет папку, куда сохраняются рисунки."""
    global FIG_DIR
    if path is not None:
        FIG_DIR = path


def out_dir() -> Path:
    return FIG_DIR


def save(fig, name: str) -> None:
    """Сохраняет в PDF (вектор, для LaTeX) и PNG (для быстрого просмотра)."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    try:                       # разводит подписи осей между панелями
        fig.tight_layout()
    except Exception:
        pass
    for ext in ("pdf", "png"):
        fig.savefig(FIG_DIR / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"  {name}.pdf")
