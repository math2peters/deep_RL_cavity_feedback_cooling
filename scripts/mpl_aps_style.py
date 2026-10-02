"""APS / PRX-safe matplotlib defaults for figure PDFs.

APS production rejects Type 3 (bitmap) fonts. Matplotlib's default
``pdf.fonttype`` is 3; set it to 42 (TrueType) before saving PDFs.

Pair Times text with Times mathtext so ``$...$`` fragments (Delta, mu,
pi, superscripts) do not render in a mismatched sans-serif face.

Figures are drawn at their final printed size (revtex4-2 ``reprint``:
``\\textwidth`` = 510 pt, ``\\columnwidth`` = 246 pt) so that the font sizes
below are the sizes that appear in the paper, identical in every figure.

The sizes follow the APS Journals Style Guide (Sec. S, Figures): at journal size,
lettering height >= 2 mm, data points >= 1 mm across, and lines >= 0.5 pt.
Times New Roman capitals and numerals are 0.662 em tall, so 2 mm needs 8.6 pt.
"""

TEXT_WIDTH_IN = 510 / 72.27  # \textwidth, figure* environments
COLUMN_WIDTH_IN = 246 / 72.27  # \columnwidth, single-column figures

LABEL_SIZE = 10  # axis labels, titles
TICK_SIZE = 9  # tick labels, legends, annotations (smallest text)
PANEL_LABEL_SIZE = 10  # bold (a), (b), ...

MIN_LETTER_HEIGHT_MM = 2.0  # APS: capitals and numerals at journal size
CAP_HEIGHT_EM = 0.662  # Times New Roman capitals and numerals

APS_PDF_RC = {
    "pdf.fonttype": 42,  # TrueType (not Type 3)
    "ps.fonttype": 42,
    "mathtext.fontset": "custom",
    "mathtext.rm": "Times New Roman",
    "mathtext.it": "Times New Roman:italic",
    "mathtext.bf": "Times New Roman:bold",
}

PAPER_RC = {
    "font.family": "Times New Roman",
    "font.size": LABEL_SIZE,
    "axes.labelsize": LABEL_SIZE,
    "axes.titlesize": LABEL_SIZE,
    "figure.titlesize": LABEL_SIZE,
    "xtick.labelsize": TICK_SIZE,
    "ytick.labelsize": TICK_SIZE,
    "legend.fontsize": TICK_SIZE,
    "legend.title_fontsize": TICK_SIZE,
    "axes.linewidth": 0.6,
    "axes.labelpad": 2.0,
    "axes.titlepad": 3.0,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.size": 3.0,
    "ytick.major.size": 3.0,
    "xtick.minor.size": 1.8,
    "ytick.minor.size": 1.8,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.minor.width": 0.5,
    "ytick.minor.width": 0.5,
    "xtick.major.pad": 2.0,
    "ytick.major.pad": 2.0,
    "lines.linewidth": 1.0,
    "patch.linewidth": 0.5,
    "lines.markersize": 3.0,
    "errorbar.capsize": 1.5,
    "axes.grid": True,
    "grid.linewidth": 0.5,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "legend.frameon": True,
    "legend.framealpha": 0.9,
    "legend.borderpad": 0.3,
    "legend.borderaxespad": 0.4,
    "legend.labelspacing": 0.25,
    "legend.handlelength": 1.6,
    "legend.handletextpad": 0.5,
    "legend.columnspacing": 1.2,
    "figure.constrained_layout.h_pad": 0.02,
    "figure.constrained_layout.w_pad": 0.02,
    "savefig.dpi": 600,
    **APS_PDF_RC,
}


def apply_aps_pdf_style():
    """Apply APS-safe PDF/mathtext settings to the current rcParams."""
    import matplotlib.pyplot as plt

    plt.rcParams.update(APS_PDF_RC)


def apply_paper_style():
    """Apply the shared print-size style used by every paper figure."""
    import matplotlib.pyplot as plt

    plt.style.use("seaborn-v0_8-paper")
    plt.rcParams.update(PAPER_RC)


def add_panel_label(ax, label, dx=-30, dy=4, **kwargs):
    """Place a bold panel label offset (in points) from the axes' top-left corner."""
    return ax.annotate(
        label,
        xy=(0, 1),
        xycoords="axes fraction",
        xytext=(dx, dy),
        textcoords="offset points",
        fontsize=PANEL_LABEL_SIZE,
        fontweight="bold",
        ha="left",
        va="bottom",
        **kwargs,
    )


def check_lettering(fig):
    """Raise if any visible text would print smaller than the APS 2 mm minimum.

    Assumes the figure is drawn at print size. Checks each Text's font size; math
    sub/superscripts inside a label print at 70% of it, as in the body text.
    """
    from matplotlib.text import Text

    min_size_pt = MIN_LETTER_HEIGHT_MM / 25.4 * 72 / CAP_HEIGHT_EM
    too_small = sorted({
        (round(t.get_fontsize(), 2), t.get_text())
        for t in fig.findobj(Text)
        if t.get_visible() and t.get_text().strip() and t.get_fontsize() < min_size_pt
    })
    if too_small:
        raise ValueError(f"Text below {MIN_LETTER_HEIGHT_MM} mm ({min_size_pt:.2f} pt): {too_small}")
