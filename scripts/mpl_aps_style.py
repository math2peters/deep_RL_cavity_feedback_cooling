"""APS / PRX-safe matplotlib defaults for figure PDFs.

APS production rejects Type 3 (bitmap) fonts. Matplotlib's default
``pdf.fonttype`` is 3; set it to 42 (TrueType) before saving PDFs.

Pair Times text with Times mathtext so ``$...$`` fragments (Delta, mu,
pi, superscripts) do not render in a mismatched sans-serif face.

Figures are drawn at their final printed size (revtex4-2 ``reprint``:
``\\textwidth`` = 510 pt, ``\\columnwidth`` = 246 pt) so that the font sizes
below are the sizes that appear in the paper, identical in every figure.
"""

TEXT_WIDTH_IN = 510 / 72.27  # \textwidth, figure* environments
COLUMN_WIDTH_IN = 246 / 72.27  # \columnwidth, single-column figures

LABEL_SIZE = 8  # axis labels, titles
TICK_SIZE = 7  # tick labels, legends, annotations
PANEL_LABEL_SIZE = 9  # bold (a), (b), ...

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
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "xtick.minor.size": 1.5,
    "ytick.minor.size": 1.5,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.minor.width": 0.4,
    "ytick.minor.width": 0.4,
    "xtick.major.pad": 2.0,
    "ytick.major.pad": 2.0,
    "lines.linewidth": 1.0,
    "lines.markersize": 3.0,
    "errorbar.capsize": 1.5,
    "axes.grid": True,
    "grid.linewidth": 0.4,
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


def add_panel_label(ax, label, dx=-24, dy=4, **kwargs):
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
