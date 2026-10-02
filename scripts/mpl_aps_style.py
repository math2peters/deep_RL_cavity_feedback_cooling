"""APS / PRX-safe matplotlib defaults for figure PDFs.

APS production rejects Type 3 (bitmap) fonts. Matplotlib's default
``pdf.fonttype`` is 3; set it to 42 (TrueType) before saving PDFs.

Pair Times text with Times mathtext so ``$...$`` fragments (Delta, mu,
pi, superscripts) do not render in a mismatched sans-serif face.
"""

APS_PDF_RC = {
    "pdf.fonttype": 42,  # TrueType (not Type 3)
    "ps.fonttype": 42,
    "mathtext.fontset": "custom",
    "mathtext.rm": "Times New Roman",
    "mathtext.it": "Times New Roman:italic",
    "mathtext.bf": "Times New Roman:bold",
}


def apply_aps_pdf_style():
    """Apply APS-safe PDF/mathtext settings to the current rcParams."""
    import matplotlib.pyplot as plt

    plt.rcParams.update(APS_PDF_RC)
