"""
Translations for plot text (English / Swedish).

Usage
-----
    import translations as tr

    tr.set_language("sv")                      # or "en"
    plt.xlabel(tr.t("n_samples"))              # simple lookup
    plt.title(tr.t("title_vs_n_samples", key="S1"))  # with placeholders
    tr.fmt(0.1234)                             # "0,12" (sv) / "0.12" (en)
    tr.localize_axis(ax)                       # decimal comma on tick labels (sv)

To add a new string: add the same key to *both* language dictionaries.
"""

from matplotlib.ticker import FuncFormatter

SUPPORTED_LANGUAGES = ("en", "sv")
_current_language = "en"

TRANSLATIONS = {
    "en": {
        # Legend entries
        "standard_solution": "Standard solution",
        "gp_solution": "GP predicted solution",
        "kde": "KDE",
        "true_value": "True value",
        # Axis labels
        "n_samples": "Number of samples",
        "frequency": "Frequency",
        # Titles
        "fig_vs_samples": "{key} vs samples",  # figure window name
        "title_vs_n_samples": "{key} vs number of samples",
        "quantiles_vs_samples": "Quantiles vs samples",
        "n_equals": "N = {n}",
    },
    "sv": {
        # Legend entries
        "standard_solution": "Standardlösning",
        "gp_solution": "GP-predikterad lösning",
        "kde": "KDE",
        "true_value": "Sant värde",
        # Axis labels
        "n_samples": "Antal stickprov",
        "frequency": "Frekvens",
        # Titles
        "fig_vs_samples": "{key} vs stickprov",  # figure window name
        "title_vs_n_samples": "{key} som funktion av antal stickprov",
        "quantiles_vs_samples": "Kvantiler som funktion av antal stickprov",
        "n_equals": "N = {n}",
    },
}


def set_language(lang):
    """Select the language used by all subsequent calls ("en" or "sv")."""
    global _current_language
    if lang not in SUPPORTED_LANGUAGES:
        raise ValueError(
            f"Unsupported language '{lang}'. Choose one of {SUPPORTED_LANGUAGES}."
        )
    _current_language = lang


def get_language():
    return _current_language


def t(text_id, /, **kwargs):
    """
    Look up a translated string and fill in any {placeholders}.

    text_id is positional-only, so placeholders can have any name,
    including "key" or "text_id".
    """
    try:
        text = TRANSLATIONS[_current_language][text_id]
    except KeyError:
        raise KeyError(
            f"Missing translation for '{text_id}' in language '{_current_language}'."
        ) from None
    return text.format(**kwargs)


def fmt(value, decimals=2, math=False):
    """
    Format a number with the decimal separator of the current language.

    Set math=True when the number is placed inside a mathtext ($...$) string,
    so that the Swedish decimal comma is not rendered with an extra space
    after it ("0{,}05" instead of "0,05").
    """
    s = f"{value:.{decimals}f}"
    if _current_language == "sv":
        s = s.replace(".", "{,}" if math else ",")
    return s


def localize_axis(ax, axis="both"):
    """
    Use decimal commas on the tick labels of a *linear* axis (Swedish only).

    Do not use on log axes - they show powers of ten and have no decimals.

    Parameters
    ----------
    axis : {"both", "x", "y"}
    """
    if _current_language != "sv":
        return

    def _format(value, pos):
        return f"{value:.10g}".replace(".", ",").replace("-", "\u2212")

    formatter = FuncFormatter(_format)
    if axis in ("both", "x"):
        ax.xaxis.set_major_formatter(formatter)
    if axis in ("both", "y"):
        ax.yaxis.set_major_formatter(formatter)
