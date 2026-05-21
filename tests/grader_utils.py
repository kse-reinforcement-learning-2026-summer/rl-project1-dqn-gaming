import nbformat
from types import ModuleType


# Patterns that indicate cells to skip (training loops, plotting, etc.)
SKIP_PATTERNS = [
    "for step in range",   # training loops
    "plt.show()",          # visualization
    "plt.imshow(",         # visualization
    "HTML(",               # video rendering
    "FuncAnimation(",      # animation
    "torch.save(",         # model saving
    "drive.mount(",        # Google Drive
    "!uv pip install",     # pip installs
    "!cp ",                # shell copy commands
]


def load_notebook_functions(path: str, skip_patterns: list[str] | None = None) -> ModuleType:
    """Load a Jupyter notebook and execute its code cells into a fresh module.

    Cells matching any skip_pattern (substring match) are skipped.
    Cells that error during execution are also skipped silently.
    """
    if skip_patterns is None:
        skip_patterns = SKIP_PATTERNS

    with open(path, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    mod = ModuleType("lab")

    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        if any(pat in cell.source for pat in skip_patterns):
            continue
        try:
            exec(cell.source, mod.__dict__)
        except Exception:
            continue

    return mod
