from pathlib import Path

def package_root() -> Path:
    """Directory containing the installed enn555 package."""
    return Path(__file__).resolve().parent


def repo_root() -> Path:
    """
    Root of the git repository (works for editable installs).
    """
    return package_root().parents[1]


def _ensure_dir(path: Path) -> Path:
    """Create directory if it does not exist and return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def data_dir(create: bool = True) -> Path:
    """
    Path to the repo data directory.

    Parameters
    ----------
    create : bool
        If True, create the directory if it does not exist.
        If False and missing, raise an informative error.
    """
    path = repo_root() / "data"

    if path.exists():
        return path

    if create:
        return _ensure_dir(path)

    raise FileNotFoundError(
        f"Data directory not found: {path}\n"
        "Create it or run data_dir(create=True)."
    )


def outputs_dir() -> Path:
    """
    Path to the repo outputs directory (always created).
    Safe place for figures and generated files.
    """
    return _ensure_dir(repo_root() / "outputs")