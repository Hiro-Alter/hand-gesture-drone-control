from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def project_path(relative_path: str | Path) -> Path:
    path = Path(relative_path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path

