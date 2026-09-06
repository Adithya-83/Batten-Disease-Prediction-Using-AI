"""Command-line entry point for the Batten disease prediction project."""

from __future__ import annotations

from src.config import ensure_directories


def main() -> None:
    """Ensure required folders exist and print a starter message."""
    ensure_directories()
    print("Project structure initialized.")
    print("Place the real MRI dataset in data/raw/ and configure the dataset classes in src/config.py.")


if __name__ == "__main__":
    main()
