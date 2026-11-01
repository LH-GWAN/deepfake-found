"""Allow the CLI to be run as ``python -m deepshield``."""

from deepshield.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
