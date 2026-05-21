#!/usr/bin/env python

import os
import getpass
from pathlib import Path
from dotenv import dotenv_values

PROJECT_ROOT = Path(__file__).parent
CONFIG_DIR = PROJECT_ROOT / ".userenv"
VENV_DIR = PROJECT_ROOT / ".venv"

# ── Markers so the script is idempotent (safe to re-run) ───────-──────────────
BLOCK_START = "# >>> project env vars >>>"
BLOCK_END   = "# <<< project env vars <<<"


def load_merged_config() -> dict:
    """Merge .env.default → .env.<user> → .env.local (highest priority last)."""
    user = getpass.getuser()
    config = {}
    for env_file in [
        CONFIG_DIR / ".env.default",
        CONFIG_DIR / f".env.{user}",
        CONFIG_DIR / ".env.local",
    ]:
        if env_file.exists():
            config.update(dotenv_values(env_file))
    return config


def make_bash_block(config: dict) -> str:
    lines = [BLOCK_START]
    for key, value in config.items():
        lines.append(f'export {key}="{value}"')
    lines.append(BLOCK_END)
    return "\n".join(lines) + "\n"


def make_fish_block(config: dict) -> str:
    lines = [BLOCK_START]
    for key, value in config.items():
        lines.append(f'set -gx {key} "{value}"')
    lines.append(BLOCK_END)
    return "\n".join(lines) + "\n"


def make_batch_block(config: dict) -> str:
    lines = [BLOCK_START]
    for key, value in config.items():
        lines.append(f'set "{key}={value}"')
    lines.append(BLOCK_END)
    return "\n".join(lines) + "\n"


def make_ps1_block(config: dict) -> str:
    lines = [BLOCK_START]
    for key, value in config.items():
        lines.append(f'$env:{key} = "{value}"')
    lines.append(BLOCK_END)
    return "\n".join(lines) + "\n"


def inject(filepath: Path, block: str):
    """Insert or replace the managed block in an activation script."""
    if not filepath.exists():
        return

    content = filepath.read_text()

    # Remove existing block if present (idempotent)
    if BLOCK_START in content:
        before = content[:content.index(BLOCK_START)]
        after  = content[content.index(BLOCK_END) + len(BLOCK_END):]
        content = before + after

    content = content.rstrip("\n") + "\n\n" + block
    filepath.write_text(content)
    print(f"  updated: {filepath.relative_to(PROJECT_ROOT)}")


def main():
    config = load_merged_config()
    if not config:
        print("No variables found. Check your config/ .env files.")
        return

    print(f"Loaded {len(config)} variable(s) for user '{getpass.getuser()}':")
    for k, v in config.items():
        print(f"  {k}={v}")
    print()

    print("Injecting into venv activation scripts...")
    inject(VENV_DIR / "bin"     / "activate",          make_bash_block(config))  # bash/zsh
    inject(VENV_DIR / "bin"     / "activate.fish",     make_fish_block(config))  # fish
    inject(VENV_DIR / "Scripts" / "activate",          make_bash_block(config))  # Git Bash on Windows
    inject(VENV_DIR / "Scripts" / "activate.bat",      make_batch_block(config)) # Windows CMD
    inject(VENV_DIR / "Scripts" / "Activate.ps1",      make_ps1_block(config))   # PowerShell
    print("\nDone. Re-activate your venv to pick up the changes.")


if __name__ == "__main__":
    main()
    