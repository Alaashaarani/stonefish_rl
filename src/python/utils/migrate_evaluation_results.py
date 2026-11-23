"""Migrate legacy evaluations, retaining originals in a separate backup."""

if __package__ in (None, ""):
    import _bootstrap  # Enable direct execution from any working directory.


import argparse
import shutil
from datetime import datetime
from pathlib import Path

from library.evaluation_results import migrate_directory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", nargs="?", default="evaluation_results")
    args = parser.parse_args()
    directory = Path(args.directory).resolve()
    backup = directory.with_name(directory.name + "_legacy_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    converted = migrate_directory(directory, backup)
    if converted:
        for name in ("summary.json", "summary.csv", "episodes_all_models.csv", "used_config.json",
                     "current_evaluation_checklist.csv", "current_evaluation_checklist.json",
                     "evaluation_errors.csv", "evaluation_errors.json", "top_models"):
            old = directory / name
            if old.exists():
                backup.mkdir(parents=True, exist_ok=True)
                shutil.move(str(old), str(backup / name))
    print(f"Converted {len(converted)} evaluations. Originals: {backup}")


if __name__ == "__main__":
    main()
