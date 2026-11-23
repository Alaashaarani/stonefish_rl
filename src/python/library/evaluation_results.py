"""Portable, independently named evaluation records."""

import csv
import glob
import json
import re
from collections import Counter
from pathlib import Path
from uuid import uuid4


def save_result(directory, summary, episodes, config=None, name=None, note=""):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    identity = uuid4().hex
    name = str(name or summary.get("evaluation_name") or summary["model_name"])
    record = {
        "format_version": 1,
        "evaluation_id": identity,
        "evaluation_name": name,
        "note": str(note or ""),
        "config": config,
        "summary": dict(summary),
        "episodes": [dict(row) for row in episodes],
    }
    # Identity belongs to the record, not to every episode or metric.
    for row in [record["summary"]] + record["episodes"]:
        for key in ("evaluation_name", "evaluation_id", "model_result_dir"):
            row.pop(key, None)
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("._")[:120] or "evaluation"
    path = directory / f"{slug}__{identity}.evaluation.json"
    with path.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2)
    return path


def load_records(directory):
    directory = Path(directory).expanduser()
    if directory.is_file():
        paths = [directory]
    elif directory.is_dir():
        paths = sorted(directory.glob("*.evaluation.json"))
    elif glob.has_magic(str(directory)):
        paths = sorted(path for path in map(Path, glob.glob(str(directory)))
                       if path.is_file() and path.name.endswith(".evaluation.json"))
        if not paths:
            raise FileNotFoundError(f"No evaluation files match: {directory}")
    else:
        raise FileNotFoundError(f"Evaluation input does not exist: {directory}")
    records = []
    for path in paths:
        with path.open(encoding="utf-8") as stream:
            record = json.load(stream)
        if record.get("format_version") != 1:
            raise ValueError(f"Unsupported evaluation format: {path}")
        records.append((path, record))
    return records


def result_directory(source):
    """Choose an output parent for a directory, file or wildcard input."""
    path = Path(source).expanduser().resolve()
    if path.is_dir():
        return path
    path = path.parent
    while glob.has_magic(str(path)):
        path = path.parent
    return path


def result_rows(directory):
    records = load_records(directory)
    counts = Counter(record["evaluation_name"] for _, record in records)
    summaries, episodes = [], []
    for path, record in records:
        name = record["evaluation_name"]
        label = name if counts[name] == 1 else f"{name} [{record['evaluation_id']}]"
        metadata = {
            "evaluation_name": label,
            "evaluation_id": record["evaluation_id"],
            "result_file": str(path),
            "saved_evaluation_name": name,
            "note": record.get("note", ""),
        }
        summaries.append(dict(record["summary"], **metadata))
        episodes.extend(dict(row, **metadata) for row in record["episodes"])
    return summaries, episodes


def update_metadata(path, name, note):
    if not name.strip():
        raise ValueError("Evaluation name cannot be empty")
    path = Path(path)
    record = load_records(path)[0][1]
    record["evaluation_name"] = name.strip()
    record["note"] = note
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2)
    temporary.replace(path)


def migrate_directory(directory, backup):
    """Convert legacy records, verifying contents before archiving originals."""
    import shutil

    directory, backup = Path(directory), Path(backup)
    converted = []
    for summary_path in sorted(directory.rglob("summary.json")):
        with summary_path.open(encoding="utf-8") as stream:
            summary = json.load(stream)
        if not isinstance(summary, dict) or "model_name" not in summary:
            continue
        episode_path = summary_path.with_name("episodes.csv")
        episodes = []
        if episode_path.exists():
            with episode_path.open(newline="", encoding="utf-8") as stream:
                episodes = list(csv.DictReader(stream))
        elif summary.get("num_episodes", 0):
            raise ValueError(f"Missing episodes for {summary_path}")
        # Old shared used_config.json may describe a different run; do not attach it.
        path = save_result(summary_path.parent.parent, summary, episodes)
        record = load_records(path)[0][1]
        expected_summary = dict(summary)
        expected_rows = [dict(row) for row in episodes]
        for row in [expected_summary] + expected_rows:
            for key in ("evaluation_name", "evaluation_id", "model_result_dir"):
                row.pop(key, None)
        assert json.dumps(record["summary"], sort_keys=True) == json.dumps(expected_summary, sort_keys=True)
        assert record["episodes"] == expected_rows
        for old in (summary_path, episode_path):
            if old.exists():
                target = backup / old.relative_to(directory)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(old), str(target))
        if not any(summary_path.parent.iterdir()):
            summary_path.parent.rmdir()
        converted.append(path)
    return converted
