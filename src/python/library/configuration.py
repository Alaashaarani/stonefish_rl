"""Resolve shared experiment settings plus a workflow-specific override."""

from copy import deepcopy

import yaml

from library.common import resolve_path

DEFAULT_CONFIG = "include/parameters/stonefish_rl.yaml"


def _merge(base, override):
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def resolve_config(document, mode):
    if mode not in ("training", "evaluation", "testing"):
        raise ValueError(f"Unknown configuration mode: {mode}")
    if not isinstance(document, dict):
        raise ValueError("Configuration must be a YAML mapping")
    if "common" not in document:
        return deepcopy(document)  # Existing flat YAMLs and generated run configs.
    common = document["common"]
    overrides = document.get(mode, {})
    if not isinstance(common, dict) or not isinstance(overrides, dict):
        raise ValueError(f"common and {mode} must be mappings")
    config = _merge(common, overrides)
    if mode != "training":
        # Never inherit the training worker count through the common section.
        instances = overrides.get("env", {}).get("instances", 1)
        if isinstance(instances, bool) or not isinstance(instances, int) or instances < 1:
            raise ValueError(f"{mode}.env.instances must be a positive integer")
        config.setdefault("env", {})["instances"] = instances
    if mode == "evaluation":
        compare = config.setdefault("compare", {})
        compare.setdefault("parallel", False)
        compare.setdefault("max_workers", config["env"]["instances"])
    return config


def load_config(path, mode):
    with open(resolve_path(path), encoding="utf-8") as stream:
        return resolve_config(yaml.safe_load(stream), mode)
