"""Load tag examples from YAML files or inline strings."""

from __future__ import annotations

from pathlib import Path

import yaml


def load_tag_examples(tag_name: str, spec: str | Path) -> str:
    """Return the example text for *tag_name*.

    - If *spec* is a ``.yaml`` file path: load it and verify the ``tag:`` field
      matches *tag_name*.  Raises ``ValueError`` on mismatch or missing key.
    - Otherwise: return *spec* as-is (inline literal example).
    """
    if isinstance(spec, Path) and spec.suffix == ".yaml":
        data = yaml.safe_load(spec.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"{spec}: expected a YAML mapping with 'tag' key")
        file_tag = str(data.get("tag", "")).strip()
        if not file_tag:
            raise ValueError(f"{spec}: missing 'tag' field")
        if file_tag != tag_name:
            raise ValueError(
                f"Tag mismatch in {spec}: expected '{tag_name}', got '{file_tag}'"
            )
        examples = data.get("examples", "")
        if not isinstance(examples, str):
            examples = str(examples)
        return examples.strip()

    return spec
