"""Thin config proxy that loads self-contained profiles from run_configs/."""

import importlib.util
from pathlib import Path

_RUN_CONFIGS_DIR = Path(__file__).parent / "run_configs"


class ConfigProxy:
    """Delegates all attribute access to the loaded profile module.

    Auto-loads run_configs/config_default.py on construction.
    Call use_profile to switch profiles at runtime.

    """

    _profile_name: str = "default"

    def __init__(self) -> None:
        self._mod = self._load("config_default")

    # -- public API ----------------------------------------------------------

    def use_profile(self, name: str) -> None:
        """Load a profile by filename stem (without .py)."""
        self._profile_name = name
        self._mod = self._load(f"config_{name}")

    # -- internals -----------------------------------------------------------

    @staticmethod
    def _load(name: str) -> object:
        path = _RUN_CONFIGS_DIR / f"{name}.py"
        if not path.exists():
            raise FileNotFoundError(
                f"Config profile not found: {path}\n"
                f"Ensure run_configs/{name}.py exists."
            )
        spec = importlib.util.spec_from_file_location(f"profile_{id(path)}", path)
        mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
        return mod

    def __getattr__(self, name: str) -> object:
        try:
            return getattr(self._mod, name)
        except AttributeError:
            raise KeyError(
                f"Missing config key '{name}' in profile '{self._profile_name}'. "
                f"Ensure run_configs/config_{self._profile_name}.py defines it."
            ) from None


cfg = ConfigProxy()
