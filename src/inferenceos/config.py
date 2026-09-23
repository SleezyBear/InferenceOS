"""Runtime configuration for InferenceOS.

Configuration is intentionally small in Phase 0: it locates the llama.cpp toolchain,
the model directory, and the artifact directories used for reports/logs/audit data.
Values may be overridden with ``INFERENCEOS_``-prefixed environment variables or a
``.env`` file at the project root.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def find_project_root() -> Path:
    """Walk up from this file to the directory containing ``pyproject.toml``."""
    here = Path(__file__).resolve()
    for candidate in (here, *here.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    # Fallback: src/inferenceos/config.py -> project root is three levels up.
    return here.parents[2]


def _rooted(*parts: str) -> Path:
    return find_project_root().joinpath(*parts)


class Settings(BaseSettings):
    """Resolved, validated configuration for the current machine."""

    model_config = SettingsConfigDict(
        env_prefix="INFERENCEOS_",
        env_file=".env",
        extra="ignore",
    )

    project_root: Path = Field(default_factory=find_project_root)
    tools_dir: Path = Field(default_factory=lambda: _rooted("tools"))
    llama_cpp_dir: Path = Field(default_factory=lambda: _rooted("tools", "llama.cpp"))
    models_dir: Path = Field(default_factory=lambda: _rooted("models"))
    reports_dir: Path = Field(default_factory=lambda: _rooted("reports"))
    logs_dir: Path = Field(default_factory=lambda: _rooted("logs"))
    data_dir: Path = Field(default_factory=lambda: _rooted("data"))

    log_level: str = "INFO"

    # Default inference knobs. ``default_threads == 0`` means "let llama.cpp decide".
    default_gpu_layers: int = 99
    default_threads: int = 0
    gpu_sample_interval_s: float = 0.5

    # Phase 1 live telemetry.
    telemetry_interval_s: float = 0.6
    telemetry_top_processes: int = 8

    def ensure_directories(self) -> None:
        """Create the artifact directories if they do not yet exist."""
        for directory in (
            self.tools_dir,
            self.llama_cpp_dir,
            self.models_dir,
            self.reports_dir,
            self.logs_dir,
            self.data_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)
