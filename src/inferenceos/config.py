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


def default_reasoning_model() -> Path | None:
    """The locally available instruct model used as the default reasoner.

    Returns ``None`` when it is not present, so the engine reports itself
    unavailable rather than silently substituting a different model.
    """
    candidate = _rooted("models", "qwen2.5-1.5b-instruct-q4_k_m.gguf")
    return candidate if candidate.is_file() else None


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

    # Phase 6 candidate generation / controlled optimization policy.
    phase6_max_candidates: int = 8
    phase6_min_diagnosis_confidence: float = 0.0
    phase6_min_evidence_strength: str = "low"
    phase6_context_reduction_fraction: float = 0.5
    phase6_min_context_length: int = 512
    phase6_autonomous_max_risk: str = "medium"
    phase6_allow_reasoning: bool = True
    phase6_max_optimizer_overhead_s: float = 120.0
    phase6_experiment_repetitions: int = 4
    phase6_weight_benefit: float = 0.30
    phase6_weight_uncertainty: float = 0.08
    phase6_weight_cost: float = 0.15
    phase6_weight_overhead: float = 0.05
    phase6_weight_risk: float = 0.15
    phase6_weight_reversibility: float = 0.07
    phase6_weight_confidence: float = 0.08
    phase6_weight_causal_strength: float = 0.06
    phase6_weight_scope: float = 0.03
    phase6_weight_approval: float = 0.02
    phase6_weight_precondition: float = 0.01

    # Phase 5 local reasoning (deep-diagnosis mode).
    reasoning_enabled: bool = True
    reasoning_provider: str = "llama_cpp"
    reasoning_model_path: Path | None = Field(default_factory=default_reasoning_model)
    reasoning_cli_path: Path | None = None
    reasoning_context_length: int = 8192
    reasoning_gpu_layers: int = 99
    reasoning_max_tokens: int = 1024
    reasoning_temperature: float = 0.0
    reasoning_seed: int = 1234
    reasoning_timeout_s: float = 300.0
    reasoning_escalation_threshold: float = 0.6
    reasoning_max_evidence: int = 12

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
