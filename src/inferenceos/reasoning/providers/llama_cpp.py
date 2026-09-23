"""Real local reasoning provider backed by the acquired llama.cpp binary.

Uses JSON-schema constrained generation (``-j``) with the chat template disabled
(``--no-jinja``) so the output is valid JSON even for a small local model. No
network access is used and no other model is substituted on failure.
"""

from __future__ import annotations

import json
from pathlib import Path

from inferenceos.backends.llama_cpp_toolchain import (
    cli_version,
    find_cli,
    parse_generated_text,
    run_cli,
)
from inferenceos.logging_config import get_logger
from inferenceos.reasoning.providers.base import ProviderResult

logger = get_logger("reasoning.providers.llama_cpp")


class LlamaCppReasoningProvider:
    """Provider name in reports/audits."""

    name = "llama.cpp"

    def __init__(
        self,
        *,
        model_path: Path | str | None,
        cli_path: Path | str | None = None,
        llama_cpp_dir: Path | str | None = None,
        context_length: int = 4096,
        n_gpu_layers: int = 99,
    ) -> None:
        self._model_path = Path(model_path) if model_path else None
        self._cli_override = Path(cli_path) if cli_path else None
        self._llama_dir = Path(llama_cpp_dir) if llama_cpp_dir else None
        self._context_length = context_length
        self._n_gpu_layers = n_gpu_layers
        self._cli: Path | None = None

    @property
    def model_id(self) -> str:
        return str(self._model_path) if self._model_path is not None else "<unset>"

    @property
    def cli_path(self) -> Path | None:
        return self._cli

    def _resolve_cli(self) -> Path | None:
        if self._cli is not None:
            return self._cli
        try:
            if self._cli_override is not None:
                self._cli = self._cli_override
            elif self._llama_dir is not None:
                self._cli = find_cli(self._llama_dir)
        except FileNotFoundError:
            return None
        return self._cli

    def is_available(self) -> bool:
        cli = self._resolve_cli()
        if cli is None or not cli.is_file():
            return False
        if self._model_path is None or not self._model_path.is_file():
            return False
        return True

    def complete(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        json_schema: dict[str, object],
        max_tokens: int,
        temperature: float,
        seed: int,
        timeout_s: float,
    ) -> ProviderResult:
        cli = self._resolve_cli()
        if cli is None or not cli.is_file():
            return ProviderResult(success=False, error="llama.cpp CLI not found")
        if self._model_path is None or not self._model_path.is_file():
            return ProviderResult(
                success=False, error=f"reasoning model not found: {self._model_path}"
            )

        prompt = f"{system_prompt}\n\n{user_prompt}"
        try:
            result = run_cli(
                cli,
                self._model_path,
                prompt,
                n_predict=max_tokens,
                seed=seed,
                temperature=temperature,
                n_gpu_layers=self._n_gpu_layers,
                context_length=self._context_length,
                verbose=False,
                log_verbosity=None,
                no_jinja=True,
                json_schema=json.dumps(json_schema),
                timeout_s=timeout_s,
                track_rss=True,
            )
        except TimeoutError:
            return ProviderResult(
                success=False, error=f"reasoning model timed out after {timeout_s:.0f}s"
            )
        except OSError as exc:  # pragma: no cover - environment dependent
            return ProviderResult(success=False, error=f"failed to run reasoner: {exc}")

        version = cli_version(cli)
        if result.returncode != 0:
            tail = "\n".join(result.stderr.splitlines()[-5:])
            return ProviderResult(
                success=False,
                error=f"reasoner exited {result.returncode}: {tail}",
                latency_s=result.duration_s,
                model_id=self.model_id,
                model_version=version,
                peak_rss_bytes=result.peak_rss_bytes,
                command=result.command,
            )

        text = parse_generated_text(result.stdout) or result.stdout
        return ProviderResult(
            success=True,
            text=text.strip(),
            latency_s=result.duration_s,
            model_id=self.model_id,
            model_version=version,
            peak_rss_bytes=result.peak_rss_bytes,
            command=result.command,
        )
