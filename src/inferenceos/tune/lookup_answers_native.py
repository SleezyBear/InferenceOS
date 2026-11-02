"""Stage-B answer adapter over Stage-A's existing single lifecycle owner.

No backend factory, launch policy, kill operation or cleanup retry owner is added.
The enclosing collector must still enforce its reviewed identity and workload
gates.  This adapter supplies bounded read/stream transport, not authorization.
"""
from __future__ import annotations

import threading
import time
from typing import Any, Callable

from inferenceos.tune.lookup_answer_transport import (AnswerCancelled, AnswerTimeout,
    bounded_operation, stream_completion)
from inferenceos.tune.lookup_fit import FitModel, FitRefused


class NativeAnswerEngine:
    """Compose an accepted ``LiveFitEngine``; keep process ownership there."""
    def __init__(self, fit_engine: Any, models: tuple[FitModel, ...], *,
                 start_timeout_s: float = 120, observe_timeout_s: float = 30, tokenize_timeout_s: float = 30,
                 answer_timeout_s: float = 120) -> None:
        self.fit_engine = fit_engine
        self.models: dict[str, FitModel] = {model.label: model for model in models}
        if len(self.models) != len(models) or set(self.models) != {"llama3b", "gemma1b"}:
            raise FitRefused("exact two-model native answer identity required")
        self.cancelled: threading.Event = fit_engine.cancelled
        if not isinstance(self.cancelled, threading.Event):
            raise FitRefused("shared lifecycle cancellation event required")
        for value, limit in ((start_timeout_s, 120), (observe_timeout_s, 30), (tokenize_timeout_s, 30), (answer_timeout_s, 120)):
            if type(value) not in (int, float) or not 0 < value <= limit:
                raise ValueError("native timeout may shorten, never extend, its registered limit")
        self.start_timeout_s = start_timeout_s
        self.observe_timeout_s = observe_timeout_s
        self.tokenize_timeout_s = tokenize_timeout_s
        self.answer_timeout_s = answer_timeout_s
        self._operation = threading.Lock()
        self._answer_used = False

    @property
    def backend(self) -> Any:
        return self.fit_engine.backend

    def cancel(self) -> None:
        # LiveFitEngine.start_if_current consults this exact event, including a
        # startup worker whose bounded caller has already stopped waiting.
        self.cancelled.set()

    def _active(self) -> None:
        if self.cancelled.is_set() or self.fit_engine.owed is not True:
            raise AnswerCancelled("native answer admission closed or no owned load")

    def _claim(self) -> None:
        if not self._operation.acquire(blocking=False):
            self.cancel()
            raise FitRefused("concurrent native study operations refused")

    def start(self, model: str, arm: str) -> None:
        deadline = time.monotonic() + self.start_timeout_s
        self._claim()
        try:
            def admission() -> None:
                if self.cancelled.is_set() or model not in self.models or arm not in ("control", "lookup"):
                    raise FitRefused("native load admission closed or unknown model/arm")
                if (self.fit_engine.owed or self.backend.is_active() or self.backend.unconfirmed_pid is not None
                        or self.backend.api_key_file is not None or self.backend.undeleted_key_files):
                    raise FitRefused("prior native process or credential remains owned")
                if self.cancelled.is_set() or time.monotonic() >= deadline:
                    raise AnswerTimeout("native start allowance exhausted before owner dispatch")
            left = deadline-time.monotonic()
            if left <= 0:
                raise AnswerTimeout("native start allowance exhausted before owner dispatch")
            bounded_operation(admission, timeout_s=left, cancelled=self.cancelled)
            # The existing owner bounds each of its checks and its start worker
            # by this same deadline. A second opaque outer worker would turn an
            # original startup error into cancellation when the owner closes its
            # shared event before returning that error.
            self.fit_engine.start(self.models[model], arm, deadline=deadline)
            if time.monotonic() >= deadline:
                raise AnswerTimeout("late native start completion refused")
            self._active()
            self._answer_used = False
        except BaseException:
            self.cancel()
            raise
        finally:
            self._operation.release()

    def observe(self) -> dict[str, Any]:
        self._claim()
        try:
            self._active()
            value = bounded_operation(self.fit_engine.observe, timeout_s=self.observe_timeout_s,
                                      cancelled=self.cancelled)
            if not isinstance(value, dict):
                raise FitRefused("native observation is not an object")
            return value
        except BaseException:
            self.cancel()
            raise
        finally:
            self._operation.release()

    def tokenize(self, text: str, *, add_special: bool) -> list[int]:
        self._claim()
        try:
            self._active()
            if self._answer_used:
                raise FitRefused("tokenization after measured generation refused")
            value = bounded_operation(lambda: self.fit_engine.tokenize(text, add_special=add_special),
                                      timeout_s=self.tokenize_timeout_s, cancelled=self.cancelled)
            if (not isinstance(value, list) or not value or len(value) > 65536
                    or any(type(token) is not int or token < 0 for token in value)):
                raise FitRefused("complete nonempty native tokenizer ID list required")
            return value
        except BaseException:
            self.cancel()
            raise
        finally:
            self._operation.release()

    def answer(self, body: dict[str, Any], chunk: Callable[[dict[str, Any]], None]) -> None:
        self._claim()
        try:
            self._active()
            if self._answer_used:
                raise FitRefused("one measured native request per fresh load required")
            self._answer_used = True
            deadline = time.monotonic() + self.answer_timeout_s

            def credentials() -> tuple[str, str]:
                self._active()
                url, path = self.backend.base_url, self.backend.api_key_file
                if not isinstance(url, str) or path is None:
                    raise FitRefused("owned native endpoint/password unavailable")
                # Bounded read; the tiny generated ASCII key must not become an
                # unbounded file input. A late read cannot dispatch HTTP itself.
                with path.open("rb") as stream:
                    raw = stream.read(4097)
                if len(raw) > 4096:
                    raise FitRefused("native password file exceeds byte limit")
                key = raw.decode("ascii", "strict").strip()
                if not key or "\r" in key or "\n" in key:
                    raise FitRefused("native password is empty or contains a header delimiter")
                self._active()
                return url, key

            url, key = bounded_operation(credentials, timeout_s=self.answer_timeout_s,
                                         cancelled=self.cancelled)
            left = deadline - time.monotonic()
            if left <= 0:
                raise AnswerTimeout("credential read exhausted native request allowance")
            stream_completion(url, key, body, chunk, cancelled=self.cancelled, timeout_s=left)
            if time.monotonic() >= deadline:
                raise AnswerTimeout("late native answer completion refused")
            self._active()
        except BaseException:
            self.cancel()
            raise
        finally:
            self._operation.release()

    def release(self, *, abort: bool) -> dict[str, Any]:
        # Release intentionally remains callable after cancellation. Only the
        # existing fit/backend owner may judge process and credential disposal.
        if abort or self._operation.locked():
            self.cancel()
            abort = True
        try:
            receipt = self.fit_engine.release(abort=abort)
            if (not isinstance(receipt, dict) or receipt.get("process_gone") is not True
                    or receipt.get("credential_removed") is not True or receipt.get("problem") is not None):
                self.cancel()
            return receipt  # type: ignore[no-any-return]
        except BaseException:
            self.cancel()
            raise

    def retry_held(self) -> dict[str, Any] | None:
        try:
            return self.fit_engine.retry_held()  # type: ignore[no-any-return]
        except BaseException:
            self.cancel()
            raise
