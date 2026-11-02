"""Bounded native /completion SSE for a separately authorized Stage-B collector.

This module owns an HTTP connection, never an engine or process.  The caller's
cancellation event closes admission permanently; an abandoned read cannot publish
chunks or dispatch another request.  Deadlines include connection, headers, body
and framing, rather than granting a fresh allowance for every arriving byte.
"""
from __future__ import annotations

import http.client
import json
import math
from queue import Empty, Full, Queue
import socket
import threading
import time
from typing import Any, Callable, TypeVar
from urllib.parse import urlsplit

from inferenceos.tune.lookup_fit import FitRefused, strict_json

T = TypeVar("T")


class AnswerTimeout(TimeoutError):
    """The complete operation's wall-clock allowance expired."""


class AnswerCancelled(FitRefused):
    """The existing lifecycle owner closed native admission."""


def bounded_operation(call: Callable[[], T], *, timeout_s: float,
                      cancelled: threading.Event) -> T:
    """Bound an observation/read; a late daemon has no publication authority."""
    if type(timeout_s) not in (int, float) or not math.isfinite(timeout_s) or timeout_s <= 0:
        raise ValueError("positive finite total deadline required")
    deadline = time.monotonic() + timeout_s
    result: Queue[tuple[bool, Any]] = Queue(maxsize=1)
    if cancelled.is_set():
        raise AnswerCancelled("native admission closed")

    def worker() -> None:
        try:
            value = call()
            result.put((True, value))
        except BaseException as exc:
            result.put((False, exc))

    threading.Thread(target=worker, daemon=True, name="lookup-answer-observation").start()
    try:
        while True:
            if cancelled.is_set():
                raise AnswerCancelled("native operation cancelled")
            left = deadline - time.monotonic()
            if left <= 0:
                raise AnswerTimeout("native operation total deadline expired")
            try:
                success, value = result.get(timeout=min(left, .02))
            except Empty:
                continue
            if time.monotonic() >= deadline:
                raise AnswerTimeout("late native operation reply refused")
            if cancelled.is_set():
                raise AnswerCancelled("native operation cancelled before publication")
            if not success:
                raise value
            return value  # type: ignore[no-any-return]
    except BaseException:
        cancelled.set()
        raise


class _Connection(http.client.HTTPConnection):
    """Check cancellation after a delayed connect and before every socket send."""
    def __init__(self, host: str, port: int | None, *, remaining: Callable[[], float],
                 remember: Callable[[socket.socket], None]) -> None:
        self.remaining = remaining
        self.remember = remember
        self._defer_close = False
        super().__init__(host, port, timeout=remaining())

    def connect(self) -> None:
        self.timeout = self.remaining()
        super().connect()
        assert self.sock is not None
        self.remember(self.sock)
        self.sock.settimeout(self.remaining())

    def send(self, data: Any) -> None:
        self.remaining()
        if self.sock is not None:
            self.sock.settimeout(self.remaining())
        super().send(data)

    def getresponse(self) -> http.client.HTTPResponse:
        # HTTPConnection normally drops its socket reference immediately for a
        # Connection: close response. Keep that reference until our one-shot body
        # consumer finishes, so a cancellation can still shutdown a blocked read.
        self._defer_close = True
        try:
            return super().getresponse()
        finally:
            self._defer_close = False

    def close(self) -> None:
        if not self._defer_close:
            super().close()


def _request_bytes(body: dict[str, Any], maximum_bytes: int) -> bytes:
    required = {"prompt", "n_predict", "seed", "temperature", "stop", "stream",
                "cache_prompt", "ignore_eos", "return_tokens"}
    if (not isinstance(body, dict) or set(body) != required
            or not isinstance(body.get("prompt"), str)
            or type(body.get("n_predict")) is not int or body["n_predict"] != 512
            or type(body.get("seed")) is not int or body["seed"] != 42
            or type(body.get("temperature")) not in (int, float) or body["temperature"] != 0
            or body.get("stop") != []
            or body.get("stream") is not True or body.get("cache_prompt") is not False
            or body.get("ignore_eos") is not False or body.get("return_tokens") is not True):
        raise FitRefused("exact fresh, bounded native Stage-B request required")
    encoded = json.dumps(body, allow_nan=False, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    if len(encoded) > maximum_bytes:
        raise FitRefused("native request exceeds byte limit")
    return encoded


def stream_completion(base_url: str, key: str, body: dict[str, Any],
                      chunk: Callable[[dict[str, Any]], None], *,
                      cancelled: threading.Event, timeout_s: float = 120,
                      maximum_bytes: int = 16 * 1024 * 1024,
                      maximum_line_bytes: int = 1024 * 1024,
                      maximum_event_bytes: int = 2 * 1024 * 1024,
                      maximum_events: int = 1024) -> None:
    """POST once; consume the native SSE stream completely before accepting it.

    Native JSON events must end in a blank SSE line.  A terminal ``stop: true``
    object is required, and any subsequent data event is contradictory evidence.
    This is the native endpoint, not the OpenAI chat ``[DONE]`` protocol.
    Every successfully decoded object is delivered before its semantic refusal,
    preserving an observed error/reuse contradiction in the caller's raw record.
    """
    url = urlsplit(base_url)
    try:
        port = url.port
    except ValueError as exc:
        raise FitRefused("invalid loopback port") from exc
    if (url.scheme != "http" or url.hostname != "127.0.0.1" or url.path not in ("", "/")
            or url.username or url.password or url.query or url.fragment
            or not isinstance(key, str) or not key or len(key) > 4096 or not key.isascii()
            or any(ord(character) < 33 or ord(character) > 126 for character in key)
            or type(timeout_s) not in (int, float) or not math.isfinite(timeout_s) or timeout_s <= 0
            or any(type(v) is not int or v <= 0 for v in
                   (maximum_bytes, maximum_line_bytes, maximum_event_bytes, maximum_events))
            or maximum_bytes > 32 * 1024 * 1024 or maximum_line_bytes > 2 * 1024 * 1024
            or maximum_event_bytes > 4 * 1024 * 1024 or maximum_events > 2048):
        raise FitRefused("bounded authenticated literal-loopback completion endpoint required")
    deadline = time.monotonic() + timeout_s
    encoded = _request_bytes(body, 2 * 1024 * 1024)
    sockets: list[socket.socket] = []
    socket_lock = threading.Lock()
    # This separate event revokes chunk publication even if an IO double ignores
    # shutdown.  A shared cancellation also prevents a later engine admission.
    finished = threading.Event()
    result: Queue[tuple[str, Any]] = Queue(maxsize=2)

    def remaining() -> float:
        if cancelled.is_set() or finished.is_set():
            raise AnswerCancelled("native request cancelled")
        left = deadline - time.monotonic()
        if left <= 0:
            raise AnswerTimeout("native completion total deadline expired")
        return left

    def remember(sock: socket.socket) -> None:
        with socket_lock:
            sockets.append(sock)
        remaining()

    def shut_down() -> None:
        with socket_lock:
            captured = list(sockets)
        for sock in captured:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            try:
                sock.close()
            except OSError:
                pass

    def worker() -> None:
        connection: http.client.HTTPConnection | None = None
        response: http.client.HTTPResponse | None = None
        failure: BaseException | None = None
        def offer(kind: str,value: Any) -> None:
            while True:
                left=remaining()
                try:
                    result.put((kind,value),timeout=min(left,.02));return
                except Full:continue
        try:
            remaining()
            connection = _Connection(url.hostname or "", port, remaining=remaining, remember=remember)
            connection.request("POST", "/completion", body=encoded, headers={
                "Authorization": "Bearer " + key, "Content-Type": "application/json",
                "Accept": "text/event-stream", "Connection": "close"})
            remaining()
            response = connection.getresponse()
            remaining()
            content_type = response.getheader("Content-Type", "").split(";", 1)[0].strip().lower()
            if (response.status != 200 or response.getheader("Location") is not None
                    or content_type != "text/event-stream"
                    or response.getheader("Content-Encoding", "identity").lower() != "identity"):
                raise FitRefused("native SSE status, redirect, encoding or content type refused")
            # A single response may not ambiguously describe two body framings.
            lengths = response.headers.get_all("Content-Length", [])
            transfers = response.headers.get_all("Transfer-Encoding", [])
            if (len(lengths) > 1 or len(transfers) > 1 or (lengths and transfers)
                    or (transfers and transfers[0].strip().lower() != "chunked")):
                raise FitRefused("ambiguous native HTTP body framing")
            if lengths:
                if not lengths[0].isascii() or not lengths[0].isdigit() or int(lengths[0]) > maximum_bytes:
                    raise FitRefused("native HTTP response size exceeds limit")
            expected_length = int(lengths[0]) if lengths else None
            pending = bytearray()
            data: list[str] = []
            size = count = event_size = 0
            terminal = False

            def line(raw: bytes) -> None:
                nonlocal count, terminal, event_size
                remaining()
                if raw.endswith(b"\r"):
                    raw = raw[:-1]
                if len(raw) > maximum_line_bytes:
                    raise FitRefused("native SSE line exceeds byte limit")
                value = raw.decode("utf-8", "strict")
                if not value:
                    if not data:
                        return
                    value = "\n".join(data)
                    data.clear()
                    event_size = 0
                    if terminal:
                        raise FitRefused("native data follows terminal completion")
                    payload = strict_json(value)
                    if not isinstance(payload, dict):
                        raise FitRefused("native SSE event is not an object")
                    count += 1
                    if count > maximum_events:
                        raise FitRefused("native SSE event count exceeds limit")
                    remaining()
                    # Read workers have no authority to publish core evidence.
                    offer('chunk',value)
                    if "error" in payload:
                        raise FitRefused("native stream reported an error")
                    if "stop" in payload and type(payload["stop"]) is not bool:
                        raise FitRefused("native stop is not boolean")
                    timings = payload.get("timings")
                    if isinstance(timings, dict) and "cache_n" in timings:
                        if type(timings["cache_n"]) is not int or timings["cache_n"] != 0:
                            raise FitRefused("requested fresh native work reports reuse")
                    terminal = payload.get("stop") is True
                    return
                if value.startswith(":"):
                    return
                if not value.startswith("data:"):
                    raise FitRefused("unsupported native SSE field")
                value = value[5:]
                if value.startswith(" "):
                    value = value[1:]
                event_size += len(raw) + 1
                if event_size > maximum_event_bytes:
                    raise FitRefused("native SSE event exceeds byte limit")
                data.append(value)

            while True:
                left = remaining()
                with socket_lock:
                    current = list(sockets)
                for sock in current:
                    sock.settimeout(left)
                block = response.read1(min(65536, maximum_bytes + 1 - size))
                remaining()
                if not block:
                    break
                size += len(block)
                if size > maximum_bytes:
                    raise FitRefused("native SSE response exceeds byte limit")
                pending.extend(block)
                while True:
                    index = pending.find(b"\n")
                    if index < 0:
                        break
                    raw = bytes(pending[:index])
                    del pending[:index + 1]
                    line(raw)
                if len(pending) > maximum_line_bytes:
                    raise FitRefused("native SSE line exceeds byte limit")
            remaining()
            if expected_length is not None and size != expected_length:
                raise FitRefused("native HTTP body ended before its declared length")
            if pending or data:
                raise FitRefused("native SSE ended inside an event")
            if not terminal:
                raise FitRefused("native SSE ended before terminal completion")
        except BaseException as exc:
            failure=exc
        finally:
            for resource in (response,connection):
                if resource is not None:
                    try:resource.close()
                    except BaseException as exc:
                        if failure is None or isinstance(exc,(KeyboardInterrupt,SystemExit)):
                            if failure is not None:exc.__context__=failure
                            failure=exc
        try:offer('error' if failure is not None else 'done',failure)
        except BaseException:pass  # caller already revoked an abandoned reader

    if cancelled.is_set():
        raise AnswerCancelled("native admission closed")
    threading.Thread(target=worker, daemon=True, name="lookup-answer-stream").start()
    try:
        while True:
            left = remaining()
            try:
                kind, value = result.get(timeout=min(left, .02))
            except Empty:
                continue
            remaining()
            if kind=='error':
                raise value
            if kind=='done':return
            if kind!='chunk':raise FitRefused('unknown native delivery event')
            payload=strict_json(value)
            remaining()
            chunk(payload)
            remaining()
    except BaseException:
        cancelled.set()
        raise
    finally:
        finished.set()
        shut_down()
