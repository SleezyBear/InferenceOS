"""Evaluation-only deadline and JSON inspection primitives; no generation endpoint."""
from __future__ import annotations

import http.client
import json
import math
from pathlib import Path
from queue import Queue, Empty
import socket
import threading
import time
from typing import Any, Callable, TypeVar
from urllib.parse import urlsplit

from inferenceos.tune.lookup_fit import FitRefused, strict_json

T = TypeVar("T")


class InspectionTimeout(TimeoutError):
    pass


def bounded_read(call: Callable[[], T], *, timeout_s: float) -> T:
    """Do not accept a late observation or wait indefinitely on an OS inspection.

    A timed-out daemon may finish its read; it cannot publish or control an engine.
    Callers must stop the whole study. Start uses the backend's admission predicate,
    so timeout also revokes any late start before process creation.
    """
    if not math.isfinite(timeout_s) or timeout_s <= 0:
        raise ValueError("positive finite deadline required")
    deadline = time.monotonic()+timeout_s
    replies: Queue[tuple[bool, Any]] = Queue(maxsize=1)

    def work() -> None:
        try:
            replies.put((True, call()))
        except BaseException as exc:
            replies.put((False, exc))

    threading.Thread(target=work, daemon=True, name="lookup-fit-inspection").start()
    try:
        success, result = replies.get(timeout=max(0, deadline-time.monotonic()))
    except Empty as exc:
        raise InspectionTimeout("inspection deadline expired; no further load") from exc
    if time.monotonic() >= deadline:
        raise InspectionTimeout("late inspection reply refused")
    if not success:
        raise result
    return result  # type: ignore[no-any-return]


def bounded_file_bytes(path: Path, *, maximum_bytes: int = 16*1024*1024, timeout_s: float = 30) -> bytes:
    if type(maximum_bytes) is not int or not 1 <= maximum_bytes <= 64*1024*1024:
        raise ValueError('bounded JSON/file size required')
    def read() -> bytes:
        with path.open('rb') as stream:
            raw = stream.read(maximum_bytes+1)
        if len(raw)>maximum_bytes:
            raise FitRefused('input exceeds file size limit')
        return raw
    return bounded_read(read,timeout_s=timeout_s)


def inspect_json(base_url: str, key: str, path: str, *, body: dict[str, Any] | None = None,
                 timeout_s: float = 30.0, maximum_bytes: int = 8*1024*1024) -> Any:
    """Loopback /slots or /tokenize only, with total deadline and bounded body.

    Each read is one available chunk. Remaining time limits sockets and a timer shuts
    down a blocked socket as a backstop against dribbling headers/body. Late replies
    are rejected even when an IO double ignores its timeout.
    """
    url = urlsplit(base_url)
    if (url.scheme != "http" or url.hostname != "127.0.0.1" or url.path not in ("", "/")
            or url.query or url.fragment or url.username or url.password
            or path not in ("/slots", "/tokenize") or (path == "/slots" and body is not None)
            or not math.isfinite(timeout_s) or timeout_s <= 0 or not 1 <= maximum_bytes <= 16*1024*1024):
        raise FitRefused("bounded loopback inspection endpoints only")
    if path == "/tokenize" and (not isinstance(body, dict) or set(body) != {"content", "add_special"}
                                or not isinstance(body["content"], str) or type(body["add_special"]) is not bool):
        raise FitRefused("explicit tokenizer body required")
    deadline = time.monotonic()+timeout_s
    connection = http.client.HTTPConnection(url.hostname, url.port, timeout=timeout_s)

    def expire() -> None:
        sock = connection.sock
        if sock is not None:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass

    timer = threading.Timer(timeout_s, expire)
    timer.daemon = True
    timer.start()
    try:
        encoded = json.dumps(body).encode("utf-8") if body is not None else None
        connection.request("POST" if encoded is not None else "GET", path, body=encoded,
                           headers={"Authorization":"Bearer "+key, "Content-Type":"application/json", "Connection":"close"})
        response = connection.getresponse()
        if response.status != 200 or response.getheader("Location") is not None:
            raise FitRefused("inspection HTTP status/redirect refused")
        chunks, size = [], 0
        while True:
            remaining = deadline-time.monotonic()
            if remaining <= 0:
                raise InspectionTimeout("inspection total deadline expired")
            if connection.sock is not None:
                connection.sock.settimeout(remaining)
            chunk = response.read1(min(65536, maximum_bytes+1-size))
            if not chunk:
                break
            chunks.append(chunk)
            size += len(chunk)
            if size > maximum_bytes:
                raise FitRefused("inspection response exceeds size limit")
        if time.monotonic() >= deadline:
            raise InspectionTimeout("late inspection response refused")
        return strict_json(b"".join(chunks).decode("utf-8", "strict"))
    finally:
        timer.cancel()
        connection.close()
