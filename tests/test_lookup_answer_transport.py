"""Native answer transport: no engine, model, process creation or hardware use."""
from __future__ import annotations

from email.message import Message
import json
from queue import Queue
import socket
import threading
import time
from typing import Any

import pytest

from inferenceos.tune import lookup_answer_transport as transport
from inferenceos.tune.lookup_answers import request_body
from inferenceos.tune.lookup_corpus import build_corpus
from inferenceos.tune.lookup_fit import FitRefused


def body() -> dict[str, Any]:
    return request_body(build_corpus().cases[0])


@pytest.mark.parametrize('prior_failure',(False,True))
def test_http_close_error_does_not_mask_primary_failure_or_become_timeout(monkeypatch: pytest.MonkeyPatch,prior_failure: bool) -> None:
    response=ResponseDouble(sse({'stop':True}));primary=OSError('original read');cleanup=OSError('connection close')
    if prior_failure:
        def read(size: int) -> bytes:raise primary
        monkeypatch.setattr(response,'read1',read)
    def close() -> None:raise cleanup
    monkeypatch.setattr(response,'close',close)
    install(monkeypatch,response)
    with pytest.raises(OSError) as caught:
        transport.stream_completion('http://127.0.0.1:17001','key',body(),lambda c:None,cancelled=threading.Event())
    assert caught.value is (primary if prior_failure else cleanup)


def test_stream_callbacks_run_only_on_calling_thread(monkeypatch: pytest.MonkeyPatch) -> None:
    install(monkeypatch,ResponseDouble(sse({'content':'a','stop':False},{'stop':True})))
    caller=threading.get_ident();threads=[]
    transport.stream_completion('http://127.0.0.1:17001','key',body(),lambda data:threads.append(threading.get_ident()),
                                cancelled=threading.Event())
    assert threads==[caller,caller]


def test_queued_chunk_cannot_publish_after_cancellation(monkeypatch: pytest.MonkeyPatch) -> None:
    cancelled=threading.Event();seen: list[Any]=[]
    class CancelOnDelivery(Queue[Any]):
        def get(self,*args: Any,**kwargs: Any) -> Any:
            item=super().get(*args,**kwargs)
            if item[0]=='chunk':cancelled.set()
            return item
    monkeypatch.setattr(transport,'Queue',CancelOnDelivery)
    install(monkeypatch,ResponseDouble(sse({'content':'late','stop':True})))
    with pytest.raises(transport.AnswerCancelled):
        transport.stream_completion('http://127.0.0.1:17001','key',body(),seen.append,cancelled=cancelled)
    assert seen==[]


def test_blocked_callback_cannot_outlive_its_caller_and_mutate_after_return(monkeypatch: pytest.MonkeyPatch) -> None:
    install(monkeypatch,ResponseDouble(sse({'content':'a','stop':False},{'stop':True})))
    entered=threading.Event();proceed=threading.Event();finished=threading.Event();cancelled=threading.Event();seen=[];errors=[]
    def receive(data: Any) -> None:
        entered.set();assert proceed.wait(2);seen.append(data)
    def call() -> None:
        try:transport.stream_completion('http://127.0.0.1:17001','key',body(),receive,cancelled=cancelled)
        except BaseException as exc:errors.append(exc)
        finally:finished.set()
    caller=threading.Thread(target=call);caller.start()
    assert entered.wait(1);cancelled.set()
    assert not finished.wait(.02)  # caller owns its callback; there is no orphan publisher
    proceed.set();caller.join(2)
    assert finished.is_set() and len(errors)==1 and isinstance(errors[0],transport.AnswerCancelled)
    count=len(seen);time.sleep(.03)
    assert len(seen)==count==1


class SocketDouble:
    def __init__(self) -> None:
        self.closed = False
        self.shutdown_called = False

    def settimeout(self, timeout: float) -> None:
        assert timeout > 0

    def shutdown(self, how: int) -> None:
        assert how == socket.SHUT_RDWR
        self.shutdown_called = True

    def close(self) -> None:
        self.closed = True


class ResponseDouble:
    def __init__(self, raw: bytes, *, headers: list[tuple[str, str]] | None = None,
                 status: int = 200, read_size: int = 65536) -> None:
        self.raw, self.status, self.read_size = raw, status, read_size
        self.headers = Message()
        for key, value in headers or [("Content-Type", "text/event-stream")]:
            self.headers[key] = value
        self.closed = False

    def getheader(self, name: str, default: str | None = None) -> Any:
        return self.headers.get(name, default)

    def read1(self, maximum: int) -> bytes:
        taken = self.raw[:min(maximum, self.read_size)]
        self.raw = self.raw[len(taken):]
        return taken

    def close(self) -> None:
        self.closed = True


class ConnectionDouble:
    def __init__(self, response: ResponseDouble, remaining: Any, remember: Any) -> None:
        self.response = response
        self.remaining = remaining
        self.sock = SocketDouble()
        remember(self.sock)
        self.requests: list[Any] = []
        self.closed = False

    def request(self, method: str, path: str, **kwargs: Any) -> None:
        self.remaining()
        self.requests.append((method, path, kwargs))

    def getresponse(self) -> ResponseDouble:
        return self.response

    def close(self) -> None:
        self.closed = True


def install(monkeypatch: pytest.MonkeyPatch, response: ResponseDouble) -> list[ConnectionDouble]:
    created: list[ConnectionDouble] = []

    def factory(host: str, port: int | None, *, remaining: Any, remember: Any) -> ConnectionDouble:
        assert host == "127.0.0.1" and port == 17001
        connection = ConnectionDouble(response, remaining, remember)
        created.append(connection)
        return connection

    monkeypatch.setattr(transport, "_Connection", factory)
    return created


def sse(*values: dict[str, Any]) -> bytes:
    return b"".join(b"data: " + json.dumps(value, ensure_ascii=False).encode() + b"\n\n" for value in values)


def run(monkeypatch: pytest.MonkeyPatch, response: ResponseDouble, **kwargs: Any) -> tuple[list[Any], list[ConnectionDouble]]:
    created = install(monkeypatch, response)
    seen: list[Any] = []
    transport.stream_completion("http://127.0.0.1:17001", "key", body(), seen.append,
                                cancelled=threading.Event(), **kwargs)
    return seen, created


@pytest.mark.parametrize("piece", [1, 2, 7, 31, 65536])
def test_incremental_utf8_raw_chunks_and_terminal(monkeypatch: pytest.MonkeyPatch, piece: int) -> None:
    expected = [{"content": "a°茶", "tokens": [1, 2], "stop": False},
                {"content": "", "tokens": [], "stop": True, "timings": {"cache_n": 0}}]
    response = ResponseDouble(b": ping\r\n\r\n" + sse(*expected).replace(b"\n", b"\r\n"), read_size=piece)
    seen, created = run(monkeypatch, response)
    assert seen == expected
    assert len(created) == 1 and created[0].closed and response.closed
    sent = created[0].requests
    assert len(sent) == 1 and sent[0][0:2] == ("POST", "/completion")
    assert json.loads(sent[0][2]["body"]) == body()
    assert sent[0][2]["headers"]["Authorization"] == "Bearer key"


@pytest.mark.parametrize("raw", [
    b"data: {\"stop\":true,\"stop\":true}\n\n",
    b"data: {\"stop\":true,\"STOP\":true}\n\n",
    b"data: {\"stop\":true,\"value\":0.1234567890123456789}\n\n",
    b"data: {\"stop\":true,\"value\":NaN}\n\n",
    b"data: {\"stop\":true,\"value\":Infinity}\n\n",
    b"data: {\"stop\":true,\"value\":1e9999}\n\n",
    b"data: {\"stop\":true,\"content\":\"\xff\"}\n\n",
    b"data: [1,2]\n\n", b"data: [DONE]\n\n",
    b"event: done\ndata: {\"stop\":true}\n\n",
    b"retry: 10\n\n", b"data: broken\n\n",
    b"data: {\"stop\":true}\n", b"data: {\"stop\":true}",
    b"data: {\"stop\":false}\n\n", b"", b": ping\n\n",
    b"data: {\"stop\":1}\n\n",
    b"data: {\"error\":{\"message\":\"bad\"},\"stop\":true}\n\n",
    b"data: {\"stop\":true}\n\ndata: {\"content\":\"extra\"}\n\n",
    b"data: {\"stop\":true}\n\ndata: [DONE]\n\n",
])
def test_invalid_stream_is_not_silently_normalized(monkeypatch: pytest.MonkeyPatch, raw: bytes) -> None:
    response = ResponseDouble(raw)
    created = install(monkeypatch, response)
    cancelled = threading.Event()
    with pytest.raises((ValueError, UnicodeError, FitRefused)):
        transport.stream_completion("http://127.0.0.1:17001", "key", body(), lambda chunk: None, cancelled=cancelled)
    assert cancelled.is_set() and len(created) == 1


@pytest.mark.parametrize("headers,status", [
    ([("Content-Type", "application/json")], 200),
    ([("Content-Type", "text/event-stream"), ("Location", "/other")], 200),
    ([("Content-Type", "text/event-stream")], 302),
    ([("Content-Type", "text/event-stream")], 500),
    ([("Content-Type", "text/event-stream"), ("Content-Encoding", "gzip")], 200),
    ([("Content-Type", "text/event-stream"), ("Content-Length", "10"), ("Content-Length", "10")], 200),
    ([("Content-Type", "text/event-stream"), ("Content-Length", "10"), ("Transfer-Encoding", "chunked")], 200),
    ([("Content-Type", "text/event-stream"), ("Content-Length", "9999999999")], 200),
    ([("Content-Type", "text/event-stream"), ("Content-Length", "-10")], 200),
    ([("Content-Type", "text/event-stream"), ("Transfer-Encoding", "gzip")], 200),
])
def test_http_contradictions_refused_before_any_chunk(monkeypatch: pytest.MonkeyPatch,
                                                     headers: list[tuple[str, str]], status: int) -> None:
    install(monkeypatch, ResponseDouble(sse({"stop": True}), headers=headers, status=status))
    seen: list[Any] = []
    with pytest.raises(FitRefused):
        transport.stream_completion("http://127.0.0.1:17001", "key", body(), seen.append, cancelled=threading.Event())
    assert seen == []


def test_truncated_http_body_preserves_reading_but_does_not_complete(monkeypatch: pytest.MonkeyPatch) -> None:
    install(monkeypatch, ResponseDouble(sse({"stop": True}), headers=[
        ("Content-Type", "text/event-stream"), ("Content-Length", "1000")]))
    seen: list[Any] = []
    with pytest.raises(FitRefused, match="declared length"):
        transport.stream_completion("http://127.0.0.1:17001", "key", body(), seen.append, cancelled=threading.Event())
    assert seen == [{"stop": True}]


@pytest.mark.parametrize("change", [
    {"cache_prompt": True}, {"ignore_eos": True}, {"n_predict": 513}, {"n_predict": True},
    {"seed": 1}, {"seed": True}, {"temperature": True}, {"temperature": 1}, {"stream": False},
    {"return_tokens": False}, {"stop": ["end"]}, {"unknown": 1}, {"prompt": [1]},
])
def test_only_registered_fresh_request_can_dispatch(monkeypatch: pytest.MonkeyPatch, change: dict[str, Any]) -> None:
    created = install(monkeypatch, ResponseDouble(sse({"stop": True})))
    with pytest.raises(FitRefused):
        transport.stream_completion("http://127.0.0.1:17001", "key", dict(body(), **change), lambda chunk: None,
                                    cancelled=threading.Event())
    assert created == []


@pytest.mark.parametrize("url", ["http://localhost:17001", "https://127.0.0.1:17001", "http://127.0.0.2:17001",
    "http://127.0.0.1:17001/other", "http://user@127.0.0.1:17001", "http://127.0.0.1:17001/?query=1",
    "http://127.0.0.1:17001/#fragment", "http://127.0.0.1:99999"])
def test_only_literal_loopback_endpoint_can_dispatch(monkeypatch: pytest.MonkeyPatch, url: str) -> None:
    created = install(monkeypatch, ResponseDouble(sse({"stop": True})))
    with pytest.raises(FitRefused):
        transport.stream_completion(url, "key", body(), lambda chunk: None, cancelled=threading.Event())
    assert created == []


@pytest.mark.parametrize("key", ["", "a\rb", "a\nb", "a b", "a\tb", "a\x00b", "🔑", "a" * 4097])
def test_invalid_credential_cannot_dispatch(monkeypatch: pytest.MonkeyPatch, key: str) -> None:
    created = install(monkeypatch, ResponseDouble(sse({"stop": True})))
    with pytest.raises(FitRefused):
        transport.stream_completion("http://127.0.0.1:17001", key, body(), lambda chunk: None,
                                    cancelled=threading.Event())
    assert created == []


@pytest.mark.parametrize("limit,raw", [
    ({"maximum_line_bytes": 10}, sse({"stop": True})),
    ({"maximum_event_bytes": 200}, sse({"content": "a" * 300, "stop": True})),
    ({"maximum_bytes": 20}, sse({"stop": True})),
    ({"maximum_events": 1}, sse({"stop": False}, {"stop": True})),
])
def test_byte_line_event_and_count_caps(monkeypatch: pytest.MonkeyPatch, limit: dict[str, Any], raw: bytes) -> None:
    install(monkeypatch, ResponseDouble(raw))
    with pytest.raises(FitRefused):
        transport.stream_completion("http://127.0.0.1:17001", "key", body(), lambda chunk: None,
                                    cancelled=threading.Event(), **limit)


def test_reuse_is_preserved_then_immediately_stops(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[Any] = []
    install(monkeypatch, ResponseDouble(sse({"stop": False, "timings": {"cache_n": 1}}, {"stop": True})))
    cancelled = threading.Event()
    with pytest.raises(FitRefused, match="reports reuse"):
        transport.stream_completion("http://127.0.0.1:17001", "key", body(), seen.append, cancelled=cancelled)
    assert seen == [{"stop": False, "timings": {"cache_n": 1}}] and cancelled.is_set()


def test_callback_exception_object_survives_without_later_chunks(monkeypatch: pytest.MonkeyPatch) -> None:
    install(monkeypatch, ResponseDouble(sse({"stop": False}, {"stop": True})))
    original = KeyboardInterrupt("stop")
    seen: list[Any] = []

    def receive(chunk: Any) -> None:
        seen.append(chunk)
        raise original

    with pytest.raises(KeyboardInterrupt) as caught:
        transport.stream_completion("http://127.0.0.1:17001", "key", body(), receive, cancelled=threading.Event())
    assert caught.value is original and len(seen) == 1


@pytest.mark.parametrize("where", ["headers", "body"])
def test_total_deadline_rejects_late_io_double_without_late_publication(monkeypatch: pytest.MonkeyPatch, where: str) -> None:
    proceeded = threading.Event()
    entered = threading.Event()
    response = ResponseDouble(sse({"stop": True}))
    created = install(monkeypatch, response)
    original_read = response.read1

    def paused_read(maximum: int) -> bytes:
        entered.set()
        proceeded.wait(2)
        return original_read(maximum)

    if where == "body":
        response.read1 = paused_read  # type: ignore[method-assign]
    else:
        factory: Any = transport._Connection

        def paused_connection(*args: Any, **kwargs: Any) -> Any:
            connection = factory(*args, **kwargs)
            original_response = connection.getresponse

            def paused_response() -> Any:
                entered.set()
                proceeded.wait(2)
                return original_response()

            connection.getresponse = paused_response
            return connection

        monkeypatch.setattr(transport, "_Connection", paused_connection)
    seen: list[Any] = []
    cancelled = threading.Event()
    began = time.monotonic()
    try:
        with pytest.raises(transport.AnswerTimeout):
            transport.stream_completion("http://127.0.0.1:17001", "key", body(), seen.append,
                                        cancelled=cancelled, timeout_s=.08)
        assert entered.is_set() and time.monotonic() - began < .8
        assert cancelled.is_set() and created[0].sock.shutdown_called
    finally:
        proceeded.set()
    # Give the now-unblocked daemon an explicit completion handshake.
    for _ in range(100):
        if response.closed:
            break
        time.sleep(.002)
    assert response.closed and seen == []


def test_cancelled_request_returns_while_ignored_read_is_still_blocked(monkeypatch: pytest.MonkeyPatch) -> None:
    response = ResponseDouble(sse({"stop": True}))
    install(monkeypatch, response)
    entered, proceed, cancelled = threading.Event(), threading.Event(), threading.Event()

    def blocked(maximum: int) -> bytes:
        entered.set()
        proceed.wait(2)
        return b""

    response.read1 = blocked  # type: ignore[method-assign]
    def cancel_when_entered() -> None:
        entered.wait(1)
        cancelled.set()

    timer = threading.Timer(.03, cancel_when_entered)
    timer.start()
    try:
        with pytest.raises(transport.AnswerCancelled):
            transport.stream_completion("http://127.0.0.1:17001", "key", body(), lambda chunk: None,
                                        cancelled=cancelled, timeout_s=1)
        assert entered.is_set()
    finally:
        proceed.set()
        timer.join()


@pytest.mark.parametrize("mode", ["headers", "body", "complete"])
def test_real_loopback_total_deadline_including_dribbling(mode: str) -> None:
    # A local socket fixture tests actual HTTPConnection buffering/shutdown. It
    # accepts exactly one request and neither invokes nor constructs a backend.
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    listener.settimeout(2)
    port = listener.getsockname()[1]
    accepted = threading.Event()
    done = threading.Event()
    sent_body = sse({"content": "x", "stop": False}, {"stop": True})

    def serve() -> None:
        connection: socket.socket | None = None
        try:
            connection, _ = listener.accept()
            connection.settimeout(1)
            request = b""
            while b"\r\n\r\n" not in request:
                block = connection.recv(65536)
                if not block:
                    return
                request += block
            request_headers, request_body_bytes = request.split(b"\r\n\r\n", 1)
            content_length = next(int(line.split(b":", 1)[1].strip()) for line in request_headers.split(b"\r\n")
                                  if line.lower().startswith(b"content-length:"))
            while len(request_body_bytes) < content_length:
                block = connection.recv(65536)
                if not block:
                    return
                request_body_bytes += block
            accepted.set()
            headers = b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n"
            if mode == "headers":
                for value in headers:
                    connection.sendall(bytes([value]))
                    if done.wait(.015):
                        return
            else:
                connection.sendall(headers)
            if mode == "body":
                for value in sent_body:
                    connection.sendall(bytes([value]))
                    if done.wait(.015):
                        return
            else:
                connection.sendall(sent_body)
        except (OSError, TimeoutError):
            pass
        finally:
            if connection is not None:
                connection.close()

    worker = threading.Thread(target=serve, daemon=True)
    worker.start()
    seen: list[Any] = []
    began = time.monotonic()
    cancelled = threading.Event()
    try:
        if mode == "complete":
            transport.stream_completion(f"http://127.0.0.1:{port}", "key", body(), seen.append,
                                        cancelled=cancelled, timeout_s=2)
            assert len(seen) == 2 and not cancelled.is_set()
        else:
            with pytest.raises((transport.AnswerTimeout, TimeoutError)):
                transport.stream_completion(f"http://127.0.0.1:{port}", "key", body(), seen.append,
                                            cancelled=cancelled, timeout_s=.12)
            assert time.monotonic() - began < 1 and cancelled.is_set()
        assert accepted.is_set()
    finally:
        done.set()
        listener.close()
        worker.join(2)
    assert not worker.is_alive()
