"""Tests for src.network_guard.

Each test opens its own guard nested inside the suite-wide one (``network_guard``), so the
attempts blocked here are recorded by the inner guard and never pollute the outer one.
"""

import contextlib
import socket
from collections.abc import Callable
from datetime import UTC
from typing import Any

import pytest

from src.errors import ExternalNetworkAccessError
from src.network_guard import NetworkGuard, is_loopback_host

# TEST-NET-3 (RFC 5737): reserved for documentation, never routed on the internet.
EXTERNAL_IP = "203.0.113.10"
# Reserved by RFC 2606. It is never resolved here: the guard blocks the lookup first.
EXTERNAL_HOST = "example.com"

_PATCHED_SOCKET_METHODS = ("connect", "connect_ex", "sendto", "sendmsg")
_PATCHED_MODULE_FUNCTIONS = ("getaddrinfo", "gethostbyname", "gethostbyname_ex")


def _socket_state() -> tuple[object, ...]:
    methods = tuple(vars(socket.socket).get(name) for name in _PATCHED_SOCKET_METHODS)
    functions = tuple(vars(socket).get(name) for name in _PATCHED_MODULE_FUNCTIONS)
    return methods + functions


def test_blocks_connect_to_external_ip_and_records_it(network_guard: NetworkGuard) -> None:
    with (
        NetworkGuard() as guard,
        socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock,
        pytest.raises(ExternalNetworkAccessError, match=EXTERNAL_IP),
    ):
        sock.connect((EXTERNAL_IP, 80))

    assert [(v.operation, v.destination) for v in guard.violations] == [
        ("connect", f"{EXTERNAL_IP}:80")
    ]
    assert guard.violations[0].timestamp.tzinfo is UTC
    assert network_guard.violations == ()


def test_blocks_connect_ex_and_udp_sendto() -> None:
    with NetworkGuard() as guard:
        with (
            socket.socket(socket.AF_INET, socket.SOCK_STREAM) as tcp,
            pytest.raises(ExternalNetworkAccessError),
        ):
            tcp.connect_ex((EXTERNAL_IP, 443))
        with (
            socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as udp,
            pytest.raises(ExternalNetworkAccessError),
        ):
            udp.sendto(b"x", (EXTERNAL_IP, 53))

    assert [v.operation for v in guard.violations] == ["connect", "sendto"]


@pytest.mark.parametrize(
    "lookup",
    [
        lambda: socket.getaddrinfo(EXTERNAL_HOST, 443),
        lambda: socket.gethostbyname(EXTERNAL_HOST),
        lambda: socket.gethostbyname_ex(EXTERNAL_HOST),
        lambda: socket.create_connection((EXTERNAL_HOST, 443), timeout=1),
    ],
    ids=["getaddrinfo", "gethostbyname", "gethostbyname_ex", "create_connection"],
)
def test_blocks_hostname_resolution(lookup: Callable[[], object]) -> None:
    with NetworkGuard() as guard, pytest.raises(ExternalNetworkAccessError, match=EXTERNAL_HOST):
        lookup()

    assert [v.operation for v in guard.violations] == ["resolve"]


def test_blocked_lookup_never_reaches_the_resolver(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[Any, ...]] = []

    def spy(*args: Any, **kwargs: Any) -> list[Any]:
        calls.append(args)
        return []

    monkeypatch.setattr(socket, "getaddrinfo", spy)

    with NetworkGuard(), pytest.raises(ExternalNetworkAccessError):
        socket.getaddrinfo(EXTERNAL_HOST, 443)

    assert calls == []


def test_allows_loopback_tcp_through_localhost() -> None:
    with NetworkGuard() as guard, socket.create_server(("127.0.0.1", 0)) as server:
        port = server.getsockname()[1]
        with socket.create_connection(("localhost", port), timeout=5):
            pass

    assert guard.violations == ()


@pytest.mark.skipif(not hasattr(socket, "AF_UNIX"), reason="Unix-domain sockets unavailable")
def test_allows_unix_domain_sockets(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)  # a short relative path stays under the AF_UNIX length limit

    with NetworkGuard() as guard, socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as server:
        server.bind("guard.sock")
        server.listen(1)
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
            client.connect("guard.sock")

    assert guard.violations == ()


def test_violation_is_recorded_even_if_the_caller_swallows_the_error() -> None:
    with (
        NetworkGuard() as guard,
        socket.socket() as sock,
        contextlib.suppress(ExternalNetworkAccessError),
    ):
        sock.connect((EXTERNAL_IP, 80))

    assert len(guard.violations) == 1


def test_allowlisted_hostname_also_allows_its_resolved_addresses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resolved = [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("10.89.0.7", 11434))]
    monkeypatch.setattr(socket, "getaddrinfo", lambda *args, **kwargs: resolved)

    with NetworkGuard(allowed_hosts={"Ollama"}) as guard:
        assert socket.getaddrinfo("ollama", 11434) == resolved
        assert guard.is_allowed_host("10.89.0.7")
        assert not guard.is_allowed_host("10.89.0.8")

    assert guard.violations == ()


def test_inner_allowlist_cannot_bypass_an_enclosing_guard() -> None:
    with (
        NetworkGuard() as outer,
        NetworkGuard(allowed_hosts={EXTERNAL_IP}) as inner,
        socket.socket() as sock,
        pytest.raises(ExternalNetworkAccessError),
    ):
        sock.connect((EXTERNAL_IP, 80))

    assert inner.violations == ()
    assert [v.destination for v in outer.violations] == [f"{EXTERNAL_IP}:80"]


def test_patches_while_active_and_restores_even_after_an_exception() -> None:
    before = _socket_state()
    guard = NetworkGuard()
    seen_inside: list[tuple[bool, tuple[object, ...]]] = []

    def failing_block() -> None:
        with guard:
            seen_inside.append((guard.active, _socket_state()))
            raise ValueError("boom")

    with pytest.raises(ValueError, match="boom"):
        failing_block()

    [(was_active, state_inside)] = seen_inside
    assert was_active
    assert state_inside != before
    assert not guard.active
    assert _socket_state() == before


def test_reentering_an_active_guard_is_rejected() -> None:
    guard = NetworkGuard()

    with guard, pytest.raises(RuntimeError, match="already active"):
        guard.__enter__()


@pytest.mark.parametrize(
    ("host", "expected"),
    [
        ("localhost", True),
        ("LOCALHOST", True),
        ("127.0.0.1", True),
        ("127.255.255.254", True),
        ("::1", True),
        ("[::1]", True),
        ("::ffff:127.0.0.1", True),
        ("0.0.0.0", False),  # noqa: S104 - asserting it is NOT treated as loopback
        ("::", False),
        ("10.0.0.1", False),
        ("169.254.1.1", False),
        ("example.com", False),
        ("localhost.example.com", False),
        ("", False),
    ],
)
def test_is_loopback_host_uses_no_dns(host: str, expected: bool) -> None:
    assert is_loopback_host(host) is expected
