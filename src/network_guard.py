"""In-process network guard: block every non-loopback connection and name lookup.

Usage::

    with NetworkGuard() as guard:
        ...  # code under audit
    assert guard.violations == ()

Scope. The guard wraps Python-level ``socket`` entry points in this process
(``socket.connect``/``connect_ex``/``sendto``/``sendmsg``, ``getaddrinfo``,
``gethostbyname``/``gethostbyname_ex``), which is what ``httpx`` (used by the ``ollama``
client) and the standard library go through. It allows loopback (``localhost``,
``127.0.0.0/8``, ``::1``), Unix-domain sockets, and an explicit allowlist (empty by
default). It does not see subprocesses or C extensions that bypass Python's ``socket``
module, so it complements, and does not replace, the listening-address audit and the
manual offline test.

Every blocked attempt is recorded in ``violations`` (operation, destination, UTC
timestamp) before :class:`ExternalNetworkAccessError` is raised, so a caller that catches
the exception cannot hide the attempt.

Nesting. Each guard saves the socket functions installed when it is entered and restores
exactly those on exit, even when the block raises. The innermost guard sees a call first:
it records and raises if the call is not allowed, otherwise it hands the call to whatever
was installed before it (the enclosing guard, if any), which applies its own policy.
Guards must be used as context managers (LIFO); entering and exiting is not thread-safe.
"""

from __future__ import annotations

import ipaddress
import logging
import socket
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from types import TracebackType
from typing import Any, Final, NoReturn, Self

from src.errors import ExternalNetworkAccessError

logger = logging.getLogger(__name__)

_LOOPBACK_NAMES: Final = frozenset({"localhost"})
_MISSING: Final = object()


def is_loopback_host(host: str) -> bool:
    """Return True for ``localhost`` and loopback IP literals. Never performs DNS."""
    candidate = host.strip().lower()
    if candidate.startswith("[") and candidate.endswith("]"):
        candidate = candidate[1:-1]
    if candidate in _LOOPBACK_NAMES:
        return True
    try:
        address = ipaddress.ip_address(candidate.split("%", 1)[0])  # drop an IPv6 zone id
    except ValueError:
        return False
    if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped is not None:
        return address.ipv4_mapped.is_loopback
    return address.is_loopback


@dataclass(frozen=True, slots=True)
class NetworkViolation:
    """One blocked network attempt."""

    timestamp: datetime
    operation: str
    destination: str


def _as_text(host: object) -> str:
    if isinstance(host, bytes | bytearray):
        return bytes(host).decode("ascii", errors="replace")
    return str(host)


def _format_destination(host: object, port: object) -> str:
    text = _as_text(host)
    if port is None:
        return text
    return f"[{text}]:{port}" if ":" in text else f"{text}:{port}"


def _family_name(family: int) -> str:
    try:
        return socket.AddressFamily(family).name
    except ValueError:
        return str(family)


class NetworkGuard:
    """Context manager that blocks and records non-loopback network access."""

    def __init__(self, allowed_hosts: Iterable[str] = ()) -> None:
        self._allowed_hosts = frozenset(h.strip().lower() for h in allowed_hosts if h.strip())
        self._resolved_allowed: set[str] = set()
        self._violations: list[NetworkViolation] = []
        self._saved: list[tuple[object, str, object]] | None = None

    @property
    def violations(self) -> tuple[NetworkViolation, ...]:
        """Every attempt blocked by this guard, in order."""
        return tuple(self._violations)

    @property
    def active(self) -> bool:
        """Whether the guard is currently installed."""
        return self._saved is not None

    def is_allowed_host(self, host: object) -> bool:
        """Whether this guard lets traffic to ``host`` through (no DNS involved)."""
        if host is None:  # getaddrinfo(None, ...) refers to the local host
            return True
        normalized = _as_text(host).strip().lower()
        return (
            is_loopback_host(normalized)
            or normalized in self._allowed_hosts
            or normalized in self._resolved_allowed
        )

    def __enter__(self) -> Self:
        if self._saved is not None:
            raise RuntimeError("NetworkGuard is already active; use a new instance to nest.")
        patches = self._build_patches()
        self._saved = [
            (target, name, vars(target).get(name, _MISSING)) for target, name, _ in patches
        ]
        for target, name, wrapper in patches:
            setattr(target, name, wrapper)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        saved, self._saved = self._saved, None
        for target, name, original in reversed(saved or []):
            if original is _MISSING:
                delattr(target, name)
            else:
                setattr(target, name, original)

    def _block(self, operation: str, destination: str) -> NoReturn:
        self._violations.append(
            NetworkViolation(
                timestamp=datetime.now(UTC), operation=operation, destination=destination
            )
        )
        logger.warning(
            "network_guard.blocked",
            extra={
                "component": "network_guard",
                "operation": operation,
                "destination": destination,
            },
        )
        raise ExternalNetworkAccessError(operation=operation, destination=destination)

    def _check_socket_address(self, sock: socket.socket, address: object, operation: str) -> None:
        family = sock.family
        if hasattr(socket, "AF_UNIX") and family == socket.AF_UNIX:
            return
        is_ip_family = family in (socket.AF_INET, socket.AF_INET6)
        if is_ip_family and isinstance(address, tuple) and len(address) >= 2:
            host, port = address[0], address[1]
            if self.is_allowed_host(host):
                return
            self._block(operation, _format_destination(host, port))
        self._block(operation, f"{_family_name(family)}:{address!r}")

    def _check_lookup(self, host: object, port: object = None) -> None:
        if not self.is_allowed_host(host):
            self._block("resolve", _format_destination(host, port))

    def _build_patches(self) -> list[tuple[object, str, Callable[..., Any]]]:
        """Wrap whatever is installed now, so an enclosing guard keeps applying its policy."""
        connect: Callable[..., Any] = socket.socket.connect
        connect_ex: Callable[..., Any] = socket.socket.connect_ex
        sendto: Callable[..., Any] = socket.socket.sendto
        sendmsg: Callable[..., Any] = socket.socket.sendmsg
        getaddrinfo: Callable[..., Any] = socket.getaddrinfo
        gethostbyname: Callable[..., Any] = socket.gethostbyname
        gethostbyname_ex: Callable[..., Any] = socket.gethostbyname_ex

        def guarded_connect(sock: socket.socket, address: Any) -> Any:
            self._check_socket_address(sock, address, "connect")
            return connect(sock, address)

        def guarded_connect_ex(sock: socket.socket, address: Any) -> Any:
            self._check_socket_address(sock, address, "connect")
            return connect_ex(sock, address)

        def guarded_sendto(sock: socket.socket, *args: Any) -> Any:
            if args:  # sendto(data, address) or sendto(data, flags, address)
                self._check_socket_address(sock, args[-1], "sendto")
            return sendto(sock, *args)

        def guarded_sendmsg(sock: socket.socket, *args: Any, **kwargs: Any) -> Any:
            address = kwargs.get("address", args[3] if len(args) > 3 else None)
            if address is not None:  # None means the already-checked connected peer
                self._check_socket_address(sock, address, "sendmsg")
            return sendmsg(sock, *args, **kwargs)

        def guarded_getaddrinfo(host: Any, port: Any, *args: Any, **kwargs: Any) -> Any:
            self._check_lookup(host, port)
            results = getaddrinfo(host, port, *args, **kwargs)
            if host is not None and not is_loopback_host(_as_text(host)):
                # An allowlisted name: the addresses it resolves to are allowed as well.
                self._resolved_allowed.update(str(info[4][0]).lower() for info in results)
            return results

        def guarded_gethostbyname(host: Any) -> Any:
            self._check_lookup(host)
            return gethostbyname(host)

        def guarded_gethostbyname_ex(host: Any) -> Any:
            self._check_lookup(host)
            return gethostbyname_ex(host)

        return [
            (socket.socket, "connect", guarded_connect),
            (socket.socket, "connect_ex", guarded_connect_ex),
            (socket.socket, "sendto", guarded_sendto),
            (socket.socket, "sendmsg", guarded_sendmsg),
            (socket, "getaddrinfo", guarded_getaddrinfo),
            (socket, "gethostbyname", guarded_gethostbyname),
            (socket, "gethostbyname_ex", guarded_gethostbyname_ex),
        ]
