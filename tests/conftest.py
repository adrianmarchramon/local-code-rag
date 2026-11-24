"""Suite-wide fixtures.

Every test runs inside a :class:`NetworkGuard`: an attempt to reach a non-loopback host
fails the test at teardown, even when the code under test catches the exception.
"""

from collections.abc import Iterator

import pytest

from src.network_guard import NetworkGuard


@pytest.fixture(autouse=True)
def network_guard() -> Iterator[NetworkGuard]:
    """Run the test under a guard and assert that nothing tried to leave the machine."""
    with NetworkGuard() as guard:
        yield guard
    assert guard.violations == (), f"external network access attempted: {guard.violations}"
