"""Run regressions with external networking disabled, including during imports."""
from pathlib import Path
import socket
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
_connect = socket.socket.connect


def _blocked(*args, **kwargs):
    raise AssertionError("Offline tests cannot use the network")


def _internal_socketpair(self, address, *args, **kwargs):
    # Windows asyncio implements its internal wakeup socketpair over loopback.
    caller = sys._getframe(1)
    if (isinstance(address, tuple) and address[0] in {"127.0.0.1", "::1"}
            and caller.f_code.co_name == "_fallback_socketpair"
            and Path(caller.f_code.co_filename).name == "socket.py"):
        return _connect(self, address, *args, **kwargs)
    return _blocked()


if __name__ == "__main__":
    with patch.object(socket.socket, "connect", _internal_socketpair), \
            patch.object(socket.socket, "connect_ex", _blocked), \
            patch.object(socket, "getaddrinfo", _blocked):
        suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"))
        result = unittest.TextTestRunner(verbosity=2).run(suite)
    sys.exit(not result.wasSuccessful())
