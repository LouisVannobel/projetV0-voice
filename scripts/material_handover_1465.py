"""Bounded native startup hold; release publication belongs exclusively to Root.

This gate verifies a release signal, never the physical joins behind that signal.
Only the successful delegation imports the existing native server.
"""

import argparse
import hashlib
import hmac
import json
import os
import re
import signal
import stat
import sys
import time
from datetime import UTC, datetime

IMAGE_DIGEST = "sha256:32515b282f458fee9b549a5cf891258c7a3b2f5b6f81d18b04a7aef751a841f6"
HASH = re.compile(r"[0-9a-f]{64}\Z")
KEYS = {"schema_version", "attempt_sha256", "candidate_sha256", "image_digest", "nonce"}
_MISSING = object()


class HoldRefused(Exception):
    """Unsafe control state or invalid input; details are deliberately private."""


class HoldExpired(HoldRefused):
    """Original expiry or maximum monotonic wait reached."""


class HoldCancelled(HoldRefused):
    """Termination during startup hold."""


def _directory_identity(value):
    return (value.st_dev, value.st_ino, value.st_mode, value.st_uid, value.st_gid)


def _file_identity(value):
    return (*_directory_identity(value), value.st_nlink, value.st_size,
            value.st_mtime_ns, value.st_ctime_ns)


class _ControlDirectory:
    """Hold every path component FD and verify every named-FD association.

    Walk from / using openat-style dir_fd operations; no component symlink is
    followed. Every ancestor and the mounted control directory must be root-owned
    and lack group/other write permissions.
    """

    def __init__(self, path):
        self.chain = []
        if not isinstance(path, str) or not path.startswith("/"):
            raise HoldRefused()
        parts = path.split("/")[1:]
        if not parts or any(part in {"", ".", ".."} for part in parts):
            raise HoldRefused()
        flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
        try:
            fd = os.open("/", flags)
            self.chain.append((None, "/", fd, _directory_identity(os.fstat(fd))))
            for part in parts:
                parent = fd
                fd = os.open(part, flags, dir_fd=parent)
                self.chain.append((parent, part, fd, _directory_identity(os.fstat(fd))))
            self.fd = fd
            self.verify()
        except BaseException:
            self.close()
            raise

    def verify(self):
        for parent, name, fd, identity in self.chain:
            current = os.fstat(fd)
            named = os.stat(name, dir_fd=parent, follow_symlinks=False)
            if _directory_identity(current) != identity or _directory_identity(named) != identity:
                raise HoldRefused()
            if (not stat.S_ISDIR(current.st_mode) or current.st_uid != 0
                    or current.st_mode & 0o022):
                raise HoldRefused()

    def close(self):
        for _, _, fd, _ in reversed(self.chain):
            os.close(fd)
        self.chain.clear()


def _pairs(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise HoldRefused()
        value[key] = item
    return value


def _nonfinite(_value):
    raise HoldRefused()


def _read_release(directory):
    directory.verify()
    try:
        fd = os.open("release.json", os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC,
                     dir_fd=directory.fd)
    except FileNotFoundError:
        directory.verify()
        return _MISSING
    try:
        before = os.fstat(fd)
        if (not stat.S_ISREG(before.st_mode) or before.st_uid != 0 or before.st_nlink != 1
                or before.st_mode & 0o022 or before.st_size > 4096):
            raise HoldRefused()
        identity = _file_identity(before)

        def verify_file():
            named = os.stat("release.json", dir_fd=directory.fd, follow_symlinks=False)
            if _file_identity(os.fstat(fd)) != identity or _file_identity(named) != identity:
                raise HoldRefused()
            directory.verify()

        def read_bounded():
            content = bytearray()
            while len(content) <= 4096:
                chunk = os.read(fd, 4097 - len(content))
                if not chunk:
                    return bytes(content)
                content.extend(chunk)
            raise HoldRefused()

        verify_file()
        content = read_bounded()
        verify_file()
        os.lseek(fd, 0, os.SEEK_SET)
        repeated = read_bounded()
        verify_file()
        if content != repeated or len(content) != before.st_size:
            raise HoldRefused()
        value = json.loads(content, object_pairs_hook=_pairs, parse_constant=_nonfinite)
        verify_file()
        return value
    finally:
        os.close(fd)


def _validate_release(value, attempt_sha256, candidate_sha256, nonce_sha256):
    if not isinstance(value, dict) or set(value) != KEYS:
        raise HoldRefused()
    if type(value["schema_version"]) is not int or value["schema_version"] != 1:
        raise HoldRefused()
    for key, expected in (("attempt_sha256", attempt_sha256),
                          ("candidate_sha256", candidate_sha256), ("image_digest", IMAGE_DIGEST)):
        if type(value[key]) is not str or not hmac.compare_digest(value[key], expected):
            raise HoldRefused()
    nonce = value["nonce"]
    if type(nonce) is not str or not HASH.fullmatch(nonce):
        raise HoldRefused()
    actual = hashlib.sha256(nonce.encode("ascii")).hexdigest()
    if not hmac.compare_digest(actual, nonce_sha256):
        raise HoldRefused()


def wait_for_release(release_directory, attempt_sha256, candidate_sha256, nonce_sha256,
                     candidate_expiry):
    """Wait at most 300 monotonic seconds; return only on one validated signal."""
    if os.name != "posix" or os.geteuid() != 10001:
        raise HoldRefused()
    if any(type(value) is not str or not HASH.fullmatch(value)
           for value in (attempt_sha256, candidate_sha256, nonce_sha256)):
        raise HoldRefused()
    if (not isinstance(candidate_expiry, datetime) or candidate_expiry.tzinfo is None
            or candidate_expiry.utcoffset() is None
            or candidate_expiry.utcoffset().total_seconds() != 0):
        raise HoldRefused()
    deadline = time.monotonic() + 300

    def check_expiry():
        if time.monotonic() >= deadline or datetime.now(UTC) >= candidate_expiry:
            raise HoldExpired()

    check_expiry()
    directory = _ControlDirectory(release_directory)
    try:
        print("material_startup_held", flush=True)
        while True:
            check_expiry()
            value = _read_release(directory)
            if value is not _MISSING:
                _validate_release(value, attempt_sha256, candidate_sha256, nonce_sha256)
                directory.verify()
                check_expiry()
                return
            remaining = (candidate_expiry - datetime.now(UTC)).total_seconds()
            time.sleep(max(0, min(0.1, deadline - time.monotonic(), remaining)))
    finally:
        directory.close()


class _Parser(argparse.ArgumentParser):
    def error(self, _message):
        raise HoldRefused()


def main():
    """The one public material operator, currently implementing native-hold only."""
    previous = {}
    try:
        parser = _Parser()
        parser.add_argument("mode", choices=["native-hold"])
        parser.add_argument("--release-directory", required=True)
        parser.add_argument("--attempt-sha256", required=True)
        parser.add_argument("--candidate-sha256", required=True)
        parser.add_argument("--nonce-sha256", required=True)
        parser.add_argument("--candidate-expiry", required=True)
        args = parser.parse_args()
        expiry = datetime.fromisoformat(args.candidate_expiry)

        def cancel(_signum, _frame):
            raise HoldCancelled()

        for signum in (signal.SIGTERM, signal.SIGINT):
            previous[signum] = signal.signal(signum, cancel)
        wait_for_release(args.release_directory, args.attempt_sha256, args.candidate_sha256,
                         args.nonce_sha256, expiry)
        if datetime.now(UTC) >= expiry:
            raise HoldExpired()
        from projetv0_voice import server

        if datetime.now(UTC) >= expiry:
            raise HoldExpired()
        server.main()
        return 0
    except HoldCancelled:
        print("material_hold_cancelled", file=sys.stderr)
        return 1
    except HoldExpired:
        print("material_hold_expired", file=sys.stderr)
        return 1
    except Exception:
        print("material_hold_refused", file=sys.stderr)
        return 1
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)


if __name__ == "__main__":
    raise SystemExit(main())
