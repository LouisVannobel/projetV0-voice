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

IMAGE_DIGEST = "sha256:a444705d6ba207f542d4cb8691ac7303c605de3ec5bcb22e7cdc879ffcd238f4"
HASH = re.compile(r"[0-9a-f]{64}\Z")
KEYS = {"schema_version", "attempt_sha256", "candidate_sha256", "image_digest", "nonce"}
_MISSING = object()


class HoldRefused(Exception):
    """Unsafe control state or invalid input; details are deliberately private."""


class HoldExpired(HoldRefused):
    """Original expiry or maximum monotonic wait reached."""


class HoldCancelled(HoldRefused):
    """Termination during startup hold."""


def derive_held_compose(before_spec, expected_preimage_sha256, original_expiry,
                        candidate_expiry, new_environment, attempt_sha256,
                        candidate_sha256, nonce_sha256):
    """Derive a bounded selection; caller separately authenticates and seals context."""
    try:
        if (type(before_spec) is not str
                or any(type(value) is not str or not HASH.fullmatch(value)
                       for value in (expected_preimage_sha256, attempt_sha256,
                                     candidate_sha256, nonce_sha256))
                or not hmac.compare_digest(hashlib.sha256(before_spec.encode("utf-8")).hexdigest(),
                                           expected_preimage_sha256)):
            raise HoldRefused()
        for expiry in (original_expiry, candidate_expiry):
            if (not isinstance(expiry, datetime) or expiry.tzinfo is None
                    or expiry.utcoffset() is None or expiry.utcoffset().total_seconds() != 0):
                raise HoldRefused()
        if original_expiry != candidate_expiry:
            raise HoldRefused()

        import copy
        from uuid import UUID

        import yaml

        from projetv0_voice.config import _UniqueKeySafeLoader

        old_image = "sha256:4a0704afbfbee02aaaefb4fcaffa8201b9ead3e696ca83cc8a851cf5cd428c26"
        repository = "ghcr.io/louisvannobel/projetv0-voice@"
        document = yaml.load(before_spec, Loader=_UniqueKeySafeLoader)
        pending = [document]
        seen = set()
        while pending:
            node = pending.pop()
            if isinstance(node, (dict, list)):
                identity = id(node)
                if identity in seen:
                    raise HoldRefused()
                seen.add(identity)
                if isinstance(node, dict):
                    pending.extend(node.keys())
                    pending.extend(node.values())
                else:
                    pending.extend(node)
        if (type(document) is not dict or type(document.get("services")) is not dict
                or len(document["services"]) != 1):
            raise HoldRefused()
        name, service = next(iter(document["services"].items()))
        if (name != "voice" or type(service) is not dict
                or service.get("image") != repository + old_image
                or service.get("restart") != "no" or service.get("read_only") is not True
                or ("user" in service and service["user"] != "10001:10001")
                or service.get("command") is not None
                or service.get("entrypoint") not in (
                    None, ["python", "-m", "projetv0_voice.server"])):
            raise HoldRefused()
        old_environment = service.get("environment")
        mutable = {"VOICE_IMAGE_DIGEST", "VOICE_QUALIFICATION_RUN_ID",
                   "VOICE_AGENT_BUNDLE_SHA256", "VOICE_INFERENCE_PROFILE_SHA256"}
        if (type(old_environment) is not dict or type(new_environment) is not dict
                or set(old_environment) != set(new_environment)
                or not mutable.issubset(old_environment)):
            raise HoldRefused()
        for environment, image in ((old_environment, old_image), (new_environment, IMAGE_DIGEST)):
            if (any(type(key) is not str or type(value) is not str
                    for key, value in environment.items())
                    or environment["VOICE_IMAGE_DIGEST"] != repository + image
                    or environment.get("VOICE_QUALIFICATION_CANDIDATE_PATH")
                    != "/run/sparra-v2/candidate.json"
                    or str(UUID(environment["VOICE_QUALIFICATION_RUN_ID"]))
                    != environment["VOICE_QUALIFICATION_RUN_ID"]
                    or any(not HASH.fullmatch(environment[key])
                           for key in ("VOICE_AGENT_BUNDLE_SHA256",
                                       "VOICE_INFERENCE_PROFILE_SHA256"))):
                raise HoldRefused()
        if any(old_environment[key] != new_environment[key]
               for key in old_environment if key not in mutable):
            raise HoldRefused()

        mounts = service.get("volumes")
        if type(mounts) is not list:
            raise HoldRefused()
        targets = []
        candidate_index = None
        added_targets = ("/run/sparra-material/material_handover_1465.py",
                         "/run/sparra-material/control")
        for index, mount in enumerate(mounts):
            if type(mount) is dict:
                target = mount.get("target")
            elif type(mount) is str:
                parts = mount.split(":")
                if len(parts) not in (2, 3):
                    raise HoldRefused()
                target = parts[1]
            else:
                raise HoldRefused()
            if (type(target) is not str or not target.startswith("/") or target == "/"
                    or any(part in ("", ".", "..") for part in target.split("/")[1:])
                    or target in targets):
                raise HoldRefused()
            targets.append(target)
            if target == "/run/sparra-material" or target.startswith("/run/sparra-material/"):
                raise HoldRefused()
            for reserved in added_targets:
                if (target == reserved or target.startswith(reserved + "/")
                        or reserved.startswith(target + "/")):
                    raise HoldRefused()
            if target == "/run/sparra-v2":
                if (type(mount) is not dict or mount.get("type") != "bind"
                        or mount.get("read_only") is not True
                        or type(mount.get("source")) is not str
                        or not mount["source"].startswith("/")
                        or any(part in ("", ".", "..")
                               for part in mount["source"].split("/")[1:])):
                    raise HoldRefused()
                candidate_index = index
            elif target.startswith("/run/sparra-v2/") or "/run/sparra-v2".startswith(target + "/"):
                raise HoldRefused()
        if candidate_index is None:
            raise HoldRefused()

        selected = copy.deepcopy(document)
        held = selected["services"][name]
        held["image"] = repository + IMAGE_DIGEST
        held["environment"] = copy.deepcopy(new_environment)
        held["volumes"][candidate_index]["source"] = (
            "/opt/studio/sparra-dokploy/material-handover-1465/build"
        )
        additions = [
            {"type": "bind", "source": (
                "/opt/studio/sparra-dokploy/material-handover-1465-source/"
                "material_handover_1465.py"),
             "target": added_targets[0], "read_only": True},
            {"type": "bind", "source": "/opt/studio/sparra-dokploy/material-handover-1465/control",
             "target": added_targets[1], "read_only": True},
        ]
        held["volumes"].extend(additions)
        held["entrypoint"] = ["python", added_targets[0]]
        held["command"] = ["native-hold", "--release-directory", added_targets[1],
                           "--attempt-sha256", attempt_sha256,
                           "--candidate-sha256", candidate_sha256,
                           "--nonce-sha256", nonce_sha256,
                           "--candidate-expiry", candidate_expiry.isoformat()]

        # Reverse only the explicit delta to prove every retained field is unchanged.
        retained = copy.deepcopy(selected)
        restored = retained["services"][name]
        for key in ("image", "environment", "entrypoint", "command"):
            if key in service:
                restored[key] = copy.deepcopy(service[key])
            else:
                del restored[key]
        if restored["volumes"][-2:] != additions:
            raise HoldRefused()
        del restored["volumes"][-2:]
        restored["volumes"][candidate_index]["source"] = mounts[candidate_index]["source"]
        if retained != document:
            raise HoldRefused()
        return yaml.safe_dump(selected, sort_keys=False)
    except Exception:
        raise HoldRefused() from None


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
