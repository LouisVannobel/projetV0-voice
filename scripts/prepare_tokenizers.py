"""Install the checksum-pinned Pipecat sentence data during the image build."""

from __future__ import annotations

import hashlib
import io
import stat
import sys
import urllib.request
import zipfile
from pathlib import Path

ARCHIVE_URL = (
    "https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/tokenizers/punkt_tab.zip"
)
ARCHIVE_SHA256 = "e57f64187974277726a3417ca6f181ec5403676c717672eef6a748a7b20e0106"
MAX_BYTES = 16 * 1024 * 1024
MEMBERS = {
    f"punkt_tab/{language}/{name}"
    for language in ("english", "french")
    for name in ("collocations.tab", "sent_starters.txt", "abbrev_types.txt", "ortho_context.tab")
}


def extract_tokenizers(raw: bytes, destination: Path) -> None:
    if len(raw) > MAX_BYTES or hashlib.sha256(raw).hexdigest() != ARCHIVE_SHA256:
        raise ValueError("Invalid pinned tokenizer archive")
    if destination.exists() or destination.is_symlink():
        raise ValueError("Tokenizer destination must be new")
    if any(parent.is_symlink() for parent in destination.parents):
        raise ValueError("Tokenizer destination cannot traverse a symlink")
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        selected = [entry for entry in archive.infolist() if entry.filename in MEMBERS]
        if len(selected) != 8 or {entry.filename for entry in selected} != MEMBERS:
            raise ValueError("Tokenizer archive must contain each required member once")
        if sum(entry.file_size for entry in selected) > MAX_BYTES:
            raise ValueError("Tokenizer data exceeds the extraction bound")
        contents = []
        for entry in selected:
            kind = stat.S_IFMT(entry.external_attr >> 16)
            if entry.is_dir() or kind not in (0, stat.S_IFREG) or entry.flag_bits & 1:
                raise ValueError("Tokenizer member must be an unencrypted regular file")
            if not 0 <= entry.file_size <= MAX_BYTES:
                raise ValueError("Tokenizer member exceeds the extraction bound")
            with archive.open(entry) as source:
                content = source.read(entry.file_size + 1)
            if len(content) != entry.file_size:
                raise ValueError("Tokenizer member length does not match its metadata")
            contents.append((entry.filename, content))
    destination.mkdir(mode=0o700)
    for name, content in contents:
        target = destination / "tokenizers" / name
        target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        with target.open("xb") as output:
            output.write(content)
        target.chmod(0o440)
    for directory in [destination, *(path for path in destination.rglob("*") if path.is_dir())]:
        directory.chmod(0o550)


if __name__ == "__main__":
    with urllib.request.urlopen(ARCHIVE_URL, timeout=60) as response:
        if not response.geturl().startswith("https://"):
            raise ValueError("Tokenizer download must use HTTPS")
        archive_bytes = response.read(MAX_BYTES + 1)
    extract_tokenizers(archive_bytes, Path(sys.argv[1]))
