"""Lecture sélective d'une archive tar distante non compressée.

L'archive Task08_HepaticVessel.tar (9,3 Go) est un tar non compressé qui
contient des fichiers .nii.gz. Comme le serveur accepte les requêtes HTTP
Range, il suffit de lire les en-têtes tar (512 octets) pour connaître la
position de chaque fichier, puis de télécharger uniquement les octets des
cas sélectionnés. Le contenu des fichiers n'est pas modifié.
"""

from __future__ import annotations

import json
import time
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path

BLOCK = 512


@dataclass
class TarMember:
    name: str
    offset: int  # position des données dans l'archive
    size: int
    typeflag: str


def _parse_number(field: bytes) -> int:
    """Décode un champ numérique tar (octal ou base 256)."""
    if field and field[0] & 0x80:
        value = int.from_bytes(field[1:], "big")
        return value
    text = field.split(b"\0", 1)[0].strip()
    return int(text, 8) if text else 0


def _parse_pax(data: bytes) -> dict[str, str]:
    """Décode un en-tête étendu pax (format "longueur clé=valeur\\n")."""
    records = {}
    pos = 0
    while pos < len(data):
        space = data.index(b" ", pos)
        length = int(data[pos:space])
        record = data[space + 1 : pos + length - 1].decode("utf-8")
        key, _, value = record.partition("=")
        records[key] = value
        pos += length
    return records


def parse_header(block: bytes) -> tuple[str, int, str] | None:
    """Retourne (nom, taille, type) d'un bloc d'en-tête, ou None en fin d'archive."""
    if len(block) < BLOCK or block[:BLOCK] == b"\0" * BLOCK:
        return None
    name = block[0:100].split(b"\0", 1)[0].decode("utf-8")
    prefix = block[345:500].split(b"\0", 1)[0].decode("utf-8")
    if block[257:262] == b"ustar" and prefix:
        name = f"{prefix}/{name}"
    size = _parse_number(block[124:136])
    typeflag = block[156:157].decode("ascii") or "0"
    return name, size, typeflag


def _padded(size: int) -> int:
    return (size + BLOCK - 1) // BLOCK * BLOCK


class RangeReader:
    """Lecture d'intervalles d'octets sur une URL (avec reprises)."""

    def __init__(self, url: str, retries: int = 5, timeout: float = 60.0):
        self.url = url
        self.retries = retries
        self.timeout = timeout

    def read(self, start: int, length: int) -> bytes:
        request = urllib.request.Request(
            self.url, headers={"Range": f"bytes={start}-{start + length - 1}"}
        )
        for attempt in range(self.retries):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    return response.read()
            except OSError:
                if attempt == self.retries - 1:
                    raise
                time.sleep(2 * (attempt + 1))
        raise RuntimeError("unreachable")


def build_index(reader, chunk: int = 4096) -> list[TarMember]:
    """Parcourt les en-têtes de l'archive et retourne la liste des fichiers.

    `reader` doit fournir une méthode `read(start, length)`. Les en-têtes
    pax ('x') et GNU longname ('L') sont pris en compte.
    """
    members: list[TarMember] = []
    offset = 0
    buffer_start, buffer = 0, b""
    pending_name = None

    def get(start: int, length: int) -> bytes:
        nonlocal buffer_start, buffer
        end = start + length
        if not (buffer_start <= start and end <= buffer_start + len(buffer)):
            buffer_start, buffer = start, reader.read(start, max(length, chunk))
        return buffer[start - buffer_start : end - buffer_start]

    while True:
        header = parse_header(get(offset, BLOCK))
        if header is None:
            break
        name, size, typeflag = header
        data_offset = offset + BLOCK
        if typeflag in ("x", "L"):
            data = get(data_offset, size)
            if typeflag == "x":
                pending_name = _parse_pax(data).get("path", pending_name)
            else:
                pending_name = data.split(b"\0", 1)[0].decode("utf-8")
        elif typeflag == "g":
            pass
        else:
            members.append(TarMember(pending_name or name, data_offset, size, typeflag))
            pending_name = None
        offset = data_offset + _padded(size)
    return members


def save_index(members: list[TarMember], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([asdict(m) for m in members], indent=1), encoding="utf-8")


def load_index(path: Path) -> list[TarMember]:
    return [TarMember(**m) for m in json.loads(path.read_text(encoding="utf-8"))]


def download_member(url: str, member: TarMember, destination: Path, chunk: int = 4 << 20) -> None:
    """Télécharge un fichier de l'archive par morceaux, avec reprise possible."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size == member.size:
        return
    partial = destination.with_name(destination.name + ".part")
    done = partial.stat().st_size if partial.exists() else 0
    reader = RangeReader(url)
    with open(partial, "ab") as handle:
        while done < member.size:
            length = min(chunk, member.size - done)
            data = reader.read(member.offset + done, length)
            handle.write(data)
            done += len(data)
    if partial.stat().st_size != member.size:
        raise IOError(f"Taille inattendue pour {destination.name}")
    partial.replace(destination)
