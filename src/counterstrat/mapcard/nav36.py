"""Reader for Source 2 navigation meshes (`.nav`), including version 36.

awpy 2.0.2 (`awpy/nav.py`) stops at version 35. The v36 delta is documented by
ValveResourceFormat 20.0 ("Added nav mesh data previously skipped as unknown
bytes"), read from these files at tag `20.0`:

* `ValveResourceFormat/NavMesh/NavMeshFile.cs` — `Read()`: v36 inserts a binary
  KV3 document right after the `unk1` flags dword (`KV3Unknown1`) and a second
  one between the movable-mesh table and the area table (`KV3Unknown2`). Their
  contents are unknown even to VRF; the area layout itself is unchanged.
* `ValveResourceFormat/NavMesh/NavMeshArea.cs` — `Read()`: area record layout.
* `ValveResourceFormat/Resource/ResourceTypes/BinaryKV3.cs` — `ReadBuffer()`:
  the KV3 header, which is all we need to compute how many bytes to skip.

Only adjacency and geometry are read; ladders, transformed bounds, generation
params and custom data trail the area table and are left unparsed.
"""

import struct
from dataclasses import dataclass, field
from pathlib import Path

MAGIC = 0xFEEDFACE
MIN_VERSION = 30
MAX_VERSION = 36
KV3_MAGIC_PREFIX = 0x4B563300  # "KV3\x00" + version byte
KV3_UNCOMPRESSED = 0
KV3_LZ4 = 1
KV3_ZSTD = 2

Corner = tuple[float, float, float]


class NavUnsupportedError(Exception):
    """The nav file's version or byte layout could not be parsed."""


@dataclass
class NavArea:
    area_id: int
    corners: list[Corner] = field(default_factory=list)
    connections: list[int] = field(default_factory=list)


@dataclass
class NavMesh:
    version: int
    areas: dict[int, NavArea] = field(default_factory=dict)


class _Cursor:
    """Little-endian reader over the whole file, bounds-checked."""

    def __init__(self, data: bytes) -> None:
        self.data = data
        self.pos = 0

    def unpack(self, fmt: str) -> tuple:
        try:
            values = struct.unpack_from("<" + fmt, self.data, self.pos)
        except struct.error as exc:
            raise NavUnsupportedError(f"truncated nav file at offset {self.pos}") from exc
        self.pos += struct.calcsize("<" + fmt)
        return values

    def u32(self) -> int:
        return self.unpack("I")[0]

    def i32(self) -> int:
        return self.unpack("i")[0]

    def u16(self) -> int:
        return self.unpack("H")[0]

    def u8(self) -> int:
        return self.unpack("B")[0]

    def skip(self, count: int) -> None:
        if count < 0 or self.pos + count > len(self.data):
            raise NavUnsupportedError(f"nav file ends mid-record at offset {self.pos}")
        self.pos += count

    def null_str(self) -> bytes:
        end = self.data.find(b"\0", self.pos)
        if end < 0:
            raise NavUnsupportedError(f"unterminated string at offset {self.pos}")
        value = self.data[self.pos : end]
        self.pos = end + 1
        return value


def _skip_kv3(cur: _Cursor) -> None:
    """Skip one binary KV3 document without decoding it.

    Mirrors `BinaryKV3.ReadBuffer` (VRF 20.0) as far as byte accounting goes:
    the header is fixed-size per version and the payload sizes it declares are
    enough to find the end of the document.
    """
    cur.pos = (cur.pos + 7) & ~7  # NavMeshFile.ReadKV3 aligns to 8 bytes
    magic = cur.u32()
    version = magic & 0xFF
    if magic & 0xFFFFFF00 != KV3_MAGIC_PREFIX or not 1 <= version <= 5:
        raise NavUnsupportedError(f"unsupported embedded KV3 magic {magic:#010x}")
    if version < 5:
        raise NavUnsupportedError(f"unsupported embedded KV3 version {version}")

    cur.skip(16)  # format guid
    method = cur.u32()
    cur.skip(4)  # compression dictionary id + frame size
    cur.skip(16)  # counts: bytes1, bytes4, bytes8, types
    cur.skip(4)  # object + array counts
    cur.i32()  # uncompressed total (== buffer1 + buffer2)
    size_compressed_total = cur.i32()
    block_count = cur.i32()
    size_blobs = cur.i32()
    cur.skip(8)  # bytes2 count + block compressed sizes (v4+)
    size_uncompressed_1 = cur.i32()
    size_compressed_1 = cur.i32()
    size_uncompressed_2 = cur.i32()
    size_compressed_2 = cur.i32()
    cur.skip(32)  # buffer 2 counts (v5)

    if method == KV3_UNCOMPRESSED:
        cur.skip(size_uncompressed_1 + size_uncompressed_2)
        if block_count:
            cur.skip(size_blobs + 4)  # blobs + 0xFFEEDD00 trailer
    elif method == KV3_ZSTD:
        cur.skip(size_compressed_1 + size_compressed_2)
        if block_count:
            cur.skip(size_compressed_total - size_compressed_1 - size_compressed_2 + 4)
    elif method == KV3_LZ4:
        cur.skip(size_compressed_1 + size_compressed_2)
        if block_count:
            # Per-frame compressed lengths live inside the compressed buffer, so
            # the end of the document cannot be found without decoding it.
            raise NavUnsupportedError("embedded KV3 uses LZ4 with binary blobs")
    else:
        raise NavUnsupportedError(f"unknown embedded KV3 compression method {method}")


def _read_polygons(cur: _Cursor, version: int) -> list[list[Corner]]:
    corner_count = cur.u32()
    corners: list[Corner] = [cur.unpack("fff") for _ in range(corner_count)]

    polygons: list[list[Corner]] = []
    for _ in range(cur.u32()):
        polygon_corner_count = cur.u8()
        indices = cur.unpack(f"{polygon_corner_count}I")
        try:
            polygons.append([corners[i] for i in indices])
        except IndexError as exc:
            raise NavUnsupportedError("polygon references a corner out of range") from exc
        if version >= 35:
            cur.u32()  # movable mesh id
    return polygons


def _read_area(cur: _Cursor, version: int, polygons: list[list[Corner]]) -> NavArea:
    area_id = cur.u32()
    cur.skip(8)  # attribute flags
    cur.u8()  # hull index

    if version >= 31:
        polygon_index = cur.u32()
        if polygon_index >= len(polygons):
            raise NavUnsupportedError(f"area {area_id} references polygon {polygon_index}")
        corners = polygons[polygon_index]
    else:
        corners = [cur.unpack("fff") for _ in range(cur.u32())]

    cur.skip(4)  # almost always 0

    connections: list[int] = []
    for _ in range(len(corners)):
        for _ in range(cur.u32()):
            connections.append(cur.u32())
            cur.skip(4)  # edge id

    cur.skip(5)  # legacy hiding spot + spot encounter counts
    for _ in range(2):  # ladders above, ladders below
        cur.skip(4 * cur.u32())

    return NavArea(area_id=area_id, corners=corners, connections=connections)


def read_nav(path: Path | str) -> NavMesh:
    """Read a `.nav` mesh, keeping only area geometry and adjacency.

    Raises:
        FileNotFoundError: the file does not exist.
        NavUnsupportedError: the version or byte layout cannot be parsed.
    """
    nav_path = Path(path)
    if not nav_path.exists():
        raise FileNotFoundError(f"Nav mesh file not found: {nav_path}")

    cur = _Cursor(nav_path.read_bytes())
    magic = cur.u32()
    if magic != MAGIC:
        raise NavUnsupportedError(f"unexpected magic {magic:#010x}, expected {MAGIC:#010x}")

    version = cur.u32()
    if not MIN_VERSION <= version <= MAX_VERSION:
        raise NavUnsupportedError(f"unsupported nav version {version}")

    cur.u32()  # sub version
    cur.u32()  # unk1 flags (bit 0 = is_analyzed)

    if version >= 36:
        _skip_kv3(cur)

    polygons = _read_polygons(cur, version) if version >= 31 else []

    if version >= 32:
        cur.u32()  # unk2, always 0
    if version >= 35:
        for _ in range(cur.u32()):  # movable mesh ids
            cur.null_str()
            cur.skip(48)  # unverified baked reference transform
    if version >= 36:
        _skip_kv3(cur)

    areas: dict[int, NavArea] = {}
    for _ in range(cur.u32()):
        area = _read_area(cur, version, polygons)
        areas[area.area_id] = area

    return NavMesh(version=version, areas=areas)
