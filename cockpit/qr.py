"""
cockpit.qr — minimal, stdlib-only QR code encoder (version 1..10, L error-level).

We deliberately avoid a third-party dep so the installer stays tiny. This is a
correct (not highly optimised) implementation that covers the payloads we need
(pairing URLs up to ~300 bytes). It returns a PNG (1-bit compressed) ready to
serve over HTTP.
"""
from __future__ import annotations

import struct
import zlib

# --- Reed-Solomon over GF(256) ---
_GF_EXP = [0] * 512
_GF_LOG = [0] * 256


def _init_gf() -> None:
    x = 1
    for i in range(255):
        _GF_EXP[i] = x
        _GF_LOG[x] = i
        x <<= 1
        if x & 0x100:
            x ^= 0x11D
    for i in range(255, 512):
        _GF_EXP[i] = _GF_EXP[i - 255]


_init_gf()


def _gf_mul(a: int, b: int) -> int:
    if a == 0 or b == 0:
        return 0
    return _GF_EXP[_GF_LOG[a] + _GF_LOG[b]]


def _rs_generator(n: int) -> list[int]:
    g = [1]
    for i in range(n):
        g = _poly_mul(g, [1, _GF_EXP[i]])
    return g


def _poly_mul(a: list[int], b: list[int]) -> list[int]:
    r = [0] * (len(a) + len(b) - 1)
    for i, av in enumerate(a):
        for j, bv in enumerate(b):
            r[i + j] ^= _gf_mul(av, bv)
    return r


def _rs_encode(data: list[int], nsym: int) -> list[int]:
    gen = _rs_generator(nsym)
    res = list(data) + [0] * nsym
    for i in range(len(data)):
        coef = res[i]
        if coef != 0:
            for j, gv in enumerate(gen):
                res[i + j] ^= _gf_mul(gv, coef)
    return res[len(data):]


# --- Version tables (L error level only, byte mode) ---
# (version, capacity_bytes_at_L, data_codewords, ec_codewords_per_block, num_blocks)
_VERSIONS = [
    (1, 17, 19, 7, 1),
    (2, 32, 34, 10, 1),
    (3, 53, 55, 15, 1),
    (4, 78, 80, 20, 1),
    (5, 106, 108, 26, 1),
    (6, 134, 136, 18, 2),
    (7, 154, 156, 20, 2),
    (8, 192, 194, 24, 2),
    (9, 230, 232, 30, 2),
    (10, 271, 274, 18, 4),
]


def _choose_version(n: int) -> tuple[int, int, int, int]:
    for v, cap, total, ec, blocks in _VERSIONS:
        if n <= cap:
            return v, total, ec, blocks
    raise ValueError("payload too large for built-in QR encoder")


def _bits_from_bytes(data: bytes, version: int) -> list[int]:
    # Mode indicator (byte = 0100) + char count indicator + data + terminator.
    bits: list[int] = []
    for b in (0, 1, 0, 0):
        bits.append(b)
    count_len = 8 if version < 10 else 16
    n = len(data)
    for i in range(count_len - 1, -1, -1):
        bits.append((n >> i) & 1)
    for byte in data:
        for i in range(7, -1, -1):
            bits.append((byte >> i) & 1)
    return bits


def _pad_bits(bits: list[int], total_data_codewords: int) -> list[int]:
    # Terminator (up to 4 zero bits), pad to byte, then 0xEC 0x11 alternation.
    max_bits = total_data_codewords * 8
    bits = list(bits)
    bits += [0] * min(4, max_bits - len(bits))
    while len(bits) % 8 != 0:
        bits.append(0)
    pad = [0xEC, 0x11]
    i = 0
    while len(bits) < max_bits:
        for b in range(7, -1, -1):
            bits.append((pad[i % 2] >> b) & 1)
        i += 1
    return bits[:max_bits]


def _bits_to_bytes(bits: list[int]) -> list[int]:
    out = []
    for i in range(0, len(bits), 8):
        v = 0
        for j in range(8):
            v = (v << 1) | bits[i + j]
        out.append(v)
    return out


# --- matrix placement ---
def _size(version: int) -> int:
    return 17 + 4 * version


def _place_finder(mat: list[list[int]], r: int, c: int) -> None:
    for dr in range(-1, 8):
        for dc in range(-1, 8):
            rr, cc = r + dr, c + dc
            if 0 <= rr < len(mat) and 0 <= cc < len(mat):
                if -1 <= dr <= 7 and -1 <= dc <= 7:
                    on = (0 <= dr <= 6 and 0 <= dc <= 6 and
                          (dr in (0, 6) or dc in (0, 6) or
                           (2 <= dr <= 4 and 2 <= dc <= 4)))
                    mat[rr][cc] = 1 if on else 0


def _alignment_positions(version: int) -> list[int]:
    if version == 1:
        return []
    tbl = {
        2: [6, 18], 3: [6, 22], 4: [6, 26], 5: [6, 30], 6: [6, 34],
        7: [6, 22, 38], 8: [6, 24, 42], 9: [6, 26, 46], 10: [6, 28, 50],
    }
    return tbl.get(version, [])


def _place_alignment(mat: list[list[int]], version: int) -> None:
    pos = _alignment_positions(version)
    for r in pos:
        for c in pos:
            if (r == 6 and c == 6) or (r == 6 and c == pos[-1]) or (r == pos[-1] and c == 6):
                continue
            for dr in range(-2, 3):
                for dc in range(-2, 3):
                    on = abs(dr) == 2 or abs(dc) == 2 or (dr == 0 and dc == 0)
                    mat[r + dr][c + dc] = 1 if on else 0


def _place_timing(mat: list[list[int]]) -> None:
    n = len(mat)
    for i in range(8, n - 8):
        if mat[6][i] == -1:
            mat[6][i] = (i + 1) % 2
        if mat[i][6] == -1:
            mat[i][6] = (i + 1) % 2


def _reserve_format(mat: list[list[int]]) -> None:
    n = len(mat)
    for i in range(9):
        if mat[8][i] == -1:
            mat[8][i] = 0
        if mat[i][8] == -1:
            mat[i][8] = 0
    for i in range(8):
        if mat[8][n - 1 - i] == -1:
            mat[8][n - 1 - i] = 0
        if mat[n - 1 - i][8] == -1:
            mat[n - 1 - i][8] = 0
    mat[n - 8][8] = 1  # dark module


def _mask(r: int, c: int, m: int) -> int:
    if m == 0:
        return (r + c) % 2 == 0
    if m == 1:
        return r % 2 == 0
    if m == 2:
        return c % 3 == 0
    if m == 3:
        return (r + c) % 3 == 0
    if m == 4:
        return ((r // 2) + (c // 3)) % 2 == 0
    if m == 5:
        return ((r * c) % 2) + ((r * c) % 3) == 0
    if m == 6:
        return (((r * c) % 2) + ((r * c) % 3)) % 2 == 0
    if m == 7:
        return (((r + c) % 2) + ((r * c) % 3)) % 2 == 0
    return 0


def _place_data(mat: list[list[int]], data_bits: list[int], mask_id: int) -> None:
    n = len(mat)
    i = 0
    col = n - 1
    up = True
    while col > 0:
        if col == 6:
            col -= 1
        for _ in range(n):
            for dc in (0, 1):
                c = col - dc
                r = (n - 1 - _) if up else _
                if mat[r][c] == -1:
                    if i < len(data_bits):
                        bit = data_bits[i]
                        i += 1
                    else:
                        bit = 0
                    if _mask(r, c, mask_id):
                        bit ^= 1
                    mat[r][c] = bit
        col -= 2
        up = not up


_FORMAT_TABLE_L = {
    0: 0b111011111000100, 1: 0b111001011110011, 2: 0b111110110101010, 3: 0b111100010011101,
    4: 0b110011000101111, 5: 0b110001100011000, 6: 0b110110001000001, 7: 0b110100101110110,
}


def _place_format(mat: list[list[int]], mask_id: int) -> None:
    n = len(mat)
    bits = _FORMAT_TABLE_L[mask_id]
    for i in range(15):
        b = (bits >> (14 - i)) & 1
        if i < 6:
            mat[8][i] = b
        elif i == 6:
            mat[8][7] = b
        elif i == 7:
            mat[8][8] = b
        elif i == 8:
            mat[7][8] = b
        else:
            mat[14 - i][8] = b
        if i < 8:
            mat[n - 1 - i][8] = b
        else:
            mat[8][n - 15 + i] = b


def _penalty(mat: list[list[int]]) -> int:
    n = len(mat)
    p = 0
    for r in range(n):
        for c in range(n - 4):
            if mat[r][c] == mat[r][c + 1] == mat[r][c + 2] == mat[r][c + 3] == mat[r][c + 4]:
                p += 3
    return p


def encode_qr(text: str) -> tuple[int, list[list[int]]]:
    """Return (size, matrix) where matrix[r][c] is 0 or 1."""
    data = text.encode("utf-8")
    version, total_data, ec_per_block, n_blocks = _choose_version(len(data))
    bits = _bits_from_bytes(data, version)
    bits = _pad_bits(bits, total_data)
    data_codewords = _bits_to_bytes(bits)
    # Split data codewords into two groups per QR spec (ISO/IEC 18004 §6.5.1):
    # group 1 has `n_short` blocks of `short_len` codewords; group 2 has `n_long`
    # blocks of `short_len + 1` codewords. short_len = total_data // n_blocks;
    # n_long = total_data % n_blocks. For v10-L this is 2×68 + 2×69 = 274.
    short_len = total_data // n_blocks
    n_long = total_data % n_blocks
    n_short = n_blocks - n_long
    blocks: list[list[int]] = []
    ec_blocks: list[list[int]] = []
    cursor = 0
    for b in range(n_blocks):
        blen = short_len if b < n_short else short_len + 1
        blk = data_codewords[cursor:cursor + blen]
        cursor += blen
        blocks.append(blk)
        ec_blocks.append(_rs_encode(blk, ec_per_block))
    # Interleave data codewords column-wise. Shorter blocks contribute nothing
    # to the final column (spec §6.5.2).
    max_data = max(len(b) for b in blocks)
    interleaved: list[int] = []
    for i in range(max_data):
        for b in blocks:
            if i < len(b):
                interleaved.append(b[i])
    for i in range(ec_per_block):
        for b in ec_blocks:
            interleaved.append(b[i])
    # back to bits
    final_bits: list[int] = []
    for v in interleaved:
        for i in range(7, -1, -1):
            final_bits.append((v >> i) & 1)

    n = _size(version)
    mat = [[-1] * n for _ in range(n)]
    _place_finder(mat, 0, 0)
    _place_finder(mat, 0, n - 7)
    _place_finder(mat, n - 7, 0)
    _place_alignment(mat, version)
    _place_timing(mat)
    _reserve_format(mat)

    # try all 8 masks, pick lowest penalty
    best = None
    for m in range(8):
        cand = [row[:] for row in mat]
        _place_data(cand, final_bits, m)
        _place_format(cand, m)
        p = _penalty(cand)
        if best is None or p < best[0]:
            best = (p, cand)
    return n, best[1]


def qr_png_bytes(text: str, module_px: int = 8, quiet: int = 4) -> bytes:
    """Render `text` as a 1-bit PNG and return its bytes."""
    n, mat = encode_qr(text)
    total = (n + quiet * 2) * module_px
    raw = bytearray()
    for y in range(total):
        raw.append(0)                                                           # filter: none
        for x in range(total):
            gx = x // module_px - quiet
            gy = y // module_px - quiet
            if 0 <= gx < n and 0 <= gy < n:
                pix = 0 if mat[gy][gx] else 255
            else:
                pix = 255
            raw.append(pix)

    def _chunk(tag: bytes, payload: bytes) -> bytes:
        c = zlib.crc32(tag + payload)
        return struct.pack(">I", len(payload)) + tag + payload + struct.pack(">I", c)

    sig = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">IIBBBBB", total, total, 8, 0, 0, 0, 0)
    idat = zlib.compress(bytes(raw), 9)
    png = sig + _chunk(b"IHDR", ihdr) + _chunk(b"IDAT", idat) + _chunk(b"IEND", b"")
    return png


__all__ = ["encode_qr", "qr_png_bytes"]
