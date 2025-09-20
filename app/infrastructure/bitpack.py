import numpy as np

def pack_phash64(bits_u8: np.ndarray) -> bytes:
    """bits_u8: shape (64,), valores 0/1 uint8 -> 8 bytes."""

    b = np.packbits(bits_u8.astype(np.uint8), bitorder="big")
    return b.tobytes()

def unpack_phash64(data: bytes) -> np.ndarray:
    """8 bytes -> (64,) uint8 0/1."""

    arr = np.frombuffer(data, dtype=np.uint8)
    bits = np.unpackbits(arr, bitorder="big")[:64].astype(np.uint8)
    return bits

def pack_bool_bits(mat_bool: np.ndarray) -> bytes:
    """
    mat_bool: shape (rows, cols) bool -> bytes empaquetados.
    cols normalmente 64.
    """

    flat = mat_bool.astype(np.uint8).reshape(-1)
    b = np.packbits(flat, bitorder="big")
    return b.tobytes()

def unpack_bool_bits(data: bytes, rows: int, cols: int) -> np.ndarray:
    """bytes -> (rows, cols) bool."""

    arr = np.frombuffer(data, dtype=np.uint8)
    bits = np.unpackbits(arr, bitorder="big")[: rows * cols]
    return bits.reshape(rows, cols).astype(bool)
