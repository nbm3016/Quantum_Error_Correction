"""Stabilizer syndrome from the six twin products and Gamma_8.

The seven extractors determine the seven address signs. A Pauli error on a
known line has one coset, and the recovery is the phase-corrected Pauli that
returns that line exactly.
"""

from __future__ import annotations

from address import (
    A,
    B,
    CBIT,
    D,
    J,
    M,
    SIGMA,
    address_ops,
    fmt_signs,
    gammas,
    psi,
)
from symplectic import G, Op, apply_op, fmt_g, fmt_op, mul_op, parse_pauli, symplectic_product

# Extractor order: the six twins in the specification, then Gamma_8.
EXTRACTOR_NAMES = ("T_M", "T_J", "T_A", "T_B", "T_C", "T_D", "Gamma_8")
_ADDRESS_OF_TWIN = (M, J, A, B, CBIT, D)


def twins() -> list[Op]:
    gens = gammas()
    pairs = ((11, 13), (10, 12), (1, 2), (4, 6), (0, 3), (5, 7))
    return [mul_op(gens[i], gens[j]) for i, j in pairs]


def extractors() -> list[Op]:
    gens = gammas()
    return twins() + [gens[8]]


def eigenvalue(op: Op, vec: list[G]) -> G:
    image = apply_op(op, vec)
    for src, img in zip(vec, image, strict=True):
        if not src.is_zero():
            return img / src
    raise ValueError("zero vector has no eigenvalue")


def address_signs_of(vec: list[G]) -> tuple[int, ...]:
    ops = address_ops()
    order = ("M", "J_2", "D", "B", "A", "C_bit", "Sigma")
    signs = []
    for name in order:
        value = eigenvalue(ops[name], vec)
        if value == G(1):
            signs.append(1)
        elif value == G(-1):
            signs.append(-1)
        else:
            raise ValueError(f"{name} eigenvalue is not a sign: {fmt_g(value)}")
    return tuple(signs)


def _flip_bits(start: tuple[int, ...], found: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(0 if a == b else 1 for a, b in zip(start, found, strict=True))


def measured_syndrome(start: tuple[int, ...], errored: list[G]) -> tuple[int, ...]:
    """Seven bits: which extractor eigenvalue flipped, in EXTRACTOR_NAMES order."""
    start_vec = psi(start, "C")
    bits = []
    for op in extractors():
        before = eigenvalue(op, start_vec)
        after = eigenvalue(op, errored)
        bits.append(0 if before == after else 1)
    return tuple(bits)


def pauli_syndrome_bits(error: Op) -> tuple[int, ...]:
    """F_2 commutation bits of the error against the seven extractors."""
    return tuple(symplectic_product(error, op) for op in extractors())


def _enumerate_paulies(n: int = 7) -> list[Op]:
    total = 4**n
    out = []
    for mask in range(total):
        letters = []
        value = mask
        for _ in range(n):
            letters.append("IXYZ"[value & 3])
            value >>= 2
        # The enumeration above writes qubit 7 in the first letter. Reverse so
        # the leftmost letter is qubit 1, matching the high-bit symplectic order.
        out.append((0, "".join(reversed(letters))))
    return out


_CATALOG: list[tuple[Op, tuple[int, ...], int]] | None = None


def _catalog() -> list[tuple[Op, tuple[int, ...], int]]:
    global _CATALOG
    if _CATALOG is None:
        rows = []
        for op in _enumerate_paulies():
            bits = pauli_syndrome_bits(op)
            weight = sum(letter != "I" for letter in op[1])
            rows.append((op, bits, weight))
        _CATALOG = rows
    return _CATALOG


def recover(error: Op, start: tuple[int, ...]) -> tuple[Op, tuple[int, ...], tuple[int, ...]]:
    """Return (recovery, extractor syndrome, address flips)."""
    errored = apply_op(error, psi(start, "C"))
    syndrome = measured_syndrome(start, errored)
    arrived = address_signs_of(errored)
    flips = _flip_bits(start, arrived)
    best: Op | None = None
    best_key: tuple[int, str] | None = None
    for op, bits, weight in _catalog():
        if bits != syndrome:
            continue
        key = (weight, op[1])
        if best_key is None or key < best_key:
            best_key = key
            best = op
    if best is None:
        raise ValueError("no Pauli has this syndrome")
    image = apply_op(best, errored)
    start_vec = psi(start, "C")
    lam = None
    for src, img in zip(start_vec, image, strict=True):
        if not src.is_zero():
            lam = img / src
            break
    if lam is None:
        raise ValueError("start line is zero")
    recovery = _phase_inverse(lam, best)
    check = apply_op(recovery, errored)
    if check != start_vec:
        raise ValueError("recovery does not restore the start line")
    return recovery, syndrome, flips


def _phase_inverse(lam: G, op: Op) -> Op:
    """Multiply op by lam^{-1}, which is a unit for these lines."""
    unit = {G(1): 0, G(0, 1): 1, G(-1): 2, G(0, -1): 3}
    if lam not in unit:
        raise ValueError(f"line ratio is not a unit: {fmt_g(lam)}")
    inv = (4 - unit[lam]) % 4
    return (op[0] + inv) % 4, op[1]


def format_syndrome(error_text: str, start: tuple[int, ...]) -> str:
    error = parse_pauli(error_text)
    recovery, syndrome, flips = recover(error, start)
    names = ("s_M", "s_J", "s_D", "s_B", "s_A", "s_c", "s_Sigma")
    lines = [
        f"error {fmt_op(error)}",
        f"start {fmt_signs(start)}",
        "syndrome " + " ".join(f"{name}:{bit}" for name, bit in zip(EXTRACTOR_NAMES, syndrome, strict=True)),
        "address-flips " + " ".join(f"{name}:{bit}" for name, bit in zip(names, flips, strict=True)),
        f"recovery {fmt_op(recovery)}",
    ]
    return "\n".join(lines)
