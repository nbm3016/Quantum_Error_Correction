"""Minimum-weight decoder in the staircase basis.

A lookup table covers every Pauli of weight at most t. Two errors that share
a syndrome are accepted only when their ratio is a stabilizer.
"""

from __future__ import annotations

from codes_high_d import (
    clock_seat,
    clock_z24_d9,
    extended_d5,
    five_qubit,
    steane,
    subsystem_4x32,
)
from lift_n import Pauli, commutes, from_string, identity, mul, symplectic_vector, to_string
from presets import compile_pauli

_BUILDERS = {
    "perfect-513": five_qubit,
    "subsystem-4x32": subsystem_4x32,
    "steane-713": steane,
    "clock-seat-833": clock_seat,
    "extended-d5": extended_d5,
    "clock-z24-d9": clock_z24_d9,
}
_CACHE: dict[str, dict] = {}


def code_by_name(name: str) -> dict:
    if name not in _CACHE:
        if name not in _BUILDERS:
            raise ValueError(f"unknown code {name}")
        _CACHE[name] = _BUILDERS[name]()
    return _CACHE[name]


def parse_error_list(text: str, n: int) -> Pauli:
    acc = identity(n)
    if text.strip() == "":
        return acc
    for piece in text.split(","):
        item = piece.strip().upper()
        letter = item[0]
        qubit = int(item[1:])
        if letter not in "IXYZ" or qubit < 1 or qubit > n:
            raise ValueError(f"bad error {piece}")
        letters = ["I"] * n
        letters[qubit - 1] = letter
        acc = mul(acc, from_string("".join(letters)))
    return acc


def _iter_errors(n: int, max_weight: int):
    from itertools import combinations

    yield 0, 0
    for weight in range(1, max_weight + 1):
        for support in combinations(range(n), weight):
            total = 3**weight
            for mask in range(total):
                x = 0
                z = 0
                value = mask
                for bit in support:
                    kind = value % 3
                    value //= 3
                    if kind == 0:
                        x |= 1 << bit
                    elif kind == 1:
                        z |= 1 << bit
                    else:
                        x |= 1 << bit
                        z |= 1 << bit
                yield x, z


def _syndrome(op: Pauli, generators: list[Pauli]) -> int:
    bits = 0
    for i, gen in enumerate(generators):
        if not commutes(op, gen):
            bits |= 1 << i
    return bits


def recovery_operator(code: dict, error: Pauli) -> Pauli:
    """Minimum-weight Pauli with the same syndrome as error."""
    target = _syndrome(error, code["stabilizers"])
    limit = int(code["t"])
    best: Pauli | None = None
    for x, z in _iter_errors(error.n, limit):
        trial = Pauli(0, x, z, error.n)
        if _syndrome(trial, code["stabilizers"]) != target:
            continue
        if best is None or trial.weight() < best.weight():
            best = trial
            if best.weight() == 0:
                break
    if best is None:
        raise ValueError("no recovery of weight <= t")
    return best


def try_recovery(code: dict, error: Pauli) -> Pauli | None:
    try:
        return recovery_operator(code, error)
    except ValueError:
        return None


def _coefficients(op: Pauli, generators: list[Pauli]) -> int | None:
    """Return generator coefficients over F_2, or None if op is outside the span."""
    data = [(symplectic_vector(gen), 1 << i) for i, gen in enumerate(generators)]
    target = symplectic_vector(op)
    coeff = 0
    width = 2 * op.n
    for col in range(width - 1, -1, -1):
        pivot = next((i for i, (row, _bit) in enumerate(data) if (row >> col) & 1), None)
        if pivot is None:
            if (target >> col) & 1:
                return None
            continue
        row, bit = data.pop(pivot)
        if (target >> col) & 1:
            target ^= row
            coeff ^= bit
        reduced = []
        for other, other_bit in data:
            if (other >> col) & 1:
                reduced.append((other ^ row, other_bit ^ bit))
            else:
                reduced.append((other, other_bit))
        data = reduced
    if target != 0:
        return None
    return coeff


def restores(code: dict, error: Pauli, recovery: Pauli) -> bool:
    """recovery * error equals a +1 stabilizer element."""
    prod = mul(recovery, error)
    coeff = _coefficients(prod, code["stabilizers"])
    if coeff is None:
        return False
    built = identity(prod.n)
    for i, gen in enumerate(code["stabilizers"]):
        if (coeff >> i) & 1:
            built = mul(built, gen)
    return built.phase == prod.phase and built.x == prod.x and built.z == prod.z


def decode_text(name: str, errors: str) -> tuple[Pauli, Pauli, tuple[int, tuple[int, ...]], bool]:
    code = code_by_name(name)
    error = parse_error_list(errors, int(code["n"]))
    recovery = try_recovery(code, error)
    if recovery is None:
        return error, error, (0, ()), False
    phase, indices = compile_pauli(recovery)
    ok = restores(code, error, recovery)
    return error, recovery, (phase, indices), ok
