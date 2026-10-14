"""O(n) staircase compiler. Subset products use increasing generator index."""

from __future__ import annotations

from lift_n import Pauli, from_string, identity, inverse, mul, phase_name, product_subset, scale, to_string
from lift_n import build_n_qubit_gammas
from rooms import fast_factors


def _pair_choice(current: Pauli, pair: tuple[Pauli, Pauli], bit: int, base_index: int) -> tuple[tuple[int, ...], Pauli]:
    want = current.letter(bit)
    options = ((), (0,), (1,), (0, 1))
    for choice in options:
        prod = identity(current.n)
        for offset in choice:
            prod = mul(prod, pair[offset])
        if prod.letter(bit) == want:
            indices = tuple(base_index + offset for offset in choice)
            return indices, mul(inverse(prod), current)
    raise ValueError("staircase pair does not match the target letter")


def compile_pauli(op: Pauli, gammas: list[Pauli] | None = None) -> tuple[int, tuple[int, ...]]:
    if gammas is None:
        gammas = build_n_qubit_gammas(op.n)
    if len(gammas) != 2 * op.n:
        raise ValueError("the staircase has 2n generators")
    current = Pauli(op.phase, op.x, op.z, op.n)
    chosen: list[int] = []
    for bit in range(op.n):
        base = 2 * bit
        indices, current = _pair_choice(current, (gammas[base], gammas[base + 1]), bit, base)
        chosen.extend(indices)
    if not current.is_scalar():
        raise ValueError("staircase remainder is not a phase")
    subset = tuple(chosen)
    built = scale(current.phase, product_subset(gammas, subset))
    if built.phase != op.phase or built.x != op.x or built.z != op.z:
        raise ValueError("compiled word does not rebuild the Pauli")
    return current.phase, subset


def compile_pauli_to_gamma(pauli_str: str) -> tuple[int, tuple[int, ...]]:
    """Return (phase, sorted Gamma indices). phase * Gamma_U equals the string."""
    return compile_pauli(from_string(pauli_str))


def _fast_operators() -> list[Pauli]:
    named = fast_factors()
    # Pairs from the right: (0,1), (2,3), (4,5), (6,7), (8,9).
    order = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
    ops = []
    for index in order:
        phase, letters = named[index]
        ops.append(from_string(letters, phase))
    return ops


def compile_fast_pauli_to_F(fast_str: str) -> tuple[int, tuple[int, ...]]:
    if len(fast_str) != 5:
        raise ValueError("fast Pauli strings have length 5")
    target = from_string(fast_str)
    ops = _fast_operators()
    names = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
    current = Pauli(target.phase, target.x, target.z, 5)
    chosen: list[int] = []
    for bit in range(5):
        pair = (ops[2 * bit], ops[2 * bit + 1])
        base_names = (names[2 * bit], names[2 * bit + 1])
        want = current.letter(bit)
        found = False
        for choice in ((), (0,), (1,), (0, 1)):
            prod = identity(5)
            for offset in choice:
                prod = mul(prod, pair[offset])
            if prod.letter(bit) == want:
                for offset in choice:
                    chosen.append(base_names[offset])
                current = mul(inverse(prod), current)
                found = True
                break
        if not found:
            raise ValueError("fast staircase missed a letter")
    if not current.is_scalar():
        raise ValueError("fast remainder is not a phase")
    subset = tuple(sorted(chosen))
    built = identity(5)
    # Multiply in increasing F index, which is the pair order.
    ordered = [ops[names.index(index)] for index in subset]
    for op in ordered:
        built = mul(built, op)
    built = scale(current.phase, built)
    if built.x != target.x or built.z != target.z or built.phase != target.phase:
        raise ValueError("compiled fast word does not rebuild the Pauli")
    return current.phase, subset


def format_word(phase: int, indices: tuple[int, ...], symbol: str) -> str:
    body = "{" + ",".join(str(index) for index in indices) + "}"
    return f"{phase_name(phase)} {symbol}_{body}"


def rebuild_string(pauli_str: str) -> str:
    phase, indices = compile_pauli_to_gamma(pauli_str)
    gammas = build_n_qubit_gammas(len(pauli_str))
    return phase_name(phase) + to_string(scale(phase, product_subset(gammas, indices)))
