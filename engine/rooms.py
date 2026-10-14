"""Fast +i rooms on five qubits, and the ten-step walk.

Walk images are w_r = F_{k_r} ... F_{k_1} v for
k = (0, 3, 5, 7, 9, 1, 2, 4, 6, 8).
"""

from __future__ import annotations

from itertools import combinations

from symplectic import (
    G,
    Op,
    add_vec,
    apply_op,
    basis_vector,
    clear_denominators,
    column_dependence,
    fmt_g,
    hermitian_inner,
    mul_op,
    rref,
    scale_vec,
    squares_to,
)

WALK = (0, 3, 5, 7, 9, 1, 2, 4, 6, 8)
_UNITS = (G(1), G(-1), G(0, 1), G(0, -1))


def fast_factors() -> dict[int, Op]:
    return {
        0: (0, "XXXXX"),
        3: (0, "XXXZI"),
        5: (0, "XXZII"),
        7: (0, "XZIII"),
        9: (0, "ZIIII"),
        1: (1, "XXXXY"),
        2: (1, "XXXYI"),
        4: (1, "XXYII"),
        6: (1, "XYIII"),
        8: (1, "YIIII"),
    }


def fast_address() -> dict[str, Op]:
    f = fast_factors()
    eye = (1, "IIIII")  # i I, used only as a phase carrier via products
    del eye
    return {
        "A": mul_op((1, "IIIII"), mul_op(f[1], f[2])),
        "B": mul_op((1, "IIIII"), mul_op(f[4], f[6])),
        "C": mul_op((1, "IIIII"), mul_op(f[0], f[3])),
        "D": mul_op((1, "IIIII"), mul_op(f[5], f[7])),
        "E_32": mul_op(mul_op(mul_op(mul_op(f[1], f[2]), f[4]), f[6]), f[8]),
        "C_32": mul_op(mul_op(mul_op(mul_op(f[0], f[3]), f[5]), f[7]), f[9]),
    }


def _equations(op: Op, lam: G) -> list[list[G]]:
    dim = 1 << len(op[1])
    columns = []
    for i in range(dim):
        image = apply_op(op, basis_vector(i, dim))
        image[i] = image[i] - lam
        columns.append(image)
    return [[columns[c][r] for c in range(dim)] for r in range(dim)]


def _kernel_line(rows: list[list[G]]) -> list[G]:
    reduced, pivots = rref(rows)
    width = len(rows[0])
    free = [col for col in range(width) if col not in set(pivots)]
    if len(free) != 1:
        raise ValueError(f"expected a line, found nullity {len(free)}")
    vec = [G() for _ in range(width)]
    vec[free[0]] = G(1)
    for row, pivot in enumerate(pivots):
        vec[pivot] = -reduced[row][free[0]]
    vec = clear_denominators(vec)
    head = next(i for i, value in enumerate(vec) if not value.is_zero())
    if vec[head] in _UNITS:
        vec = [value / vec[head] for value in vec]
    return vec


def room_lines() -> list[tuple[tuple[int, int, int, int], list[G]]]:
    """Sixteen +i lines labelled by (s_A, s_B, s_C, s_D)."""
    ops = fast_address()
    lines = []
    for s_a in (1, -1):
        for s_b in (1, -1):
            for s_c in (1, -1):
                for s_d in (1, -1):
                    rows: list[list[G]] = []
                    rows.extend(_equations(ops["A"], G(s_a)))
                    rows.extend(_equations(ops["B"], G(s_b)))
                    rows.extend(_equations(ops["C"], G(s_c)))
                    rows.extend(_equations(ops["D"], G(s_d)))
                    rows.extend(_equations(ops["E_32"], G(0, 1)))
                    lines.append(((s_a, s_b, s_c, s_d), _kernel_line(rows)))
    return lines


def walk_images(vec: list[G]) -> list[list[G]]:
    factors = fast_factors()
    current = vec
    images = []
    for index in WALK:
        current = apply_op(factors[index], current)
        images.append(current)
    return images


def walk_rank(vec: list[G]) -> tuple[int, list[int], list[G] | None]:
    return column_dependence(walk_images(vec))


def combine(terms: list[tuple[G, list[G]]]) -> list[G]:
    total = [G() for _ in range(32)]
    for coeff, vec in terms:
        total = add_vec(total, scale_vec(coeff, vec))
    return total


def pair_rank_report(lines: list[tuple[tuple[int, int, int, int], list[G]]]) -> tuple[int, int]:
    """Lines that differ only in s_D, combined as L + i L'.

    The pair (s_C, s_D) changes, and every such combination has rank 7.
    """
    by_sign = {signs: vec for signs, vec in lines}
    checked = 0
    rank7 = 0
    for (s_a, s_b, s_c, s_d), vec in lines:
        if s_d != -1:
            continue
        partner = by_sign[(s_a, s_b, s_c, 1)]
        combined = combine([(G(1), partner), (G(0, 1), vec)])
        checked += 1
        rank, _pivots, _rel = walk_rank(combined)
        if rank == 7:
            rank7 += 1
    return checked, rank7


def simplest_rank9(
    lines: list[tuple[tuple[int, int, int, int], list[G]]],
) -> tuple[list[tuple[tuple[int, int, int, int], G]], list[G], int, list[int], list[G]]:
    """Fewest unit-coefficient +i lines whose walk has rank 9.

    The order is: fewer lines, then lexicographic sign pattern, then
    lexicographic unit coefficients in the order 1, -1, i, -i.
    """
    # combinations() and the unit digits run in the simplicity order, so the
    # first success is the simplest witness.
    for count in range(1, 6):
        for choice in combinations(range(len(lines)), count):
            for powers in range(4**count):
                coeffs = []
                digits = powers
                for _ in range(count):
                    coeffs.append(_UNITS[digits & 3])
                    digits >>= 2
                vec = combine([(coeff, lines[i][1]) for coeff, i in zip(coeffs, choice, strict=True)])
                rank, pivots, relation = walk_rank(vec)
                if rank != 9 or relation is None:
                    continue
                return (
                    [(lines[i][0], coeff) for coeff, i in zip(coeffs, choice, strict=True)],
                    clear_denominators(vec),
                    rank,
                    pivots,
                    relation,
                )
    raise ValueError("no rank-9 witness with at most five unit-coefficient lines")


def format_relation(relation: list[G], pivots: list[int]) -> str:
    terms = []
    for index, coeff in enumerate(relation):
        if coeff.is_zero():
            continue
        terms.append(f"({fmt_g(coeff)}) w_{index + 1}")
    pivot_text = ",".join(str(index + 1) for index in pivots)
    return f"pivots {pivot_text}\nrelation " + " + ".join(terms) + " = 0"


def format_witness(found) -> str:
    terms, vec, rank, pivots, relation = found
    term_text = " ".join(f"({fmt_g(coeff)})*{signs}" for signs, coeff in terms)
    support = []
    for index, value in enumerate(vec):
        if not value.is_zero():
            support.append(f"v[{index}]={fmt_g(value)}")
    lines = [
        f"lines {term_text}",
        f"rank {rank}",
        "support " + " ".join(support),
        format_relation(relation, pivots),
    ]
    return "\n".join(lines)


def f9_bridge_ok(lines: list[tuple[tuple[int, int, int, int], list[G]]]) -> bool:
    factors = fast_factors()
    c32 = fast_address()["C_32"]
    for (s_a, s_b, s_c, s_d), vec in lines:
        del s_a, s_b
        image = apply_op(c32, vec)
        bridged = scale_vec(G(-(s_c * s_d)), apply_op(factors[9], vec))
        if image != bridged:
            return False
        if not hermitian_inner(vec, apply_op(factors[9], vec)).is_zero():
            return False
    return True


def factor_identities() -> list[str]:
    """Human-readable failures. Empty means the fast words match the specification."""
    factors = fast_factors()
    named = fast_address()
    expected = {
        "A": (0, "IIIZY"),
        "B": (0, "IZYII"),
        "C": (0, "IIIYX"),
        "D": (0, "IYZII"),
        "E_32": (3, "YZYZY"),
        "C_32": (2, "ZYZYX"),
    }
    failures = []
    for name, op in expected.items():
        if named[name] != op:
            failures.append(name)
    for index, op in factors.items():
        want = 1 if index in {0, 3, 5, 7, 9} else -1
        if squares_to(op) != want:
            failures.append(f"square {index}")
    return failures
