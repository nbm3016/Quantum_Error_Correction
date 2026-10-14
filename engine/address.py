"""128 address lines, the dihedral triple, and both gauges.

Sign order is (s_M, s_J, s_D, s_B, s_A, s_c, s_Sigma).
"""

from __future__ import annotations

from symplectic import G, Op, PHASE, apply_op, build_gammas, fmt_g, mul_op, scale

SIGN_NAMES = ("s_M", "s_J", "s_D", "s_B", "s_A", "s_c", "s_Sigma")
# Positions in the sign tuple.
M, J, D, B, A, CBIT, SIGMA = range(7)

# Flip sets of Gamma_k, as sign-tuple indices.
FLIPS: tuple[tuple[int, ...], ...] = (
    (CBIT, SIGMA),  # 0
    (A,),  # 1
    (A,),  # 2
    (CBIT, SIGMA),  # 3
    (B,),  # 4
    (D, SIGMA),  # 5
    (B,),  # 6
    (D, SIGMA),  # 7
    (),  # 8
    (SIGMA,),  # 9
    (J,),  # 10
    (M,),  # 11
    (J,),  # 12
    (M,),  # 13
)

# Gauge C monomial: (phase index in PHASE, sign indices that multiply it).
_GAUGE_C: tuple[tuple[int, tuple[int, ...]], ...] = (
    (2, (M, J, A, CBIT, D, SIGMA)),
    (1, (M, J, A, CBIT, D, SIGMA)),
    (2, (M, J, CBIT, D, SIGMA)),
    (1, (M, J, A, D, SIGMA)),
    (0, (M, J, A, D, SIGMA)),
    (2, (M, J, A, B, D, SIGMA)),
    (1, (M, J, A, B, D, SIGMA)),
    (1, (M, J, A, B, SIGMA)),
    (0, (M, J, A, B, SIGMA)),
    (2, (M, J)),
    (1, (M, J)),
    (1, (M,)),
    (0, (M,)),
    (0, ()),
)

_U = {
    (1, 1): (G(1), G(0, -1), G(0, 1), G(-1)),
    (1, -1): (G(1), G(0, 1), G(0, 1), G(1)),
    (-1, 1): (G(1), G(0, -1), G(0, -1), G(1)),
    (-1, -1): (G(1), G(0, 1), G(0, -1), G(-1)),
}
# Joint +1-normalized eigenvectors of A=ZY and C_bit=YX.
# The coordinate list (1, 1, i, -i) is not an eigenvector of those operators.
_U2 = {
    (1, 1): (G(1), G(0, 1), G(-1), G(0, 1)),
    (1, -1): (G(1), G(0, 1), G(1), G(0, -1)),
    (-1, 1): (G(1), G(0, -1), G(1), G(0, 1)),
    (-1, -1): (G(1), G(0, -1), G(-1), G(0, -1)),
}


def gammas() -> list[Op]:
    return build_gammas(14)


def address_ops() -> dict[str, Op]:
    return {
        "M": (0, "YZIIIII"),
        "J_2": (2, "ZYIIIII"),
        "D": (0, "IIIYZII"),
        "B": (0, "IIIZYII"),
        "A": (0, "IIIIIZY"),
        "C_bit": (0, "IIIIIYX"),
        "Sigma": (2, "IIYZYZY"),
    }


def parse_signs(text: str) -> tuple[int, ...]:
    parts = [piece.strip() for piece in text.split(",")]
    if len(parts) != 7:
        raise ValueError("expected seven signs")
    signs = []
    for piece in parts:
        if piece in {"+1", "1", "+"}:
            signs.append(1)
        elif piece in {"-1", "-"}:
            signs.append(-1)
        else:
            raise ValueError(f"bad sign {piece}")
    return tuple(signs)


def fmt_signs(signs: tuple[int, ...]) -> str:
    return ",".join("+1" if bit == 1 else "-1" for bit in signs)


def flip_signs(signs: tuple[int, ...], k: int) -> tuple[int, ...]:
    flipped = set(FLIPS[k])
    return tuple(-bit if i in flipped else bit for i, bit in enumerate(signs))


def phi_c(k: int, signs: tuple[int, ...]) -> G:
    phase, indices = _GAUGE_C[k]
    value = PHASE[phase]
    for index in indices:
        if signs[index] == -1:
            value = -value
    return value


def bit_x(sign: int) -> int:
    return (1 - sign) // 2


def charge_q(signs: tuple[int, ...]) -> int:
    x_m, x_j, x_d, x_b, x_a, _x_c, x_s = (bit_x(sign) for sign in signs)
    total = (
        x_m
        + x_d
        + x_s
        + x_m * x_j
        + x_a * x_b
        + x_a * x_d
        + x_b * x_s
        + x_d * x_s
    )
    return total % 2


def phi_l(k: int, signs: tuple[int, ...]) -> G:
    dest = flip_signs(signs, k)
    value = phi_c(k, signs)
    if (charge_q(signs) + charge_q(dest)) % 2:
        value = -value
    return value


def phi(k: int, signs: tuple[int, ...], gauge: str) -> G:
    if gauge == "C":
        return phi_c(k, signs)
    if gauge == "L":
        return phi_l(k, signs)
    raise ValueError("gauge must be C or L")


def _block_u(s_left: int, s_right: int) -> tuple[G, G, G, G]:
    return _U[(s_left, s_right)]


def _block_u_prime(s_d: int, s_b: int) -> tuple[G, G, G, G]:
    return _U[(s_d, -s_b)]


def _block_u_second(s_a: int, s_c: int) -> tuple[G, G, G, G]:
    return _U2[(s_a, s_c)]


def _block_y(signs: tuple[int, ...]) -> tuple[G, G]:
    t = -signs[SIGMA] * signs[B] * signs[A]
    return G(1), G(0, t)


def blocks(signs: tuple[int, ...]) -> dict[str, tuple[G, ...]]:
    return {
        "u": _block_u(signs[M], signs[J]),
        "y": _block_y(signs),
        "u_prime": _block_u_prime(signs[D], signs[B]),
        "u_second": _block_u_second(signs[A], signs[CBIT]),
    }


def psi_c(signs: tuple[int, ...]) -> list[G]:
    """Gauge C line. Coordinate 0 is +1."""
    parts = blocks(signs)
    u, y, up, ud = parts["u"], parts["y"], parts["u_prime"], parts["u_second"]
    vec = [G() for _ in range(128)]
    for i12, a in enumerate(u):
        for i3, b in enumerate(y):
            for i45, c in enumerate(up):
                for i67, d in enumerate(ud):
                    index = ((i12 * 2 + i3) * 4 + i45) * 4 + i67
                    vec[index] = a * b * c * d
    return vec


def psi(signs: tuple[int, ...], gauge: str = "C") -> list[G]:
    base = psi_c(signs)
    if gauge == "C":
        return base
    if gauge == "L":
        factor = G(1) if charge_q(signs) == 0 else G(-1)
        return [factor * value for value in base]
    raise ValueError("gauge must be C or L")


def gamma8_eigenvalue(signs: tuple[int, ...]) -> G:
    value = G(1)
    for index in (M, J, A, B, SIGMA):
        if signs[index] == -1:
            value = -value
    return value


def e_eigenvalue(signs: tuple[int, ...]) -> G:
    """i * s_M * s_Sigma."""
    value = PHASE[1]
    if signs[M] == -1:
        value = -value
    if signs[SIGMA] == -1:
        value = -value
    return value


def all_sign_tuples() -> list[tuple[int, ...]]:
    out = []
    for mask in range(128):
        out.append(tuple(-1 if (mask >> shift) & 1 else 1 for shift in range(7)))
    return out


def dihedral() -> dict[str, Op]:
    gens = gammas()
    product = (0, "I" * 7)
    for op in gens:
        product = mul_op(product, op)
    omega = scale(3, product)  # -i * P
    charge = (0, "ZYZYZYX")
    exhaust = mul_op(omega, charge)
    return {"P": product, "Omega": omega, "C": charge, "E": exhaust}


def fmt_blocks(signs: tuple[int, ...]) -> str:
    parts = blocks(signs)
    lines = []
    for name in ("u", "y", "u_prime", "u_second"):
        body = ", ".join(fmt_g(value) for value in parts[name])
        lines.append(f"{name}=({body})")
    t = -signs[SIGMA] * signs[B] * signs[A]
    lines.append(f"t={t}")
    return "\n".join(lines)


def fmt_vector(vec: list[G]) -> str:
    return " ".join(fmt_g(value) for value in vec)


def route_phase(signs: tuple[int, ...], word: list[int], gauge: str) -> tuple[G, tuple[int, ...]]:
    """O(1) table routing. Returns the accumulated phase and the final signs."""
    phase = G(1)
    current = signs
    for k in word:
        phase = phase * phi(k, current, gauge)
        current = flip_signs(current, k)
    return phase, current


def apply_gammas(signs: tuple[int, ...], word: list[int], gauge: str) -> list[G]:
    vec = psi(signs, gauge)
    gens = gammas()
    for k in word:
        vec = apply_op(gens[k], vec)
    return vec
