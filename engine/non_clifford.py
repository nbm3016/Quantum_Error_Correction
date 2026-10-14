"""Exact Clifford+T arithmetic in Q(i, sqrt(2)). No floats."""

from __future__ import annotations

from fractions import Fraction

from address import flip_signs, phi_c, psi
from lift_n import from_string, to_string
from presets import compile_pauli_to_gamma
from symplectic import G

Q = Fraction


class QISqrt2:
    """a + b i + c sqrt(2) + d i sqrt(2), with rational a,b,c,d."""

    __slots__ = ("a", "b", "c", "d")

    def __init__(self, a=0, b=0, c=0, d=0) -> None:
        self.a = a if isinstance(a, Fraction) else Q(a)
        self.b = b if isinstance(b, Fraction) else Q(b)
        self.c = c if isinstance(c, Fraction) else Q(c)
        self.d = d if isinstance(d, Fraction) else Q(d)

    def __add__(self, other: QISqrt2) -> QISqrt2:
        return QISqrt2(self.a + other.a, self.b + other.b, self.c + other.c, self.d + other.d)

    def __sub__(self, other: QISqrt2) -> QISqrt2:
        return QISqrt2(self.a - other.a, self.b - other.b, self.c - other.c, self.d - other.d)

    def __neg__(self) -> QISqrt2:
        return QISqrt2(-self.a, -self.b, -self.c, -self.d)

    def __mul__(self, other: QISqrt2) -> QISqrt2:
        a1, b1, c1, d1 = self.a, self.b, self.c, self.d
        a2, b2, c2, d2 = other.a, other.b, other.c, other.d
        return QISqrt2(
            a1 * a2 - b1 * b2 + 2 * c1 * c2 - 2 * d1 * d2,
            a1 * b2 + b1 * a2 + 2 * c1 * d2 + 2 * d1 * c2,
            a1 * c2 - b1 * d2 + c1 * a2 - d1 * b2,
            a1 * d2 + b1 * c2 + c1 * b2 + d1 * a2,
        )

    def conj(self) -> QISqrt2:
        return QISqrt2(self.a, -self.b, self.c, -self.d)

    def norm_sq(self) -> QISqrt2:
        return self * self.conj()

    def __truediv__(self, other: QISqrt2) -> QISqrt2:
        return self * other.inverse()

    def inverse(self) -> QISqrt2:
        norm = self.norm_sq()
        if norm.b != 0 or norm.d != 0:
            raise ValueError("norm is not in Q(sqrt(2))")
        p, q = norm.a, norm.c
        den = p * p - 2 * q * q
        if den == 0:
            raise ZeroDivisionError("zero Q(i, sqrt(2)) element")
        # 1/(p + q sqrt(2)) = (p - q sqrt(2)) / (p^2 - 2 q^2)
        inv_norm = QISqrt2(p / den, 0, -q / den, 0)
        return self.conj() * inv_norm

    def is_zero(self) -> bool:
        return self.a == 0 and self.b == 0 and self.c == 0 and self.d == 0

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, QISqrt2)
            and self.a == other.a
            and self.b == other.b
            and self.c == other.c
            and self.d == other.d
        )


def one() -> QISqrt2:
    return QISqrt2(1, 0, 0, 0)


def omega8() -> QISqrt2:
    """(1 + i) / sqrt(2) = sqrt(2)/2 + i sqrt(2)/2."""
    return QISqrt2(0, 0, Q(1, 2), Q(1, 2))


def from_g(value: G) -> QISqrt2:
    return QISqrt2(value.re, value.im, 0, 0)


def fmt_qi(value: QISqrt2) -> str:
    parts = []
    for coeff, label in ((value.a, ""), (value.b, "i"), (value.c, "sqrt(2)"), (value.d, "i*sqrt(2)")):
        if coeff == 0:
            continue
        parts.append(f"{coeff}" if label == "" else f"{coeff}*{label}")
    return "0" if not parts else " + ".join(parts)


def z_string(qubit: int, n: int = 7) -> str:
    letters = ["I"] * n
    letters[qubit - 1] = "Z"
    return "".join(letters)


def apply_word(state: dict[tuple[int, ...], QISqrt2], word: list[int]) -> dict[tuple[int, ...], QISqrt2]:
    out: dict[tuple[int, ...], QISqrt2] = {}
    for signs, coeff in state.items():
        if coeff.is_zero():
            continue
        current = signs
        phase = from_g(G(1))
        for k in word:
            phase = phase * from_g(phi_c(k, current))
            current = flip_signs(current, k)
        out[current] = out.get(current, QISqrt2()) + phase * coeff
    return {key: val for key, val in out.items() if not val.is_zero()}


def apply_pauli_string(state: dict[tuple[int, ...], QISqrt2], text: str) -> dict[tuple[int, ...], QISqrt2]:
    phase, indices = compile_pauli_to_gamma(text)
    out = apply_word(state, list(indices))
    factor = from_g(G(1) if phase == 0 else [G(1), G(0, 1), G(-1), G(0, -1)][phase])
    return {key: factor * val for key, val in out.items()}


def apply_t(state: dict[tuple[int, ...], QISqrt2], qubit: int, dagger: bool = False) -> dict[tuple[int, ...], QISqrt2]:
    root = omega8().conj() if dagger else omega8()
    half = QISqrt2(Q(1, 2), 0, 0, 0)
    ident = one()
    coeff_i = (ident + root) * half
    coeff_z = (ident - root) * half
    moved = apply_pauli_string(state, z_string(qubit))
    out: dict[tuple[int, ...], QISqrt2] = {}
    for signs, coeff in state.items():
        out[signs] = out.get(signs, QISqrt2()) + coeff_i * coeff
    for signs, coeff in moved.items():
        out[signs] = out.get(signs, QISqrt2()) + coeff_z * coeff
    return {key: val for key, val in out.items() if not val.is_zero()}


def norm_sq(state: dict[tuple[int, ...], QISqrt2]) -> QISqrt2:
    """Each address line has squared Euclidean norm 128 and the lines are orthogonal."""
    total = QISqrt2()
    line = QISqrt2(128, 0, 0, 0)
    for coeff in state.values():
        total = total + coeff.norm_sq() * line
    return total


def run_circuit(start: tuple[int, ...], gates: list[str]) -> dict[tuple[int, ...], QISqrt2]:
    state = {start: one()}
    for gate in gates:
        token = gate.strip().upper()
        if token.startswith("TD"):
            state = apply_t(state, int(token[2:]), dagger=True)
        elif token.startswith("T"):
            state = apply_t(state, int(token[1:]))
        elif token.startswith("G"):
            state = apply_word(state, [int(token[1:])])
        else:
            raise ValueError(f"bad gate {gate}")
    return state


def format_t(start: tuple[int, ...], gates: list[str]) -> str:
    state = run_circuit(start, gates)
    before = QISqrt2(128, 0, 0, 0)
    after = norm_sq(state)
    lines = [f"gates {','.join(gates)}", f"terms {len(state)}", f"norm {fmt_qi(after)}"]
    lines.append("norm PASS" if after == before else "norm FAIL")
    return "\n".join(lines)
