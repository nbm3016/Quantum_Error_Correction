"""Pauli arithmetic on 7 qubits over Z[i] and F_2.

No floats. Leftmost letter is the slowest qubit, and it is the high bit
of a computational-basis index.
"""

from __future__ import annotations

from fractions import Fraction

LETTERS = ("I", "X", "Y", "Z")
PHASE_NAME = ("+1", "+i", "-1", "-i")


class G:
    """Gaussian rational a + b i."""

    __slots__ = ("re", "im")

    def __init__(self, re: int | Fraction = 0, im: int | Fraction = 0) -> None:
        self.re = re if isinstance(re, Fraction) else Fraction(re)
        self.im = im if isinstance(im, Fraction) else Fraction(im)

    def __add__(self, other: G) -> G:
        return G(self.re + other.re, self.im + other.im)

    def __sub__(self, other: G) -> G:
        return G(self.re - other.re, self.im - other.im)

    def __neg__(self) -> G:
        return G(-self.re, -self.im)

    def __mul__(self, other: G) -> G:
        return G(
            self.re * other.re - self.im * other.im,
            self.re * other.im + self.im * other.re,
        )

    def __truediv__(self, other: G) -> G:
        den = other.re * other.re + other.im * other.im
        if den == 0:
            raise ZeroDivisionError("division by zero Gaussian")
        return G(
            (self.re * other.re + self.im * other.im) / den,
            (self.im * other.re - self.re * other.im) / den,
        )

    def conj(self) -> G:
        return G(self.re, -self.im)

    def is_zero(self) -> bool:
        return self.re == 0 and self.im == 0

    def __eq__(self, other: object) -> bool:
        return isinstance(other, G) and self.re == other.re and self.im == other.im

    def __hash__(self) -> int:
        return hash((self.re, self.im))


PHASE = (G(1), G(0, 1), G(-1), G(0, -1))
Op = tuple[int, str]


def fmt_g(value: G) -> str:
    re, im = value.re, value.im
    if im == 0:
        return _fmt_rat(re)
    if re == 0:
        return _fmt_imag(im)
    return f"({_fmt_rat(re)}{_fmt_imag(im, signed=True)})"


def _fmt_rat(value: Fraction) -> str:
    if value.denominator == 1:
        return str(value.numerator)
    return f"{value.numerator}/{value.denominator}"


def _fmt_imag(value: Fraction, signed: bool = False) -> str:
    sign = ""
    if value < 0:
        sign = "-"
        value = -value
    elif signed:
        sign = "+"
    if value == 1:
        body = "i"
    elif value.denominator == 1:
        body = f"{value.numerator}*i"
    else:
        body = f"{value.numerator}/{value.denominator}*i"
    return sign + body


def fmt_op(op: Op) -> str:
    return PHASE_NAME[op[0]] + op[1]


def mul_letter(left: str, right: str) -> tuple[int, str]:
    if left == "I":
        return 0, right
    if right == "I":
        return 0, left
    if left == right:
        return 0, "I"
    cycle = "XYZ"
    diff = (cycle.index(right) - cycle.index(left)) % 3
    if diff == 1:
        return 1, cycle[(cycle.index(left) + 2) % 3]
    return 3, cycle[(cycle.index(left) + 1) % 3]


def mul_op(left: Op, right: Op) -> Op:
    phase = (left[0] + right[0]) % 4
    letters = []
    for a, b in zip(left[1], right[1], strict=True):
        extra, letter = mul_letter(a, b)
        phase = (phase + extra) % 4
        letters.append(letter)
    return phase, "".join(letters)


def scale(phase: int, op: Op) -> Op:
    return (op[0] + phase) % 4, op[1]


def ident(n: int) -> Op:
    return 0, "I" * n


def build_gammas(n: int = 14) -> list[Op]:
    """Six lifts from the seed (X, Y). Every phase stays +1."""
    if n % 2:
        raise ValueError("the even lift is required")
    gammas: list[Op] = [(0, "X"), (0, "Y")]
    while len(gammas) < n:
        width = len(gammas[0][1])
        lifted = [(phase, "X" + word) for phase, word in gammas]
        lifted.append((0, "Y" + ("I" * width)))
        lifted.append((0, "Z" + ("I" * width)))
        gammas = lifted
    return gammas


def y_count(op: Op) -> int:
    return op[1].count("Y")


def transpose_op(op: Op) -> Op:
    """Complex-linear transpose. Each Y contributes a minus."""
    extra = 2 * (y_count(op) % 2)
    return (op[0] + extra) % 4, op[1]


def comm_sign(left: Op, right: Op) -> int:
    ab = mul_op(left, right)
    ba = mul_op(right, left)
    if ab[1] != ba[1]:
        raise ValueError("product left the Pauli group")
    diff = (ab[0] - ba[0]) % 4
    if diff == 0:
        return 1
    if diff == 2:
        return -1
    raise ValueError("operators neither commute nor anticommute")


def squares_to(op: Op) -> int:
    phase, letters = mul_op(op, op)
    if set(letters) != {"I"}:
        raise ValueError("square is not scalar")
    if phase == 0:
        return 1
    if phase == 2:
        return -1
    raise ValueError("square is not ±I")


def act_index(letters: str, phase: int, index: int, nq: int) -> tuple[int, int]:
    out = index
    ph = phase
    for q, letter in enumerate(letters):
        shift = nq - 1 - q
        bit = (index >> shift) & 1
        if letter == "I":
            continue
        if letter == "Z":
            if bit:
                ph = (ph + 2) % 4
        elif letter == "X":
            out ^= 1 << shift
        else:
            out ^= 1 << shift
            ph = (ph + (1 if bit == 0 else 3)) % 4
    return out, ph


def apply_op(op: Op, vec: list[G]) -> list[G]:
    nq = len(op[1])
    dim = 1 << nq
    if len(vec) != dim:
        raise ValueError("vector length is not 2^n")
    out = [G() for _ in range(dim)]
    phase, letters = op
    for i, value in enumerate(vec):
        if value.is_zero():
            continue
        j, ph = act_index(letters, phase, i, nq)
        out[j] = out[j] + value * PHASE[ph]
    return out


def apply_word(ops: list[Op], vec: list[G]) -> list[G]:
    current = vec
    for op in ops:
        current = apply_op(op, current)
    return current


def to_symplectic(op: Op) -> tuple[int, int]:
    """Return (x_mask, z_mask) over F_2. Y sets both bits. Phase is dropped."""
    x_mask = 0
    z_mask = 0
    nq = len(op[1])
    for q, letter in enumerate(op[1]):
        shift = nq - 1 - q
        if letter == "X":
            x_mask |= 1 << shift
        elif letter == "Z":
            z_mask |= 1 << shift
        elif letter == "Y":
            x_mask |= 1 << shift
            z_mask |= 1 << shift
    return x_mask, z_mask


def symplectic_product(left: Op, right: Op) -> int:
    """F_2 commutation bit. Zero means the operators commute up to phase."""
    x1, z1 = to_symplectic(left)
    x2, z2 = to_symplectic(right)
    nq = len(left[1])
    bit = 0
    for q in range(nq):
        bit ^= ((x1 >> q) & 1) * ((z2 >> q) & 1)
        bit ^= ((z1 >> q) & 1) * ((x2 >> q) & 1)
    return bit


def parse_pauli(text: str, n: int = 7) -> Op:
    raw = text.strip().replace(" ", "")
    phase = 0
    if raw.startswith("+i") or raw.startswith("i"):
        phase = 1
        raw = raw[2:] if raw.startswith("+i") else raw[1:]
    elif raw.startswith("-i"):
        phase = 3
        raw = raw[2:]
    elif raw.startswith("+"):
        raw = raw[1:]
    elif raw.startswith("-"):
        phase = 2
        raw = raw[1:]
    if len(raw) != n or any(letter not in LETTERS for letter in raw):
        raise ValueError(f"expected a length-{n} Pauli, got {text}")
    return phase, raw


def basis_vector(index: int, dim: int) -> list[G]:
    vec = [G() for _ in range(dim)]
    vec[index] = G(1)
    return vec


def vector_equal(left: list[G], right: list[G]) -> bool:
    return len(left) == len(right) and all(a == b for a, b in zip(left, right, strict=True))


def scale_vec(factor: G, vec: list[G]) -> list[G]:
    return [factor * value for value in vec]


def add_vec(left: list[G], right: list[G]) -> list[G]:
    return [a + b for a, b in zip(left, right, strict=True)]


def first_nonzero(vec: list[G]) -> int | None:
    for i, value in enumerate(vec):
        if not value.is_zero():
            return i
    return None


def primitive_z(vec: list[G]) -> list[G]:
    """Clear denominators and cancel a common Gaussian content. First nonzero entry stays in Z[i]."""
    dens = []
    for value in vec:
        if value.is_zero():
            continue
        dens.append(value.re.denominator)
        dens.append(value.im.denominator)
    if not dens:
        return vec
    common = 1
    for den in dens:
        common = _lcm(common, den)
    scaled = [G(value.re * common, value.im * common) for value in vec]
    contents_re = []
    contents_im = []
    for value in scaled:
        if value.re.denominator != 1 or value.im.denominator != 1:
            raise ValueError("cleared vector is not integral")
        if value.re != 0:
            contents_re.append(abs(value.re.numerator))
        if value.im != 0:
            contents_im.append(abs(value.im.numerator))
    content = 0
    for number in contents_re + contents_im:
        content = number if content == 0 else _gcd(content, number)
    if content > 1:
        scaled = [G(value.re / content, value.im / content) for value in scaled]
    head = first_nonzero(scaled)
    if head is None:
        return scaled
    unit = scaled[head]
    return [value / unit for value in scaled]


def _gcd(a: int, b: int) -> int:
    while b:
        a, b = b, a % b
    return abs(a)


def _lcm(a: int, b: int) -> int:
    if a == 0 or b == 0:
        return 0
    return abs(a // _gcd(a, b) * b)


def rref(rows: list[list[G]]) -> tuple[list[list[G]], list[int]]:
    if not rows:
        return [], []
    matrix = [row[:] for row in rows]
    m = len(matrix)
    n = len(matrix[0])
    pivots: list[int] = []
    row = 0
    for col in range(n):
        pivot = next((i for i in range(row, m) if not matrix[i][col].is_zero()), None)
        if pivot is None:
            continue
        matrix[row], matrix[pivot] = matrix[pivot], matrix[row]
        pivot_value = matrix[row][col]
        matrix[row] = [entry / pivot_value for entry in matrix[row]]
        for i in range(m):
            if i == row or matrix[i][col].is_zero():
                continue
            factor = matrix[i][col]
            matrix[i] = [matrix[i][j] - factor * matrix[row][j] for j in range(n)]
        pivots.append(col)
        row += 1
        if row == m:
            break
    return matrix, pivots


def column_dependence(columns: list[list[G]]) -> tuple[int, list[int], list[G] | None]:
    """Rank, pivot column indices, and one primitive Z[i] relation among the columns."""
    if not columns:
        return 0, [], None
    dim = len(columns[0])
    matrix = [[columns[c][r] for c in range(len(columns))] for r in range(dim)]
    reduced, pivots = rref(matrix)
    pivot_set = set(pivots)
    free = [c for c in range(len(columns)) if c not in pivot_set]
    if not free:
        return len(pivots), pivots, None
    col = free[0]
    coeffs = [G() for _ in columns]
    coeffs[col] = G(1)
    for row, pivot_col in enumerate(pivots):
        coeffs[pivot_col] = -reduced[row][col]
    return len(pivots), pivots, coeffs


def clear_denominators(vec: list[G]) -> list[G]:
    """Return a primitive Z[i] vector in the same Q(i) line."""
    dens = [1]
    for value in vec:
        dens.append(value.re.denominator)
        dens.append(value.im.denominator)
    common = 1
    for den in dens:
        common = _lcm(common, den)
    scaled = [G(value.re * common, value.im * common) for value in vec]
    content = 0
    for value in scaled:
        for part in (value.re.numerator, value.im.numerator):
            if part:
                content = abs(part) if content == 0 else _gcd(content, abs(part))
    if content > 1:
        scaled = [G(value.re / content, value.im / content) for value in scaled]
    return scaled


def hermitian_inner(left: list[G], right: list[G]) -> G:
    total = G()
    for a, b in zip(left, right, strict=True):
        total = total + a.conj() * b
    return total
