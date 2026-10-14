"""Proposition 5.2: the 2n-generator staircase on n qubits.

Bit 0 of a mask is the rightmost qubit. The leftmost letter of a word is qubit 1.
No floats.
"""

from __future__ import annotations

from symplectic import mul_letter

_LETTER = "IXZY"


class Pauli:
    """phase in {0,1,2,3} for +1,+i,-1,-i, with x and z masks."""

    __slots__ = ("phase", "x", "z", "n")

    def __init__(self, phase: int, x: int, z: int, n: int) -> None:
        self.phase = phase % 4
        self.x = x
        self.z = z
        self.n = n

    def letter(self, bit: int) -> str:
        return _LETTER[((self.x >> bit) & 1) + 2 * ((self.z >> bit) & 1)]

    def weight(self) -> int:
        return (self.x | self.z).bit_count()

    def is_scalar(self) -> bool:
        return self.x == 0 and self.z == 0

    def key(self) -> tuple[int, int]:
        return self.x, self.z


def identity(n: int) -> Pauli:
    return Pauli(0, 0, 0, n)


def from_string(text: str, phase: int = 0) -> Pauli:
    raw = text.strip()
    n = len(raw)
    x = 0
    z = 0
    for i, letter in enumerate(raw):
        bit = n - 1 - i
        if letter == "X":
            x |= 1 << bit
        elif letter == "Z":
            z |= 1 << bit
        elif letter == "Y":
            x |= 1 << bit
            z |= 1 << bit
        elif letter != "I":
            raise ValueError(f"bad Pauli letter {letter}")
    return Pauli(phase, x, z, n)


def to_string(op: Pauli) -> str:
    letters = []
    for i in range(op.n):
        letters.append(op.letter(op.n - 1 - i))
    return "".join(letters)


def phase_name(phase: int) -> str:
    return ("+1", "+i", "-1", "-i")[phase % 4]


def mul(left: Pauli, right: Pauli) -> Pauli:
    if left.n != right.n:
        raise ValueError("Pauli length mismatch")
    phase = (left.phase + right.phase) % 4
    for bit in range(left.n):
        extra, _letter = mul_letter(left.letter(bit), right.letter(bit))
        phase = (phase + extra) % 4
    return Pauli(phase, left.x ^ right.x, left.z ^ right.z, left.n)


def scale(phase: int, op: Pauli) -> Pauli:
    return Pauli(op.phase + phase, op.x, op.z, op.n)


def inverse(op: Pauli) -> Pauli:
    for phase in range(4):
        trial = scale(phase, op)
        prod = mul(trial, op)
        if prod.is_scalar() and prod.phase == 0:
            return trial
    raise ValueError("Pauli has no inverse")


def commutes(left: Pauli, right: Pauli) -> bool:
    overlaps = (left.x & right.z).bit_count() + (left.z & right.x).bit_count()
    return overlaps % 2 == 0


def symplectic_vector(op: Pauli) -> int:
    """Pack x in the high n bits and z in the low n bits."""
    return (op.x << op.n) | op.z


def rank_mod2(rows: list[int], width: int) -> int:
    basis = [row for row in rows if row]
    rank = 0
    for col in range(width - 1, -1, -1):
        pivot = next((i for i in range(rank, len(basis)) if (basis[i] >> col) & 1), None)
        if pivot is None:
            continue
        basis[rank], basis[pivot] = basis[pivot], basis[rank]
        for i in range(len(basis)):
            if i != rank and (basis[i] >> col) & 1:
                basis[i] ^= basis[rank]
        rank += 1
    return rank


def build_n_qubit_gammas(n: int) -> list[Pauli]:
    """n-1 lifts of the seed (X, Y). Index 0 is the all-X string."""
    if n < 1:
        raise ValueError("n >= 1")
    words = ["X", "Y"]
    while len(words[0]) < n:
        width = len(words[0])
        lifted = ["X" + word for word in words]
        lifted.append("Y" + ("I" * width))
        lifted.append("Z" + ("I" * width))
        words = lifted
    return [from_string(word) for word in words]


def build_charge(n: int) -> Pauli:
    """Rightmost letter X, then Y and Z alternating toward the left."""
    letters = ["X"]
    y_turn = True
    while len(letters) < n:
        letters.insert(0, "Y" if y_turn else "Z")
        y_turn = not y_turn
    return from_string("".join(letters))


def build_n_qubit_dihedral(n: int) -> dict[str, Pauli]:
    gammas = build_n_qubit_gammas(n)
    product = identity(n)
    for op in gammas:
        product = mul(product, op)
    charge = build_charge(n)
    hermitian = None
    # Prefer Omega = -i P, the n=7 convention, when that representative is Hermitian.
    for phase in (3, 0, 1, 2):
        trial = scale(phase, product)
        square = mul(trial, trial)
        if trial.phase in (0, 2) and square.is_scalar() and square.phase == 0:
            hermitian = trial
            break
    if hermitian is None:
        raise ValueError("no Hermitian volume")
    exhaust = mul(hermitian, charge)
    return {"P": product, "Omega": hermitian, "C": charge, "E": exhaust, "gammas": gammas}


def _pair_ops(kind: str) -> tuple[str, str]:
    if kind == "face":
        return "YZ", "ZY"
    if kind == "tail":
        return "YZ", "ZY"
    raise ValueError(kind)


def build_odd_n_address_bits(n: int) -> list[tuple[str, Pauli]]:
    """n commuting involutions for odd n = 2m+1 >= 3.

    The n=7 operators are the router address. Each later disjoint pair
    (8,9), (10,11), ... adds YZ and -ZY and is invisible to Sigma.
    """
    if n < 3 or n % 2 == 0:
        raise ValueError("address bits are defined for odd n >= 3")
    bits: list[tuple[str, Pauli]] = [
        ("M", from_string("YZ" + "I" * (n - 2))),
        ("J_2", from_string("ZY" + "I" * (n - 2), phase=2)),
        ("D", from_string("IIIYZ" + "I" * (n - 5))),
        ("B", from_string("IIIZY" + "I" * (n - 5))),
        ("A", from_string("IIIIIZY" + "I" * (n - 7))),
        ("C_bit", from_string("IIIIIYX" + "I" * (n - 7))),
    ]
    sigma = ["I"] * n
    sigma[2] = "Y"
    sigma[3] = "Z"
    sigma[4] = "Y"
    sigma[5] = "Z"
    sigma[6] = "Y"
    bits.append(("Sigma", from_string("".join(sigma), phase=2)))
    qubit = 8
    extra = 0
    while qubit + 1 <= n:
        left = qubit - 1
        letters_y = ["I"] * n
        letters_z = ["I"] * n
        letters_y[left] = "Y"
        letters_y[left + 1] = "Z"
        letters_z[left] = "Z"
        letters_z[left + 1] = "Y"
        bits.append((f"M_{extra}", from_string("".join(letters_y))))
        bits.append((f"J_{extra}", from_string("".join(letters_z), phase=2)))
        extra += 1
        qubit += 2
    if len(bits) != n:
        raise ValueError(f"address count {len(bits)} != {n}")
    return bits


def address_independent(n: int) -> bool:
    ops = [op for _name, op in build_odd_n_address_bits(n)]
    if any(not commutes(ops[i], ops[j]) for i in range(n) for j in range(i + 1, n)):
        return False
    for op in ops:
        square = mul(op, op)
        if not square.is_scalar() or square.phase != 0:
            return False
    return rank_mod2([symplectic_vector(op) for op in ops], 2 * n) == n


def product_subset(ops: list[Pauli], indices: tuple[int, ...]) -> Pauli:
    prod = identity(ops[0].n if ops else 0)
    for index in indices:
        prod = mul(prod, ops[index])
    return prod
