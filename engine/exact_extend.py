"""Exact Pauli-fault engine over Z[i, 1/sqrt(2)] and a symbolic rate p.

This file extends the router. It does not replace it. No floats.
"""

from __future__ import annotations

from fractions import Fraction

from codes_high_d import DODECA_STABS, STEANE_STABS, _ops, _weight_scan
from lift_n import Pauli, commutes, from_string, mul
from non_clifford import QISqrt2, one, omega8

Q = Fraction


class Poly:
    """Polynomial in one symbol p, coefficients in Q(i, sqrt(2))."""

    __slots__ = ("c",)

    def __init__(self, coeffs: tuple[QISqrt2, ...] | list[QISqrt2] | None = None) -> None:
        raw = list(coeffs or [])
        while raw and raw[-1].is_zero():
            raw.pop()
        self.c = tuple(raw)

    def __add__(self, other: Poly) -> Poly:
        n = max(len(self.c), len(other.c))
        out = []
        for i in range(n):
            a = self.c[i] if i < len(self.c) else QISqrt2()
            b = other.c[i] if i < len(other.c) else QISqrt2()
            out.append(a + b)
        return Poly(out)

    def __mul__(self, other: Poly) -> Poly:
        if not self.c or not other.c:
            return Poly()
        out = [QISqrt2() for _ in range(len(self.c) + len(other.c) - 1)]
        for i, a in enumerate(self.c):
            for j, b in enumerate(other.c):
                out[i + j] = out[i + j] + a * b
        return Poly(out)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Poly) and self.c == other.c


def mon(power: int, coeff: QISqrt2 | None = None) -> Poly:
    coeffs = [QISqrt2() for _ in range(power + 1)]
    coeffs[power] = one() if coeff is None else coeff
    return Poly(coeffs)


def one_minus_p() -> Poly:
    return Poly([one(), QISqrt2(-1)])


def sym_p() -> Poly:
    return mon(1)


class FaultLocation:
    """One incoherent Pauli fault. Identity has weight 1-p, the fault has weight p."""

    __slots__ = ("name", "kind", "pauli")

    def __init__(self, name: str, kind: str, pauli: str) -> None:
        self.name = name
        self.kind = kind
        self.pauli = pauli


def cultivation_locations() -> tuple[FaultLocation, ...]:
    """Injection followed by a T-dagger X check. Not a colour-code cultivation circuit."""
    return (
        FaultLocation("prep_plus", "prep", "Z"),
        FaultLocation("after_T", "gate", "X"),
        FaultLocation("after_T", "gate", "Y"),
        FaultLocation("after_T", "gate", "Z"),
        FaultLocation("measure_X", "measure", "flip"),
    )


def _t_amplitudes() -> tuple[QISqrt2, QISqrt2]:
    root = QISqrt2(0, 0, Q(1, 2), 0)
    return root, QISqrt2(Q(1, 2), Q(1, 2), 0, 0)


def _apply_z(amp0: QISqrt2, amp1: QISqrt2, qubit: int) -> tuple[QISqrt2, QISqrt2, QISqrt2, QISqrt2]:
    raise NotImplementedError


def _ideal_two_t() -> list[QISqrt2]:
    """Length-4 state |T>|T>, index = data*2 + ancilla."""
    t0, t1 = _t_amplitudes()
    return [t0 * t0, t0 * t1, t1 * t0, t1 * t1]


def _z_on(state: list[QISqrt2], qubit: str) -> list[QISqrt2]:
    out = state[:]
    for i, amp in enumerate(state):
        data, anc = divmod(i, 2) if False else (i // 2, i % 2)
        bit = data if qubit == "data" else anc
        if bit:
            out[i] = -amp
    return out


def _x_on_data(state: list[QISqrt2]) -> list[QISqrt2]:
    out = [QISqrt2() for _ in range(4)]
    for i, amp in enumerate(state):
        out[i ^ 2] = amp
    return out


def _cnot_data_ancilla(state: list[QISqrt2]) -> list[QISqrt2]:
    out = [QISqrt2() for _ in range(4)]
    for i, amp in enumerate(state):
        data, anc = i // 2, i % 2
        if data:
            anc ^= 1
        out[data * 2 + anc] = amp
    return out


def _branch_state(mask: int) -> tuple[list[QISqrt2], bool]:
    """Apply the six faults selected by mask, then say whether the X readout is +."""
    state = _ideal_two_t()
    # prep faults are Z before the ideal T is folded in, so they flip the T eigenvalue.
    # A Z before T is a Z after T, because T Z = Z T. We insert Z on that qubit.
    if mask & 1:
        state = _z_on(state, "data")
    if mask & 2:
        state = _z_on(state, "data")
    if mask & 4:
        state = _z_on(state, "ancilla")
    if mask & 8:
        state = _z_on(state, "ancilla")
    if mask & 16:
        state = _x_on_data(state)
    state = _cnot_data_ancilla(state)
    flipped = bool(mask & 32)
    return state, flipped


def _accept_and_fidelity(state: list[QISqrt2], flipped: bool) -> tuple[QISqrt2, QISqrt2]:
    """Return (acceptance probability, unnormalized <T|ρ|T> mass) for one pure branch."""
    plus0 = state[0] + state[1]
    plus1 = state[2] + state[3]
    minus0 = state[0] - state[1]
    minus1 = state[2] - state[3]
    half = QISqrt2(Q(1, 2), 0, 0, 0)
    if flipped:
        a0, a1 = minus0 * half, minus1 * half
    else:
        a0, a1 = plus0 * half, plus1 * half
    # |+> projector already includes 1/sqrt(2) twice via the two 1/2 factors? 
    # (<0|+<1|) / sqrt(2) applied to amplitudes, then |amp|^2 sums to the probability.
    acc = a0.norm_sq() + a1.norm_sq()
    t0, t1 = _t_amplitudes()
    overlap = t0.conj() * a0 + t1.conj() * a1
    # The postselected pure state is a/sqrt(acc). Fidelity mass is |overlap|^2 / acc * acc = |overlap|^2
    # if we later divide by total acceptance. Store |overlap|^2, which equals acc * F.
    return acc, overlap.norm_sq()


def _apply_pauli_1(amp0: QISqrt2, amp1: QISqrt2, letter: str) -> tuple[QISqrt2, QISqrt2]:
    if letter == "I":
        return amp0, amp1
    if letter == "Z":
        return amp0, -amp1
    if letter == "X":
        return amp1, amp0
    if letter == "Y":
        # Y|0> = i|1>, Y|1> = -i|0>.
        return QISqrt2(0, -1) * amp1, QISqrt2(0, 1) * amp0
    raise ValueError(letter)


def cultivation_fidelity(order: int = 3) -> tuple[Poly, Poly, tuple[QISqrt2, ...]]:
    """Post-selected fidelity of the injected state under a four-outcome Pauli channel.

    The channel after T is identity with weight 1-p and X, Y, Z with weight p/3 each.
    A following T-dagger and X measurement keeps the + outcome. The state whose
    fidelity is scored is the state before that check.
    """
    t0, t1 = _t_amplitudes()
    third = QISqrt2(Q(1, 3), 0, 0, 0)
    numer = Poly()
    denom = Poly()
    for letter, weight in (
        ("I", one_minus_p()),
        ("X", sym_p() * Poly([third])),
        ("Y", sym_p() * Poly([third])),
        ("Z", sym_p() * Poly([third])),
    ):
        a0, a1 = _apply_pauli_1(t0, t1, letter)
        # T-dagger returns the ideal branch to |+>. For a general Pauli, apply T-dagger
        # by sending |T> back: T†|0> = |0> component and T†(ω|1>) = |1> only on the ideal ray.
        # Conjugate the stored amplitudes by the diagonal of T-dagger: |1> gains ω*.
        root = omega8().conj()
        b0, b1 = a0, root * a1
        plus = b0 + b1
        acc = plus.norm_sq() * QISqrt2(Q(1, 2), 0, 0, 0)
        overlap = t0.conj() * a0 + t1.conj() * a1
        numer = numer + weight * Poly([acc * overlap.norm_sq()])
        denom = denom + weight * Poly([acc])
    return numer, denom, _series(numer, denom, order)


def _series(numer: Poly, denom: Poly, order: int) -> tuple[QISqrt2, ...]:
    if not denom.c or denom.c[0].is_zero():
        raise ValueError("acceptance vanishes at p = 0")
    inv0 = denom.c[0].inverse()
    out = []
    acc = Poly()
    for k in range(order + 1):
        num_k = numer.c[k] if k < len(numer.c) else QISqrt2()
        known = acc.c[k] if k < len(acc.c) else QISqrt2()
        coeff = (num_k - known) * inv0
        out.append(coeff)
        acc = acc + mon(k, coeff) * denom
    return tuple(out)


def stage1_infidelity_is_p() -> bool:
    """A single Z fault after T is orthogonal to |T>, so the fidelity is exactly 1-p."""
    t0, t1 = _t_amplitudes()
    overlap = t0.conj() * t0 + t1.conj() * (-t1)
    return overlap.is_zero()


def _stabilizer_fault_distance(strings: tuple[str, ...], logical: str) -> int:
    stabs = _ops(strings)
    logical_op = from_string(logical)
    best = len(strings[0]) + 1
    for weight, x, z in _weight_scan(logical_op.n, logical_op.n):
        err = Pauli(0, x, z, logical_op.n)
        if any(not commutes(err, stab) for stab in stabs):
            continue
        if commutes(err, logical_op):
            continue
        best = weight
        break
    return best


def steane_plus_t_fault_distance() -> int:
    """Pauli faults inserted after one T. They are ordinary Paulis, so the distance stays 3."""
    return _stabilizer_fault_distance(STEANE_STABS, "XXXXXXX")


def dodeca_fault_distance() -> int:
    return _stabilizer_fault_distance(DODECA_STABS, "XXXXXXXXXXX")


def transversal_steane() -> dict[str, bool]:
    stabs = _ops(STEANE_STABS)
    group = set()
    acc = [from_string("I" * 7)]
    for gen in stabs:
        acc = acc + [mul(item, gen) for item in acc]
    for item in acc:
        group.add(item.key())

    def image_in_group(op: Pauli, rule: str) -> bool:
        out = from_string("I" * 7)
        for qubit in range(7):
            letter = op.letter(6 - qubit)
            mapped = letter
            phase = 0
            if rule == "H":
                mapped = {"I": "I", "X": "Z", "Z": "X", "Y": "Y"}[letter]
                if letter == "Y":
                    phase = 2
            elif rule == "S":
                mapped = {"I": "I", "X": "Y", "Y": "X", "Z": "Z"}[letter]
                if letter == "Y":
                    phase = 2
            local = from_string("I" * qubit + mapped + "I" * (6 - qubit), phase)
            out = mul(out, local)
        return out.key() in group

    logical_x = from_string("XXXXXXX")
    h_image = from_string("I" * 7)
    for qubit in range(7):
        h_image = mul(h_image, from_string("I" * qubit + "Z" + "I" * (6 - qubit)))
    return {
        "stabs_closed_under_H": all(image_in_group(op, "H") for op in stabs),
        "stabs_closed_under_S": all(image_in_group(op, "S") for op in stabs),
        "H_sends_X_to_Z": h_image.key() == from_string("ZZZZZZZ").key(),
    }


def lattice_merge() -> dict[str, bool]:
    """ZZ seam between two 2-qubit repetition codes.

    Stabilizers ZZII and IIZZ, seam IZZI. The two original Z logicals
    differ by the product of those three stabilizers, so one logical qubit remains.
    """
    gens = [from_string("ZZII"), from_string("IIZZ"), from_string("IZZI")]
    logical_x = from_string("XXXX")
    logical_z = from_string("ZIII")
    other_z = from_string("IIIZ")
    seam_product = mul(mul(gens[0], gens[1]), gens[2])
    commute_ok = all(commutes(gens[i], gens[j]) for i in range(3) for j in range(i + 1, 3))
    logical_commute = all(commutes(logical_x, g) and commutes(logical_z, g) for g in gens)
    anticommute = not commutes(logical_x, logical_z)
    same_coset = mul(logical_z, other_z).key() == seam_product.key()
    return {
        "generators_commute": commute_ok,
        "merged_logicals_commute_with_stabilizers": logical_commute,
        "merged_logicals_anticommute": anticommute,
        "two_Z_logicals_differ_by_the_seam": same_coset,
    }


class Sym:
    """Polynomial in real symbols c, s reduced by c^2 + s^2 = 1. No floats."""

    __slots__ = ("t",)

    def __init__(self, terms: dict[tuple[int, int], QISqrt2] | None = None) -> None:
        self.t = _reduce_cs(dict(terms or {}))

    def __add__(self, other: Sym) -> Sym:
        out: dict[tuple[int, int], QISqrt2] = dict(self.t)
        for key, val in other.t.items():
            _accum(out, key, val)
        return Sym(out)

    def __mul__(self, other: Sym) -> Sym:
        out: dict[tuple[int, int], QISqrt2] = {}
        for (c1, s1), v1 in self.t.items():
            for (c2, s2), v2 in other.t.items():
                _accum(out, (c1 + c2, s1 + s2), v1 * v2)
        return Sym(out)

    def conj(self) -> Sym:
        return Sym({key: val.conj() for key, val in self.t.items()})

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Sym) and self.t == other.t


def _accum(bucket: dict[tuple[int, int], QISqrt2], key: tuple[int, int], val: QISqrt2) -> None:
    if val.is_zero():
        return
    bucket[key] = bucket.get(key, QISqrt2()) + val
    if bucket[key].is_zero():
        del bucket[key]


def _reduce_cs(raw: dict[tuple[int, int], QISqrt2]) -> dict[tuple[int, int], QISqrt2]:
    raw = {key: val for key, val in raw.items() if not val.is_zero()}
    while any(cp >= 2 for cp, _sp in raw):
        nxt: dict[tuple[int, int], QISqrt2] = {}
        for (cp, sp), val in raw.items():
            if cp >= 2:
                _accum(nxt, (cp - 2, sp), val)
                _accum(nxt, (cp - 2, sp + 2), -val)
            else:
                _accum(nxt, (cp, sp), val)
        raw = nxt
    return raw


def sym_cs(c_power: int, s_power: int, coeff: QISqrt2 | None = None) -> Sym:
    return Sym({(c_power, s_power): one() if coeff is None else coeff})


def depolarizing_1() -> tuple[tuple[str, Poly], ...]:
    """Pauli channel on one qubit, including every H. Weights (1-p, p/3, p/3, p/3)."""
    third = QISqrt2(Q(1, 3))
    share = sym_p() * Poly([third])
    return (("I", one_minus_p()), ("X", share), ("Y", share), ("Z", share))


def dephasing_z() -> tuple[tuple[str, Poly], ...]:
    """Z dephasing. Identity has weight 1-p and Z has weight p."""
    return (("I", one_minus_p()), ("Z", sym_p()))


def depolarizing_2() -> tuple[tuple[str, Poly], ...]:
    """The 15-element Pauli channel on one CNOT. Identity has weight 1-p, each other Pauli p/15."""
    fifteenth = QISqrt2(Q(1, 15))
    share = sym_p() * Poly([fifteenth])
    out = [("II", one_minus_p())]
    for left in "IXYZ":
        for right in "IXYZ":
            if left == "I" and right == "I":
                continue
            out.append((left + right, share))
    return tuple(out)


def _scale_poly(poly: Poly, coeff: QISqrt2) -> Poly:
    return Poly([coeff * item for item in poly.c])


def channel_weight_sum(branches: tuple[tuple[str, Poly], ...]) -> Poly:
    total = Poly()
    for _letter, weight in branches:
        total = total + weight
    return total


def hadamard_depolarizing_fidelity() -> Poly:
    """Fidelity to |+> after H on |0> and the one-qubit Pauli channel."""
    root = QISqrt2(0, 0, Q(1, 2), 0)
    plus = (root, root)
    total = Poly()
    for letter, weight in depolarizing_1():
        a0, a1 = _apply_pauli_1(plus[0], plus[1], letter)
        overlap = plus[0].conj() * a0 + plus[1].conj() * a1
        total = total + weight * Poly([overlap.norm_sq()])
    return total


def dephasing_fidelity_on_plus() -> Poly:
    root = QISqrt2(0, 0, Q(1, 2), 0)
    total = Poly()
    for letter, weight in dephasing_z():
        a0, a1 = _apply_pauli_1(root, root, letter)
        overlap = root.conj() * a0 + root.conj() * a1
        total = total + weight * Poly([overlap.norm_sq()])
    return total


def _pauli_matrix(letter: str) -> tuple[tuple[QISqrt2, QISqrt2], tuple[QISqrt2, QISqrt2]]:
    z = QISqrt2()
    i = QISqrt2(0, 1)
    if letter == "I":
        return ((one(), z), (z, one()))
    if letter == "X":
        return ((z, one()), (one(), z))
    if letter == "Y":
        return ((z, -i), (i, z))
    if letter == "Z":
        return ((one(), z), (z, -one()))
    raise ValueError(letter)


def _matmul(
    left: tuple[tuple[QISqrt2, QISqrt2], tuple[QISqrt2, QISqrt2]],
    right: tuple[tuple[QISqrt2, QISqrt2], tuple[QISqrt2, QISqrt2]],
) -> tuple[tuple[QISqrt2, QISqrt2], tuple[QISqrt2, QISqrt2]]:
    out = [[QISqrt2(), QISqrt2()], [QISqrt2(), QISqrt2()]]
    for row in range(2):
        for col in range(2):
            out[row][col] = left[row][0] * right[0][col] + left[row][1] * right[1][col]
    return ((out[0][0], out[0][1]), (out[1][0], out[1][1]))


def conjugate_by_t(letter: str, dagger_on_the_left: bool) -> tuple[tuple[QISqrt2, QISqrt2], tuple[QISqrt2, QISqrt2]]:
    """Return T P T† or T† P T, as a 2 by 2 matrix."""
    phase = omega8()
    gate = ((one(), QISqrt2()), (QISqrt2(), phase))
    gate_d = ((one(), QISqrt2()), (QISqrt2(), phase.conj()))
    pauli = _pauli_matrix(letter)
    if dagger_on_the_left:
        return _matmul(_matmul(gate_d, pauli), gate)
    return _matmul(_matmul(gate, pauli), gate_d)


def _scaled_xy(sign_y: int) -> tuple[tuple[QISqrt2, QISqrt2], tuple[QISqrt2, QISqrt2]]:
    """(X + sign_y Y) / sqrt(2). sign_y is +1 or -1."""
    root = QISqrt2(0, 0, Q(1, 2), 0)
    ex = _pauli_matrix("X")
    why = _pauli_matrix("Y")
    sign = QISqrt2(sign_y)
    mixed = (
        (ex[0][0] + sign * why[0][0], ex[0][1] + sign * why[0][1]),
        (ex[1][0] + sign * why[1][0], ex[1][1] + sign * why[1][1]),
    )
    return (
        (mixed[0][0] * root, mixed[0][1] * root),
        (mixed[1][0] * root, mixed[1][1] * root),
    )


def _neg_matrix(
    matrix: tuple[tuple[QISqrt2, QISqrt2], tuple[QISqrt2, QISqrt2]],
) -> tuple[tuple[QISqrt2, QISqrt2], tuple[QISqrt2, QISqrt2]]:
    return (
        (-matrix[0][0], -matrix[0][1]),
        (-matrix[1][0], -matrix[1][1]),
    )


def fault_before_t_identities() -> dict[str, bool]:
    """Local check of arXiv:2609.18922 equation (7) and the forward image in equation (16)."""
    x_minus_y = _scaled_xy(-1)
    x_plus_y = _scaled_xy(1)
    return {
        "Tdag_X_T_is_(X-Y)/sqrt(2)": conjugate_by_t("X", True) == x_minus_y,
        "Tdag_Y_T_is_(X+Y)/sqrt(2)": conjugate_by_t("Y", True) == x_plus_y,
        "Tdag_Z_T_is_Z": conjugate_by_t("Z", True) == _pauli_matrix("Z"),
        "T_X_Tdag_is_(X+Y)/sqrt(2)": conjugate_by_t("X", False) == x_plus_y,
        "T_Y_Tdag_is_(Y-X)/sqrt(2)": conjugate_by_t("Y", False) == _neg_matrix(x_minus_y),
        "T_Z_Tdag_is_Z": conjugate_by_t("Z", False) == _pauli_matrix("Z"),
    }


# Sourced from arXiv:2609.18922, not recomputed on their circuit.
SOURCED_D3_WEIGHT2 = ("X3 in the fold layers", "Y9 in the unfold layers")
SOURCED_D3_Q = "X0 X3 Y9"
COMPANION_URL = "https://github.com/kh428/exact_ler_for_msc"
COMPANION_SERIES = "data/series/d3_coefficients.csv"


def sourced_d3_not_recomputed() -> dict[str, object]:
    """The public series file was read. This engine did not contract the SOFT circuit."""
    return {
        "url": COMPANION_URL,
        "series_file": COMPANION_SERIES,
        "L2": Q(32, 75),
        "L3": Q(2009039, 4500),
        "acceptance_a0": Q(1, 1),
        "acceptance_a1": Q(77, 3),
        "acceptance_a2": Q(109433, 100),
        "recomputed_here": False,
        "weight2": SOURCED_D3_WEIGHT2,
        "Q": SOURCED_D3_Q,
    }


_Z_STABS = (0b1111000, 0b1100110, 0b1010101)  # IIIZZZZ, IZZIIZZ, ZIZIZIZ with bit i = qubit i
_X_STABS = _Z_STABS
_ANC = 7
_FLAG = 8


def _cnot_masks(x: int, z: int, control: int, target: int) -> tuple[int, int]:
    if (x >> control) & 1:
        x ^= 1 << target
    if (z >> target) & 1:
        z ^= 1 << control
    return x, z


def _had_masks(x: int, z: int, qubit: int) -> tuple[int, int]:
    xb = (x >> qubit) & 1
    zb = (z >> qubit) & 1
    x = (x & ~(1 << qubit)) | (zb << qubit)
    z = (z & ~(1 << qubit)) | (xb << qubit)
    return x, z


def _add_mask(x: int, z: int, qubit: int, letter: str) -> tuple[int, int]:
    if letter in ("X", "Y"):
        x ^= 1 << qubit
    if letter in ("Z", "Y"):
        z ^= 1 << qubit
    return x, z


def _steane_ops() -> list[tuple]:
    ops: list[tuple] = []
    slot = 0
    for mask in _Z_STABS:
        qubits = [i for i in range(7) if (mask >> i) & 1]
        for qubit in qubits:
            ops.append(("cnot", qubit, _ANC))
            slot += 1
            ops.append(("slot", slot))
        ops.append(("meas",))
    for mask in _X_STABS:
        qubits = [i for i in range(7) if (mask >> i) & 1]
        ops.append(("h", _ANC))
        for qubit in qubits:
            ops.append(("cnot", _ANC, qubit))
            slot += 1
            ops.append(("slot", slot))
        ops.append(("h", _ANC))
        ops.append(("meas",))
    return ops


def _in_span(target: int, gens: tuple[int, ...]) -> bool:
    for bits in range(1 << len(gens)):
        acc = 0
        for i, gen in enumerate(gens):
            if (bits >> i) & 1:
                acc ^= gen
        if acc == target:
            return True
    return False


def _is_steane_logical(dx: int, dz: int) -> bool:
    for mask in _Z_STABS:
        if (dx & mask).bit_count() & 1:
            return False
    for mask in _X_STABS:
        if (dz & mask).bit_count() & 1:
            return False
    if _in_span(dx, _X_STABS) and _in_span(dz, _Z_STABS):
        return False
    anti_x = (dz & 0b1111111).bit_count() & 1
    anti_z = (dx & 0b1111111).bit_count() & 1
    return bool(anti_x or anti_z)


def steane_extraction(faults: dict[int, tuple[tuple[int, str], ...]], flag: bool) -> dict[str, object]:
    """One round of Steane syndrome extraction. The flag, when on, wraps slot 2."""
    x = z = 0
    detectors: list[int] = []
    flag_fires = 0
    for qubit, letter in faults.get(0, ()):
        x, z = _add_mask(x, z, qubit, letter)
    for op in _steane_ops():
        if op[0] == "cnot":
            x, z = _cnot_masks(x, z, op[1], op[2])
        elif op[0] == "h":
            x, z = _had_masks(x, z, op[1])
        elif op[0] == "slot":
            if flag and op[1] == 2:
                x, z = _cnot_masks(x, z, _FLAG, _ANC)
            for qubit, letter in faults.get(op[1], ()):
                x, z = _add_mask(x, z, qubit, letter)
            if flag and op[1] == 2:
                x, z = _cnot_masks(x, z, _FLAG, _ANC)
                flag_fires = (z >> _FLAG) & 1
                x &= ~(1 << _FLAG)
                z &= ~(1 << _FLAG)
        else:
            detectors.append((x >> _ANC) & 1)
            x &= ~(1 << _ANC)
            z &= ~(1 << _ANC)
    dx = x & 0b1111111
    dz = z & 0b1111111
    return {
        "detectors": tuple(detectors),
        "dx": dx,
        "dz": dz,
        "logical": _is_steane_logical(dx, dz),
        "flag": flag_fires,
    }


def our_weight2_set() -> dict[str, object]:
    """Explicit weight-2 set on this Steane extraction. Not the colour-code pair from 2609.18922."""
    faults = {0: ((0, "Z"),), 2: ((_ANC, "Z"),)}
    bare = steane_extraction(faults, flag=False)
    flagged = steane_extraction(faults, flag=True)
    return {"bare": bare, "flagged": flagged, "faults": "Z on data qubit 0 before extraction; Z on the ancilla after the second CNOT of IIIZZZZ"}


def coherent_y_gap() -> Sym:
    """<Y> after U = c I - i s Z on |+>, minus the Pauli twirl, which has <Y> = 0."""
    root = QISqrt2(0, 0, Q(1, 2), 0)
    imag = QISqrt2(0, 1)
    amp0 = Sym({(1, 0): root, (0, 1): (-imag) * root})
    amp1 = Sym({(1, 0): root, (0, 1): imag * root})
    left = amp0.conj() * (sym_cs(0, 0, -imag) * amp1)
    right = amp1.conj() * (sym_cs(0, 0, imag) * amp0)
    return left + right


def check_fidelity(phase_conj: QISqrt2, order: int = 2) -> tuple[QISqrt2, ...]:
    """Conditional fidelity to |T> when the check multiplies |1> by phase_conj."""
    t0, t1 = _t_amplitudes()
    third = QISqrt2(Q(1, 3))
    numer = Poly()
    denom = Poly()
    for letter, weight in (
        ("I", one_minus_p()),
        ("X", sym_p() * Poly([third])),
        ("Y", sym_p() * Poly([third])),
        ("Z", sym_p() * Poly([third])),
    ):
        a0, a1 = _apply_pauli_1(t0, t1, letter)
        b0, b1 = a0, phase_conj * a1
        acc = (b0 + b1).norm_sq() * QISqrt2(Q(1, 2))
        overlap = t0.conj() * a0 + t1.conj() * a1
        numer = numer + weight * Poly([acc * overlap.norm_sq()])
        denom = denom + weight * Poly([acc])
    return _series(numer, denom, order)


def t_vs_s_proxy() -> dict[str, tuple[QISqrt2, ...]]:
    """Same noisy |T>, checked either by T† or by the S† proxy."""
    return {
        "T": check_fidelity(omega8().conj()),
        "S_proxy": check_fidelity(QISqrt2(0, -1)),
    }


def nodal_branches() -> tuple[tuple[str, QISqrt2, bool], ...]:
    """Every Pauli outcome of the one-qubit check. Post-selection is the + outcome of X after T†."""
    t0, t1 = _t_amplitudes()
    rows = []
    for letter in "IXYZ":
        a0, a1 = _apply_pauli_1(t0, t1, letter)
        b0, b1 = a0, omega8().conj() * a1
        acc = (b0 + b1).norm_sq() * QISqrt2(Q(1, 2))
        rows.append((letter, acc, not acc.is_zero()))
    return tuple(rows)


def repetition_ml() -> Poly:
    """Likelihood gap on the 3-qubit repetition code, syndrome of a single X on qubit 0.

    Weight-1 explanation minus the weight-2 explanation X on qubits 1 and 2.
    """
    quiet = one_minus_p()
    weight1 = sym_p() * quiet * quiet
    weight2 = sym_p() * sym_p() * quiet
    return weight1 + _scale_poly(weight2, QISqrt2(-1))


def cheaper_t_linear() -> tuple[QISqrt2, QISqrt2]:
    """Leading infidelity of one accepted injection, and of two independent injections."""
    _numer, _denom, series = cultivation_fidelity(2)
    one_inf = -series[1]
    squared = Poly(series) * Poly(series)
    two_inf = -squared.c[1]
    return one_inf, two_inf


def amplitude_damping_half() -> tuple[QISqrt2, QISqrt2]:
    """Amplitude damping at the one exact rate γ = 1/2, applied to |T|. Leakage is not this channel."""
    t0, t1 = _t_amplitudes()
    root = QISqrt2(0, 0, Q(1, 2), 0)
    kept = t0.norm_sq() + (root * t1).norm_sq()
    jumped = (root * t1).norm_sq()
    return kept, jumped


def repetition_graft() -> dict[str, bool]:
    """Smallest exact graft in this engine: grow one qubit to the 3-qubit repetition code.

    This is not the colour-code escape stage.
    """
    x, z = _cnot_masks(1, 0, 0, 1)
    x, z = _cnot_masks(x, z, 0, 2)
    zx, zz = _cnot_masks(0, 1, 0, 1)
    zx, zz = _cnot_masks(zx, zz, 0, 2)
    # Fresh |0> stabilizers IZI and IIZ become ZZI and ZIZ.
    s1x, s1z = _cnot_masks(0, 0b010, 0, 1)
    s1x, s1z = _cnot_masks(s1x, s1z, 0, 2)
    s2x, s2z = _cnot_masks(0, 0b100, 0, 1)
    s2x, s2z = _cnot_masks(s2x, s2z, 0, 2)
    return {
        "logical_X_becomes_XXX": x == 0b111 and z == 0,
        "logical_Z_stays_ZII": zx == 0 and zz == 0b001,
        "new_stabilizer_ZZI": s1x == 0 and s1z == 0b011,
        "new_stabilizer_ZIZ": s2x == 0 and s2z == 0b101,
    }


def run_extend() -> str:
    if not stage1_infidelity_is_p():
        raise AssertionError("stage-1 Z fault is not orthogonal to |T>")
    _numer, _denom, series = cultivation_fidelity(3)
    if series[0] != QISqrt2(1) or series[1] != QISqrt2(Q(-1, 6)) or series[2] != QISqrt2(Q(-1, 9)):
        raise AssertionError("post-selected fidelity is not 1 - p/6 - p^2/9 + O(p^3)")
    if series[3] != QISqrt2(Q(-2, 27)):
        raise AssertionError("cubic fidelity coefficient")
    if channel_weight_sum(depolarizing_1()) != Poly([one()]):
        raise AssertionError("one-qubit Pauli channel weights")
    if channel_weight_sum(depolarizing_2()) != Poly([one()]):
        raise AssertionError("CNOT Pauli channel weights")
    if len(depolarizing_2()) != 16:
        raise AssertionError("CNOT channel should have 16 Pauli branches")
    if hadamard_depolarizing_fidelity() != Poly([one(), QISqrt2(Q(-2, 3))]):
        raise AssertionError("H plus depolarizing fidelity")
    if dephasing_fidelity_on_plus() != Poly([one(), QISqrt2(-1)]):
        raise AssertionError("dephasing fidelity")
    images = fault_before_t_identities()
    if not all(images.values()):
        raise AssertionError(str(images))
    published = sourced_d3_not_recomputed()
    if published["recomputed_here"] or published["L2"] != Q(32, 75) or published["L3"] != Q(2009039, 4500):
        raise AssertionError("sourced d=3 series was altered or marked recomputed")
    found = our_weight2_set()
    bare = found["bare"]
    flagged = found["flagged"]
    if bare["detectors"] != (0, 0, 0, 0, 0, 0) or bare["dz"] != 0b1100001 or not bare["logical"] or bare["dx"] != 0:
        raise AssertionError("unflagged weight-2 set")
    if flagged["flag"] != 1 or not flagged["logical"]:
        raise AssertionError("flag did not reject the weight-2 logical")
    if coherent_y_gap() != sym_cs(1, 1, QISqrt2(2)):
        raise AssertionError("coherent minus twirl is not 2cs")
    gap = t_vs_s_proxy()
    if gap["T"][0] != QISqrt2(1) or gap["T"][1] != QISqrt2(Q(-1, 6)):
        raise AssertionError("T check linear term")
    if gap["S_proxy"][1] != QISqrt2(Q(-5, 3), 0, 1, 0):
        raise AssertionError("S-proxy linear term")
    # Leading infidelity ratio (S-proxy) / (T check) = 10 - 6 sqrt(2).
    proxy_inf = -gap["S_proxy"][1]
    true_inf = -gap["T"][1]
    if proxy_inf != true_inf * QISqrt2(10, 0, -6, 0):
        raise AssertionError("T versus S-proxy ratio")
    branches = nodal_branches()
    expected_acc = {"I": QISqrt2(1), "X": QISqrt2(Q(1, 2)), "Y": QISqrt2(Q(1, 2)), "Z": QISqrt2()}
    for letter, acc, accepted in branches:
        if acc != expected_acc[letter] or accepted == (letter == "Z"):
            raise AssertionError("nodal branch " + letter)
    ml_gap = repetition_ml()
    if ml_gap.c != (QISqrt2(), one(), QISqrt2(-3), QISqrt2(2)):
        raise AssertionError("repetition ML likelihood gap")
    one_inf, two_inf = cheaper_t_linear()
    if one_inf != QISqrt2(Q(1, 6)) or two_inf != QISqrt2(Q(1, 3)):
        raise AssertionError("two-injection cost")
    kept, jumped = amplitude_damping_half()
    if kept != QISqrt2(Q(3, 4)) or jumped != QISqrt2(Q(1, 4)):
        raise AssertionError("amplitude damping at 1/2")
    graft = repetition_graft()
    if not all(graft.values()):
        raise AssertionError(str(graft))
    if steane_plus_t_fault_distance() != 3:
        raise AssertionError("Steane+T Pauli fault distance")
    if dodeca_fault_distance() != 5:
        raise AssertionError("dodecacode Pauli fault distance")
    clifford = transversal_steane()
    if not all(clifford.values()):
        raise AssertionError(str(clifford))
    merge = lattice_merge()
    if not all(merge.values()):
        raise AssertionError(str(merge))
    return "EXACT EXTEND PASS"
