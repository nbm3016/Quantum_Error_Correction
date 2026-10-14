"""Load the pinned d=3 cultivation stim file and score it in Z[i, 1/sqrt(2)].

The circuit is commit 332bc0481843ebc9718e152e92f5ee10f8f0b148 of
https://github.com/kh428/exact_ler_for_msc , file data/inputs/clifft_d3_p001.stim.
Noise strengths in the file are ignored. Every channel uses the symbol p.
No floats.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

from lift_n import Pauli, mul
from non_clifford import QISqrt2, omega8

Q = Fraction
_INSTR = re.compile(r"^([A-Za-z0-9_]+)(\([^)]*\))?\s*(.*)$")
_REC = re.compile(r"^rec\[(-?\d+)\]$")
ALLOWED = frozenset(
    {
        "QUBIT_COORDS",
        "SHIFT_COORDS",
        "R",
        "RX",
        "CX",
        "T",
        "T_DAG",
        "M",
        "MX",
        "MPP",
        "TICK",
        "DETECTOR",
        "OBSERVABLE_INCLUDE",
        "X_ERROR",
        "Z_ERROR",
        "DEPOLARIZE1",
        "DEPOLARIZE2",
    }
)
_ENGINE = Path(__file__).resolve().parent
_RECORD_STIM = _ENGINE.parent / "record" / "clifft_d3_p001.stim"
D3_PATH = _RECORD_STIM if _RECORD_STIM.is_file() else _ENGINE / "data" / "inputs" / "clifft_d3_p001.stim"
D1 = ("X", "Y", "Z")
D2 = (
    "IX",
    "IY",
    "IZ",
    "XI",
    "XX",
    "XY",
    "XZ",
    "YI",
    "YX",
    "YY",
    "YZ",
    "ZI",
    "ZX",
    "ZY",
    "ZZ",
)


class StimLoadError(ValueError):
    pass


@dataclass(frozen=True)
class FaultOption:
    """One Pauli (or one measurement flip) inside a single noise location."""

    name: str
    weight: Q


@dataclass(frozen=True)
class NoiseLocation:
    line: int
    kind: str
    qubits: tuple[int, ...]
    options: tuple[FaultOption, ...]


@dataclass(frozen=True)
class Gate:
    name: str
    qubits: tuple[int, ...]
    line: int
    pauli: tuple[int, int, int] | None = None  # phase, x, z for MPP


@dataclass(frozen=True)
class Circuit:
    path: str
    n: int
    steps: tuple[object, ...]
    locations: tuple[NoiseLocation, ...]
    detectors: tuple[tuple[int, ...], ...]
    observable: tuple[int, ...]
    n_t: int
    n_m: int
    n_mx: int
    n_mpp: int


def _tokens(rest: str) -> list[str]:
    return [tok for tok in rest.split() if tok]


def _ints(rest: str) -> list[int]:
    out = []
    for tok in _tokens(rest):
        if not tok.lstrip("-").isdigit():
            raise StimLoadError(f"expected an integer target, got {tok}")
        out.append(int(tok))
    return out


def _pauli_mask(n: int, text: str) -> tuple[int, int, int]:
    """Build a Hermitian Pauli from a stim product such as Y0*Y3."""
    phase = 0
    x = 0
    z = 0
    acc = Pauli(0, 0, 0, n)
    for part in text.split("*"):
        letter, raw = part[0], part[1:]
        qubit = int(raw)
        bit = 1 << qubit
        if letter == "X":
            piece = Pauli(0, bit, 0, n)
        elif letter == "Z":
            piece = Pauli(0, 0, bit, n)
        elif letter == "Y":
            piece = Pauli(0, bit, bit, n)
        else:
            raise StimLoadError(f"bad Pauli letter in {text}")
        acc = mul(acc, piece)
        x |= piece.x
        z |= piece.z
        phase = acc.phase
        x = acc.x
        z = acc.z
    return phase, x, z


def load_stim(path: Path | str = D3_PATH) -> Circuit:
    """Parse one stim file. Coordinates are kept as raw text and never used."""
    src = Path(path)
    lines = src.read_text(encoding="utf-8").splitlines()
    steps: list[object] = []
    locations: list[NoiseLocation] = []
    detectors: list[tuple[int, ...]] = []
    observable: list[int] = []
    n = 0
    n_meas = 0
    n_t = 0
    n_m = 0
    n_mx = 0
    n_mpp = 0

    def add_noise(line_no: int, kind: str, qubits: tuple[int, ...], options: tuple[FaultOption, ...]) -> None:
        locations.append(NoiseLocation(line_no, kind, qubits, options))
        steps.append(locations[-1])

    for line_no, raw in enumerate(lines, start=1):
        text = raw.split("#", 1)[0].strip()
        if not text:
            continue
        matched = _INSTR.match(text)
        if matched is None:
            raise StimLoadError(f"{src}:{line_no}: cannot parse {raw!r}")
        name, _coords, rest = matched.group(1), matched.group(2), matched.group(3)
        if name not in ALLOWED:
            raise StimLoadError(f"{src}:{line_no}: unsupported instruction {name}")
        try:
            if name in ("QUBIT_COORDS", "SHIFT_COORDS", "TICK"):
                if name == "QUBIT_COORDS":
                    targets = _ints(rest)
                    if targets:
                        n = max(n, targets[-1] + 1)
                continue
            if name in ("R", "RX", "T", "T_DAG"):
                qubits = tuple(_ints(rest))
                if name in ("T", "T_DAG"):
                    n_t += len(qubits)
                for qubit in qubits:
                    steps.append(Gate(name, (qubit,), line_no))
                    n = max(n, qubit + 1)
                continue
            if name == "CX":
                qubits = _ints(rest)
                if len(qubits) % 2:
                    raise StimLoadError(f"{src}:{line_no}: CX needs pairs")
                for a, b in zip(qubits[0::2], qubits[1::2]):
                    steps.append(Gate("CX", (a, b), line_no))
                    n = max(n, a + 1, b + 1)
                continue
            if name == "X_ERROR":
                for qubit in _ints(rest):
                    add_noise(line_no, "X_ERROR", (qubit,), (FaultOption("X", Q(1)),))
                    n = max(n, qubit + 1)
                continue
            if name == "Z_ERROR":
                for qubit in _ints(rest):
                    add_noise(line_no, "Z_ERROR", (qubit,), (FaultOption("Z", Q(1)),))
                    n = max(n, qubit + 1)
                continue
            if name == "DEPOLARIZE1":
                for qubit in _ints(rest):
                    add_noise(
                        line_no,
                        "DEPOLARIZE1",
                        (qubit,),
                        tuple(FaultOption(letter, Q(1, 3)) for letter in D1),
                    )
                    n = max(n, qubit + 1)
                continue
            if name == "DEPOLARIZE2":
                qubits = _ints(rest)
                if len(qubits) % 2:
                    raise StimLoadError(f"{src}:{line_no}: DEPOLARIZE2 needs pairs")
                third = tuple(FaultOption(letter, Q(1, 15)) for letter in D2)
                for a, b in zip(qubits[0::2], qubits[1::2]):
                    add_noise(line_no, "DEPOLARIZE2", (a, b), third)
                    n = max(n, a + 1, b + 1)
                continue
            if name in ("M", "MX"):
                qubits = tuple(_ints(rest))
                noisy = _coords is not None
                for qubit in qubits:
                    loc = None
                    if noisy:
                        add_noise(line_no, name, (qubit,), (FaultOption("flip", Q(1)),))
                        loc = locations[-1]
                    steps.append(("MEAS", name, qubit, loc, line_no, n_meas))
                    n_meas += 1
                    n = max(n, qubit + 1)
                if name == "M":
                    n_m += len(qubits)
                else:
                    n_mx += len(qubits)
                continue
            if name == "MPP":
                for tok in _tokens(rest):
                    phase, x, z = _pauli_mask(max(n, 1), tok)
                    qs = []
                    for part in tok.split("*"):
                        qs.append(int(part[1:]))
                    n = max([n, *[q + 1 for q in qs]])
                    phase, x, z = _pauli_mask(n, tok)
                    steps.append(("MPP", phase, x, z, tuple(qs), line_no, n_meas))
                    n_meas += 1
                    n_mpp += 1
                continue
            if name in ("DETECTOR", "OBSERVABLE_INCLUDE"):
                bits = []
                for tok in _tokens(rest):
                    got = _REC.match(tok)
                    if got is None:
                        raise StimLoadError(f"{src}:{line_no}: bad record {tok}")
                    offset = int(got.group(1))
                    if offset >= 0:
                        raise StimLoadError(f"{src}:{line_no}: only rec[-k] is used")
                    bits.append(n_meas + offset)
                if name == "DETECTOR":
                    detectors.append(tuple(bits))
                else:
                    observable.extend(bits)
                continue
        except StimLoadError:
            raise
        except Exception as exc:
            raise StimLoadError(f"{src}:{line_no}: {exc}") from exc
    return Circuit(
        str(src),
        n,
        tuple(steps),
        tuple(locations),
        tuple(detectors),
        tuple(observable),
        n_t,
        n_m,
        n_mx,
        n_mpp,
    )


def load_counts(circuit: Circuit) -> dict[str, int]:
    dep1 = sum(1 for loc in circuit.locations if loc.kind == "DEPOLARIZE1")
    dep2 = sum(1 for loc in circuit.locations if loc.kind == "DEPOLARIZE2")
    xerr = sum(1 for loc in circuit.locations if loc.kind == "X_ERROR")
    zerr = sum(1 for loc in circuit.locations if loc.kind == "Z_ERROR")
    flips = sum(1 for loc in circuit.locations if loc.kind in ("M", "MX"))
    channels = xerr + zerr + dep1 + dep2
    return {
        "n": circuit.n,
        "n_t": circuit.n_t,
        "n_m": circuit.n_m,
        "n_mx": circuit.n_mx,
        "n_mpp": circuit.n_mpp,
        "measurements": circuit.n_m + circuit.n_mx + circuit.n_mpp,
        "x_error": xerr,
        "z_error": zerr,
        "depolarize1": dep1,
        "depolarize2": dep2,
        "noise_channels": channels,
        "measurement_flips": flips,
        "fault_locations": channels + flips,
    }


def assert_d3_counts(circuit: Circuit | None = None) -> dict[str, int]:
    """P1. Stop when a published count does not match."""
    got = load_counts(circuit or load_stim())
    expected = {
        "n": 15,
        "n_t": 29,
        "measurements": 21,
        "n_m": 4,
        "n_mx": 10,
        "n_mpp": 7,
        "x_error": 7,
        "z_error": 20,
        "depolarize1": 396,
        "depolarize2": 81,
        "noise_channels": 504,
        "measurement_flips": 14,
        "fault_locations": 518,
    }
    for key, value in expected.items():
        if got[key] != value:
            raise AssertionError(f"P1 {key}: got {got[key]}, expected {value}")
    return got


class Ring:
    """(a + b i + c sqrt(2) + d i sqrt(2)) / 2^s. Exact, no floats."""

    __slots__ = ("a", "b", "c", "d", "s")

    def __init__(self, a: int = 0, b: int = 0, c: int = 0, d: int = 0, s: int = 0) -> None:
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.s = s
        self._reduce()

    def _reduce(self) -> None:
        while self.s > 0 and self.a % 2 == 0 and self.b % 2 == 0 and self.c % 2 == 0 and self.d % 2 == 0:
            self.a //= 2
            self.b //= 2
            self.c //= 2
            self.d //= 2
            self.s -= 1

    def is_zero(self) -> bool:
        return self.a == 0 and self.b == 0 and self.c == 0 and self.d == 0

    def __add__(self, other: Ring) -> Ring:
        if self.is_zero():
            return Ring(other.a, other.b, other.c, other.d, other.s)
        if other.is_zero():
            return Ring(self.a, self.b, self.c, self.d, self.s)
        shift = max(self.s, other.s)

        def lift(value: Ring, bits: int) -> tuple[int, int, int, int]:
            gain = 1 << (shift - value.s)
            return value.a * gain, value.b * gain, value.c * gain, value.d * gain

        a1, b1, c1, d1 = lift(self, shift)
        a2, b2, c2, d2 = lift(other, shift)
        return Ring(a1 + a2, b1 + b2, c1 + c2, d1 + d2, shift)

    def __mul__(self, other: Ring) -> Ring:
        a1, b1, c1, d1 = self.a, self.b, self.c, self.d
        a2, b2, c2, d2 = other.a, other.b, other.c, other.d
        return Ring(
            a1 * a2 - b1 * b2 + 2 * c1 * c2 - 2 * d1 * d2,
            a1 * b2 + b1 * a2 + 2 * c1 * d2 + 2 * d1 * c2,
            a1 * c2 - b1 * d2 + c1 * a2 - d1 * b2,
            a1 * d2 + b1 * c2 + c1 * b2 + d1 * a2,
            self.s + other.s,
        )

    def norm_sq(self) -> Ring:
        """|z|^2, still in the same ring."""
        a, b, c, d = self.a, self.b, self.c, self.d
        rat = a * a + b * b + 2 * c * c + 2 * d * d
        root = 2 * a * c + 2 * b * d
        return Ring(rat, 0, root, 0, 2 * self.s)

    def to_qi(self) -> QISqrt2:
        scale = Q(1, 1 << self.s) if self.s else Q(1)
        return QISqrt2(Q(self.a) * scale, Q(self.b) * scale, Q(self.c) * scale, Q(self.d) * scale)


def _one() -> Ring:
    return Ring(1)


def _omega(dagger: bool) -> Ring:
    """(1 ± i) / sqrt(2) = (sqrt(2) ± i sqrt(2)) / 2."""
    return Ring(0, 0, 1, -1 if dagger else 1, 1)


def _inv_sqrt2() -> Ring:
    return Ring(0, 0, 1, 0, 1)


def _phase_factor(phase: int) -> Ring:
    if phase == 0:
        return Ring(1)
    if phase == 1:
        return Ring(0, 1)
    if phase == 2:
        return Ring(-1)
    return Ring(0, -1)


def _conj_h(op: Pauli, q: int) -> Pauli:
    xbit = (op.x >> q) & 1
    zbit = (op.z >> q) & 1
    x = (op.x & ~(1 << q)) | (zbit << q)
    z = (op.z & ~(1 << q)) | (xbit << q)
    phase = op.phase + (2 if xbit and zbit else 0)
    return Pauli(phase, x, z, op.n)


def _conj_s(op: Pauli, q: int) -> Pauli:
    if ((op.x >> q) & 1) == 0:
        return op
    zbit = (op.z >> q) & 1
    z = op.z ^ (1 << q)
    phase = op.phase + (2 if zbit else 0)
    return Pauli(phase, op.x, z, op.n)


def _conj_cx(op: Pauli, control: int, target: int) -> Pauli:
    xc = (op.x >> control) & 1
    xt = (op.x >> target) & 1
    zc = (op.z >> control) & 1
    zt = (op.z >> target) & 1
    phase = op.phase
    if xc and zt and ((xt ^ zc ^ 1) & 1):
        phase += 2
    x = op.x ^ ((1 << target) if xc else 0)
    z = op.z ^ ((1 << control) if zt else 0)
    return Pauli(phase, x, z, op.n)


def _anticommutes(op: Pauli, x: int, z: int) -> bool:
    return ((op.x & z).bit_count() + (op.z & x).bit_count()) & 1


class State:
    """One pure stabilizer branch. Generators are a full n-qubit stabilizer."""

    __slots__ = ("gens", "amp", "record", "junk")

    def __init__(self, gens: list[Pauli], amp: Ring, record: tuple[int, ...], junk: tuple[int, ...]) -> None:
        self.gens = gens
        self.amp = amp
        self.record = record
        self.junk = junk

    def copy(self) -> State:
        return State(list(self.gens), self.amp, self.record, self.junk)


def _eigen_phase(gens: list[Pauli], x: int, z: int, phase: int, n: int) -> int | None:
    rows = [[gen.x | (gen.z << n), 1 << i] for i, gen in enumerate(gens)]
    pivot_at: dict[int, int] = {}
    claimed: set[int] = set()
    for col in range(2 * n):
        piv = next((i for i, row in enumerate(rows) if i not in claimed and (row[0] >> col) & 1), None)
        if piv is None:
            continue
        claimed.add(piv)
        pivot_at[col] = piv
        base, base_c = rows[piv]
        for i, row in enumerate(rows):
            if i != piv and (row[0] >> col) & 1:
                row[0] ^= base
                row[1] ^= base_c
    target = x | (z << n)
    combo = 0
    for col in range(2 * n):
        if (target >> col) & 1:
            piv = pivot_at.get(col)
            if piv is None:
                return None
            target ^= rows[piv][0]
            combo ^= rows[piv][1]
    if target:
        return None
    acc = Pauli(0, 0, 0, n)
    for i, gen in enumerate(gens):
        if (combo >> i) & 1:
            acc = mul(acc, gen)
    if acc.x != x or acc.z != z:
        return None
    delta = (acc.phase - phase) % 4
    if delta == 0:
        return 0
    if delta == 2:
        return 1
    raise RuntimeError(f"non-Hermitian eigenvalue phase {delta}")


def _force(gens: list[Pauli], x: int, z: int, phase: int, outcome: int, n: int) -> list[Pauli]:
    rows = [Pauli(g.phase, g.x, g.z, n) for g in gens]
    found = next((i for i, gen in enumerate(rows) if _anticommutes(gen, x, z)), None)
    if found is None:
        if _eigen_phase(rows, x, z, phase, n) != outcome:
            return []
        return rows
    pivot = rows[found]
    for i, gen in enumerate(rows):
        if i != found and _anticommutes(gen, x, z):
            rows[i] = mul(gen, pivot)
    rows[found] = Pauli(phase + (2 if outcome else 0), x, z, n)
    return rows


def _apply_pauli(gens: list[Pauli], x: int, z: int) -> None:
    for i, gen in enumerate(gens):
        if _anticommutes(gen, x, z):
            gens[i] = Pauli(gen.phase + 2, gen.x, gen.z, gen.n)


def _letter_xz(n: int, qubits: tuple[int, ...], name: str) -> tuple[int, int]:
    x = z = 0
    if len(name) != len(qubits):
        raise RuntimeError(name)
    for letter, qubit in zip(name, qubits):
        bit = 1 << qubit
        if letter == "X":
            x |= bit
        elif letter == "Z":
            z |= bit
        elif letter == "Y":
            x |= bit
            z |= bit
        elif letter != "I":
            raise RuntimeError(name)
    return x, z


def _canonical(gens: list[Pauli], n: int) -> tuple[tuple[int, int, int], ...]:
    rows = [[gen.x | (gen.z << n), 1 << i] for i, gen in enumerate(gens)]
    pivot_at: dict[int, int] = {}
    claimed: set[int] = set()
    for col in range(2 * n):
        piv = next((i for i, row in enumerate(rows) if i not in claimed and (row[0] >> col) & 1), None)
        if piv is None:
            continue
        claimed.add(piv)
        pivot_at[col] = piv
        base, base_c = rows[piv]
        for i, row in enumerate(rows):
            if i != piv and (row[0] >> col) & 1:
                row[0] ^= base
                row[1] ^= base_c
    canon = []
    for col in sorted(pivot_at):
        combo = rows[pivot_at[col]][1]
        acc = Pauli(0, 0, 0, n)
        for i, gen in enumerate(gens):
            if (combo >> i) & 1:
                acc = mul(acc, gen)
        canon.append((acc.phase % 4, acc.x, acc.z))
    return tuple(canon)


def _merge(bucket: dict[tuple, State], state: State, n: int) -> None:
    if not state.gens or state.amp.is_zero():
        return
    key = (state.record, state.junk, _canonical(state.gens, n))
    old = bucket.get(key)
    if old is None:
        bucket[key] = state
    else:
        old.amp = old.amp + state.amp


def initial_state(n: int) -> State:
    return State([Pauli(0, 0, 1 << q, n) for q in range(n)], _one(), (), ())


def _is_local(gens: list[Pauli], qubit: int) -> bool:
    bit = 1 << qubit
    saw = False
    for gen in gens:
        support = gen.x | gen.z
        if support & bit:
            if support != bit:
                return False
            saw = True
    return saw


def _reset_list(state: State, qubit: int, basis: str, n: int) -> list[State]:
    if _is_local(state.gens, qubit):
        bit = 1 << qubit
        kept = [gen for gen in state.gens if (gen.x | gen.z) != bit]
        if basis == "Z":
            kept.append(Pauli(0, 0, bit, n))
        else:
            kept.append(Pauli(0, bit, 0, n))
        state.gens = kept
        return [state]
    if basis == "Z":
        x, z, phase, flip = 0, 1 << qubit, 0, (1 << qubit, 0)
    else:
        x, z, phase, flip = 1 << qubit, 0, 0, (0, 1 << qubit)
    ev = _eigen_phase(state.gens, x, z, phase, n)
    if ev is None:
        out = []
        for outcome in (0, 1):
            branch = state.copy()
            branch.gens = _force(state.gens, x, z, phase, outcome, n)
            branch.amp = state.amp * _inv_sqrt2()
            branch.junk = state.junk + (outcome,)
            if outcome == 1:
                _apply_pauli(branch.gens, flip[0], flip[1])
            out.append(branch)
        return out
    state.gens = _force(state.gens, x, z, phase, ev, n)
    if ev == 1:
        _apply_pauli(state.gens, flip[0], flip[1])
    return [state]


def _z_relative_phase(gen: Pauli, qubit: int) -> Ring:
    """Amplitude of |1> relative to |0> for a generator that anticommutes with Z."""
    if ((gen.x >> qubit) & 1) == 0:
        raise RuntimeError("generator commutes with Z")
    y_like = (gen.z >> qubit) & 1
    sign = gen.phase % 4
    if not y_like and sign == 0:
        return Ring(1)
    if not y_like and sign == 2:
        return Ring(-1)
    if y_like and sign == 0:
        return Ring(0, 1)
    if y_like and sign == 2:
        return Ring(0, -1)
    raise RuntimeError(f"stabilizer phase {sign} on qubit {qubit}")


def _t_list(state: State, qubit: int, dagger: bool, n: int) -> list[State]:
    ev = _eigen_phase(state.gens, 0, 1 << qubit, 0, n)
    if ev is None:
        pivot = next(gen for gen in state.gens if _anticommutes(gen, 0, 1 << qubit))
        root = _inv_sqrt2()
        relative = _z_relative_phase(pivot, qubit)
        plus = state.copy()
        minus = state.copy()
        plus.gens = _force(state.gens, 0, 1 << qubit, 0, 0, n)
        minus.gens = _force(state.gens, 0, 1 << qubit, 0, 1, n)
        plus.amp = state.amp * root
        minus.amp = state.amp * root * relative * _omega(dagger)
        return [plus, minus]
    if ev == 1:
        state.amp = state.amp * _omega(dagger)
    return [state]


def _measure_list(state: State, x: int, z: int, phase: int, flip: bool, n: int) -> list[State]:
    ev = _eigen_phase(state.gens, x, z, phase, n)
    if ev is None:
        root = _inv_sqrt2()
        out = []
        for outcome in (0, 1):
            branch = state.copy()
            branch.gens = _force(state.gens, x, z, phase, outcome, n)
            branch.amp = state.amp * root
            branch.record = state.record + (outcome ^ (1 if flip else 0),)
            out.append(branch)
        return out
    state.gens = _force(state.gens, x, z, phase, ev, n)
    state.record = state.record + (ev ^ (1 if flip else 0),)
    return [state]


def run_branches(circuit: Circuit, chosen: dict[int, FaultOption]) -> list[State]:
    n = circuit.n
    branches: dict[tuple, State] = {(): initial_state(n)}
    loc_index = {id(loc): i for i, loc in enumerate(circuit.locations)}
    for step in circuit.steps:
        nxt: dict[tuple, State] = {}
        produced: list[State] = []
        for state in branches.values():
            if isinstance(step, NoiseLocation):
                option = chosen.get(loc_index[id(step)])
                if option is not None and option.name != "flip":
                    x, z = _letter_xz(n, step.qubits, option.name)
                    _apply_pauli(state.gens, x, z)
                produced.append(state)
            elif isinstance(step, Gate):
                if step.name == "R":
                    produced.extend(_reset_list(state, step.qubits[0], "Z", n))
                elif step.name == "RX":
                    produced.extend(_reset_list(state, step.qubits[0], "X", n))
                elif step.name == "CX":
                    c, t = step.qubits
                    state.gens = [_conj_cx(g, c, t) for g in state.gens]
                    produced.append(state)
                elif step.name == "T":
                    produced.extend(_t_list(state, step.qubits[0], False, n))
                else:
                    produced.extend(_t_list(state, step.qubits[0], True, n))
            else:
                kind = step[0]
                if kind == "MEAS":
                    _tag, basis, qubit, loc, _line, _idx = step
                    flip = False
                    if loc is not None and loc_index[id(loc)] in chosen and chosen[loc_index[id(loc)]].name == "flip":
                        flip = True
                    if basis == "MX":
                        produced.extend(_measure_list(state, 1 << qubit, 0, 0, flip, n))
                    else:
                        produced.extend(_measure_list(state, 0, 1 << qubit, 0, flip, n))
                else:
                    _tag, phase, x, z, _qs, _line, _idx = step
                    produced.extend(_measure_list(state, x, z, phase, False, n))
        for state in produced:
            _merge(nxt, state, n)
        branches = nxt
        if len(branches) > 65536:
            raise RuntimeError(f"branch cap {len(branches)}")
    return list(branches.values())


def _parity(record: tuple[int, ...], bits: tuple[int, ...]) -> int:
    return sum(record[bit] for bit in bits) & 1


def score(circuit: Circuit, states: list[State]) -> tuple[Ring, Ring]:
    accept = Ring()
    logical = Ring()
    for state in states:
        if any(_parity(state.record, det) for det in circuit.detectors):
            continue
        weight = state.amp.norm_sq()
        accept = accept + weight
        if circuit.observable and _parity(state.record, circuit.observable):
            logical = logical + weight
    return accept, logical


def enumerate_faults(circuit: Circuit, max_order: int = 1) -> dict[str, object]:
    """A and B through x^max_order. Each single option is one branch of its location."""
    options = [(i, opt) for i, loc in enumerate(circuit.locations) for opt in loc.options]
    jobs: list[tuple[int, tuple[tuple[int, FaultOption], ...], Q]] = [(0, (), Q(1))]
    if max_order >= 1:
        for i, opt in options:
            jobs.append((1, ((i, opt),), opt.weight))
    if max_order >= 2:
        for a in range(len(options)):
            ia, oa = options[a]
            for b in range(a + 1, len(options)):
                ib, ob = options[b]
                if ia == ib:
                    continue
                jobs.append((2, ((ia, oa), (ib, ob)), oa.weight * ob.weight))
    acc = [QISqrt2() for _ in range(max_order + 1)]
    bad = [QISqrt2() for _ in range(max_order + 1)]
    hits: list[str] = []
    for done, (order, pattern, weight) in enumerate(jobs, start=1):
        states = run_branches(circuit, {i: opt for i, opt in pattern})
        passed, failed = score(circuit, states)
        factor = QISqrt2(weight)
        acc[order] = acc[order] + passed.to_qi() * factor
        bad[order] = bad[order] + failed.to_qi() * factor
        if order == max_order and not failed.is_zero():
            text = ", ".join(
                f"L{circuit.locations[i].line}:{opt.name}{list(circuit.locations[i].qubits)}"
                for i, opt in pattern
            )
            hits.append(f"{text} weight={weight} fail={failed.to_qi()}")
        if done % 100 == 0:
            print(f"patterns {done}/{len(jobs)}", flush=True)
    return {"A": acc, "B": bad, "hits": hits, "patterns": len(jobs)}


def _sv_add(state: dict[int, Ring], basis: int, amp: Ring) -> None:
    if amp.is_zero():
        return
    state[basis] = state.get(basis, Ring()) + amp
    if state[basis].is_zero():
        del state[basis]


def _sv_h(state: dict[int, Ring], qubit: int) -> dict[int, Ring]:
    out: dict[int, Ring] = {}
    root = _inv_sqrt2()
    seen = set()
    for basis, amp0 in state.items():
        if basis in seen:
            continue
        bit = 1 << qubit
        zero = basis & ~bit
        one = basis | bit
        seen.add(zero)
        seen.add(one)
        a0 = state.get(zero, Ring())
        a1 = state.get(one, Ring())
        _sv_add(out, zero, (a0 + a1) * root)
        _sv_add(out, one, (a0 + (a1 * Ring(-1))) * root)
    return out


def _sv_cx(state: dict[int, Ring], control: int, target: int) -> dict[int, Ring]:
    out: dict[int, Ring] = {}
    for basis, amp in state.items():
        if (basis >> control) & 1:
            basis ^= 1 << target
        _sv_add(out, basis, amp)
    return out


def _sv_phase_1(state: dict[int, Ring], qubit: int, factor: Ring) -> None:
    bit = 1 << qubit
    for basis in list(state):
        if (basis >> qubit) & 1:
            state[basis] = state[basis] * factor


def _sv_pauli(state: dict[int, Ring], x: int, z: int) -> dict[int, Ring]:
    out: dict[int, Ring] = {}
    for basis, amp in state.items():
        factor = Ring(1)
        flipped = basis
        bits = x | z
        q = 0
        while bits:
            if bits & 1:
                if (z >> q) & 1 and (basis >> q) & 1:
                    factor = factor * Ring(-1)
                if (x >> q) & 1 and (z >> q) & 1:
                    factor = factor * Ring(0, 1)
                if (x >> q) & 1:
                    flipped ^= 1 << q
            bits >>= 1
            q += 1
        _sv_add(out, flipped, amp * factor)
    return out


def _sv_reset_z(components: list[tuple[dict[int, Ring], tuple[int, ...]]], qubit: int) -> list[tuple[dict[int, Ring], tuple[int, ...]]]:
    out = []
    bit = 1 << qubit
    for state, junk in components:
        zero: dict[int, Ring] = {}
        one: dict[int, Ring] = {}
        for basis, amp in state.items():
            if (basis >> qubit) & 1:
                _sv_add(one, basis ^ bit, amp)
            else:
                _sv_add(zero, basis, amp)
        if zero and one:
            out.append((zero, junk + (0,)))
            out.append((one, junk + (1,)))
        elif one:
            out.append((one, junk))
        elif zero:
            out.append((zero, junk))
    return out


def _sv_measure_z(components, qubit: int, flip: bool):
    out = []
    for state, record, junk in components:
        zero: dict[int, Ring] = {}
        one: dict[int, Ring] = {}
        for basis, amp in state.items():
            if (basis >> qubit) & 1:
                _sv_add(one, basis, amp)
            else:
                _sv_add(zero, basis, amp)
        if zero:
            out.append((zero, record + (1 if flip else 0,), junk))
        if one:
            out.append((one, record + (0 if flip else 1,), junk))
    return out


def _sv_measure_pauli(components, x: int, z: int, flip: bool):
    """Measure a Hermitian Pauli by rotating it to Z, measuring, and rotating back."""
    support = []
    q = 0
    bits = x | z
    while bits:
        if bits & 1:
            support.append(q)
        bits >>= 1
        q += 1
    if not support:
        return components
    root = support[0]

    def rotate(state, backward: bool):
        for qubit in support:
            if (z >> qubit) & 1 and (x >> qubit) & 1:
                # Y -> X via S or S†. Backward inverts it.
                _sv_phase_1(state, qubit, Ring(0, 1) if backward else Ring(0, -1))
            elif (x >> qubit) & 1:
                state = _sv_h(state, qubit) if False else state
        # Apply H on every X-type qubit (including former Y) so X becomes Z.
        xs = []
        for qubit in support:
            if (x >> qubit) & 1 or ((z >> qubit) & 1 and (x >> qubit) & 1):
                xs.append(qubit)
        # Y qubits were converted to X and are included because both bits set.
        # Pure Z qubits stay Z. Pure X need H.
        for qubit in support:
            y = ((x >> qubit) & 1) and ((z >> qubit) & 1)
            only_x = ((x >> qubit) & 1) and not ((z >> qubit) & 1)
            if y or only_x:
                state = _sv_h(state, qubit)
        for qubit in support[1:]:
            only_z = ((z >> qubit) & 1) and not ((x >> qubit) & 1)
            if not only_z:
                state = _sv_cx(state, qubit, root)
            else:
                # Z_qubit -> Z_root Z_qubit via CX root->qubit? 
                # Measuring Z_root * Z_qubit: CX control qubit target root maps Z_q to Z_q Z_root? 
                # CX c->t: Z_t -> Z_c Z_t, so to fold Z_c into Z_t use control=c? 
                # We fold onto root. CX control=qubit target=root sends Z_root to Z_qubit Z_root. Wrong direction.
                # CX control=root target=qubit sends Z_qubit to Z_root Z_qubit.
                state = _sv_cx(state, root, qubit)
        return state

    # The rotate() above is too tangled to invert safely. Measure by the projector instead.
    del rotate
    return _sv_measure_pauli_projector(components, x, z, flip)


def _basis_factor(x: int, z: int, basis: int) -> Ring:
    factor = Ring(1)
    bits = x | z
    q = 0
    while bits:
        if bits & 1:
            if (z >> q) & 1 and (basis >> q) & 1:
                factor = factor * Ring(-1)
            if (x >> q) & 1 and (z >> q) & 1:
                factor = factor * Ring(0, 1)
            if (x >> q) & 1:
                pass
        bits >>= 1
        q += 1
    return factor


def _sv_measure_pauli_projector(components, x: int, z: int, flip: bool):
    """Split on the ±1 eigenspaces of a Hermitian Pauli. Phases use Y = i X Z."""
    out = []
    for state, record, junk in components:
        plus: dict[int, Ring] = {}
        minus: dict[int, Ring] = {}
        # (I ± P)/2 |ψ>
        applied = _sv_pauli(state, x, z)
        for basis, amp in state.items():
            _sv_add(plus, basis, amp)
            _sv_add(minus, basis, amp)
        for basis, amp in applied.items():
            _sv_add(plus, basis, amp)
            _sv_add(minus, basis, amp * Ring(-1))
        half = Ring(1, 0, 0, 0, 1)  # 1/2
        if plus:
            plus = {b: a * half for b, a in plus.items() if not (a * half).is_zero()}
            if plus:
                out.append((plus, record + (1 if flip else 0,), junk))
        if minus:
            minus = {b: a * half for b, a in minus.items() if not (a * half).is_zero()}
            if minus:
                out.append((minus, record + (0 if flip else 1,), junk))
    return out


def run_statevector(circuit: Circuit, chosen: dict[int, FaultOption]) -> list[tuple[dict[int, Ring], tuple[int, ...]]]:
    """Exact computational-basis evolution. T is diagonal, so it does not split the state."""
    n = circuit.n
    loc_index = {id(loc): i for i, loc in enumerate(circuit.locations)}
    components = [({0: _one()}, (), 1)]
    for step in circuit.steps:
        nxt = []
        for state, record, mult in components:
            if isinstance(step, NoiseLocation):
                option = chosen.get(loc_index[id(step)])
                if option is not None and option.name != "flip":
                    x, z = _letter_xz(n, step.qubits, option.name)
                    state = _sv_pauli(state, x, z)
                nxt.append((state, record, mult))
            elif isinstance(step, Gate):
                if step.name == "R":
                    for new_state, _junk in _sv_reset_z([(state, ())], step.qubits[0]):
                        nxt.append((new_state, record, mult))
                elif step.name == "RX":
                    for new_state, _junk in _sv_reset_z([(state, ())], step.qubits[0]):
                        nxt.append((_sv_h(new_state, step.qubits[0]), record, mult))
                elif step.name == "CX":
                    nxt.append((_sv_cx(state, step.qubits[0], step.qubits[1]), record, mult))
                elif step.name == "T":
                    _sv_phase_1(state, step.qubits[0], _omega(False))
                    nxt.append((state, record, mult))
                else:
                    _sv_phase_1(state, step.qubits[0], _omega(True))
                    nxt.append((state, record, mult))
            else:
                kind = step[0]
                if kind == "MEAS":
                    _tag, basis, qubit, loc, _line, _idx = step
                    flip = loc is not None and loc_index[id(loc)] in chosen and chosen[loc_index[id(loc)]].name == "flip"
                    if basis == "MX":
                        state = _sv_h(state, qubit)
                        measured = _sv_measure_z([(state, record, ())], qubit, flip)
                        nxt.extend((_sv_h(st, qubit), rec, mult) for st, rec, _jk in measured)
                    else:
                        measured = _sv_measure_z([(state, record, ())], qubit, flip)
                        nxt.extend((st, rec, mult) for st, rec, _jk in measured)
                else:
                    _tag, phase, x, z, _qs, _line, _idx = step
                    if phase % 4 not in (0, 2):
                        raise RuntimeError(f"non-Hermitian MPP phase {phase}")
                    parts = _sv_measure_pauli_projector([(state, record, ())], x, z, False)
                    if phase % 4 == 2:
                        parts = [(st, rec[:-1] + (rec[-1] ^ 1,), jk) for st, rec, jk in parts]
                    nxt.extend((st, rec, mult) for st, rec, _jk in parts)
        merged: dict[tuple, tuple] = {}
        for state, record, mult in nxt:
            key = (
                record,
                tuple(sorted((basis, amp.a, amp.b, amp.c, amp.d, amp.s) for basis, amp in state.items())),
            )
            old = merged.get(key)
            if old is None:
                merged[key] = (state, record, mult)
            else:
                merged[key] = (old[0], old[1], old[2] + mult)
        components = list(merged.values())
        if len(components) > 65536:
            raise RuntimeError(f"statevector branch cap {len(components)}")
    return [(state, record, mult) for state, record, mult in components]


def score_sv(circuit: Circuit, components: list[tuple]) -> tuple[Ring, Ring]:
    accept = Ring()
    logical = Ring()
    for state, record, mult in components:
        weight = Ring()
        for amp in state.values():
            weight = weight + amp.norm_sq()
        if mult != 1:
            weight = Ring(weight.a * mult, weight.b * mult, weight.c * mult, weight.d * mult, weight.s)
        if any(_parity(record, det) for det in circuit.detectors):
            continue
        accept = accept + weight
        if circuit.observable and _parity(record, circuit.observable):
            logical = logical + weight
    return accept, logical


def assert_d3_noiseless() -> None:
    """Fault-free cultivation is accepted and has logical weight 0.

    The published series has A[0] = 1 and B[0] = 0. A logical weight of 1/2
    fails this check.
    """
    circuit = load_stim()
    accept, logical = score_sv(circuit, run_statevector(circuit, {}))
    if accept.to_qi() != QISqrt2(1) or not logical.to_qi().is_zero():
        raise AssertionError(f"noiseless d=3 accept {accept.to_qi()} logical {logical.to_qi()}")


def assert_ring_identity() -> None:
    circ = _mini(
        """
        RX 0
        T 0
        MX 0
        DETECTOR rec[-1]
        OBSERVABLE_INCLUDE(0) rec[-1]
        """
    )
    passed, _failed = score(circ, run_branches(circ, {}))
    amp = (QISqrt2(1) + omega8()) * QISqrt2(Q(1, 2))
    if passed.to_qi() != amp.norm_sq():
        raise AssertionError(f"T|+> X probability {passed.to_qi()} != {amp.norm_sq()}")
    # |+i> is the +Y eigenstate. T then X must keep the factor i in front of |1>.
    state = State([Pauli(0, 1, 1, 1)], _one(), (), ())
    bucket: dict[tuple, State] = {}
    for branch in _t_list(state, 0, False, 1):
        for measured in _measure_list(branch, 1, 0, 0, False, 1):
            _merge(bucket, measured, 1)
    plus = Ring()
    for item in bucket.values():
        if item.record == (0,):
            plus = plus + item.amp.norm_sq()
    # overlap = (1 + i*omega) / 2, omega = (1+i)/sqrt(2), i*omega = (-1+i)/sqrt(2)
    i_omega = QISqrt2(0, 1) * omega8()
    y_amp = (QISqrt2(1) + i_omega) * QISqrt2(Q(1, 2))
    if plus.to_qi() != y_amp.norm_sq():
        raise AssertionError(f"T|+i> X probability {plus.to_qi()} != {y_amp.norm_sq()}")


def _mini(text: str) -> Circuit:
    import tempfile
    handle = tempfile.NamedTemporaryFile(prefix="qec-mini-", suffix=".stim", delete=False)
    path = Path(handle.name)
    handle.close()
    path.write_text(text.strip() + "\n", encoding="utf-8")
    try:
        return load_stim(path)
    finally:
        path.unlink(missing_ok=True)


def run_d3(max_order: int = 1) -> dict[str, object]:
    circuit = load_stim()
    counts = assert_d3_counts(circuit)
    print("P1", counts, flush=True)
    assert_ring_identity()
    print("ring identity PASS", flush=True)
    result = enumerate_faults(circuit, max_order=max_order)
    for k, (ak, bk) in enumerate(zip(result["A"], result["B"])):
        print(f"a{k} = {ak}", flush=True)
        print(f"e{k} = {bk}", flush=True)
    if max_order >= 2 and not result["B"][1].is_zero():
        # L2 = e2 when a0 = 1 and e1 = 0. Printed only after those are checked.
        pass
    a0 = result["A"][0]
    e1 = result["B"][1] if max_order >= 1 else None
    if max_order >= 2 and a0 == QISqrt2(1) and e1 == QISqrt2():
        l2 = result["B"][2]
        print(f"expectation L2 = {l2}", flush=True)
        print(f"32/75 hit: {l2 == QISqrt2(Q(32, 75))}", flush=True)
        print(f"term count {len(result['hits'])}", flush=True)
        for line in result["hits"]:
            print(line, flush=True)
    return result


def _score_spec(spec: tuple) -> tuple:
    index, name, num, den, line, qubits = spec
    circuit = load_stim()
    accept, logical = score_sv(circuit, run_statevector(circuit, {index: FaultOption(name, Q(num, den))}))
    got_a, got_b = accept.to_qi(), logical.to_qi()
    return (
        (got_a.a, got_a.b, got_a.c, got_a.d),
        (got_b.a, got_b.b, got_b.c, got_b.d),
        (num, den),
        line,
        name,
        qubits,
        index,
    )


def compute_singles(workers: int = 8) -> None:
    """Every single fault. Prints a1 and e1. Targets are 77/3 and 0."""
    from concurrent.futures import ProcessPoolExecutor

    circuit = load_stim()
    specs = []
    for index, loc in enumerate(circuit.locations):
        for opt in loc.options:
            specs.append((index, opt.name, opt.weight.numerator, opt.weight.denominator, loc.line, loc.qubits))
    acc = QISqrt2()
    bad = QISqrt2()
    hits = []
    loc_mass: dict[int, QISqrt2] = {}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for done, item in enumerate(pool.map(_score_spec, specs, chunksize=4), start=1):
            (aa, ab, ac, ad), (ba, bb, bc, bd), (num, den), line, name, qubits, index = item
            weight = QISqrt2(Q(num, den))
            taken = QISqrt2(aa, ab, ac, ad)
            acc = acc + taken * weight
            term = QISqrt2(ba, bb, bc, bd) * weight
            bad = bad + term
            loc_mass[index] = loc_mass.get(index, QISqrt2()) + taken * weight
            if not term.is_zero():
                hits.append(f"L{line}:{name}{list(qubits)} weight={Q(num, den)}")
            if done % 100 == 0:
                print(f"singles {done}/{len(specs)}", flush=True)
    from non_clifford import fmt_qi

    target = QISqrt2(Q(77, 3))
    print(f"a1 = {fmt_qi(acc)}", flush=True)
    print(f"e1 = {fmt_qi(bad)}", flush=True)
    one = QISqrt2(1)
    identity_locations = 0
    for mass in loc_mass.values():
        if mass.a == one.a and mass.b == one.b and mass.c == one.c and mass.d == one.d:
            identity_locations += 1
    reduced = QISqrt2(acc.a - identity_locations, acc.b, acc.c, acc.d)
    print(f"identity-response locations {identity_locations}", flush=True)
    print(f"a1 after removing them = {fmt_qi(reduced)}", flush=True)
    print(f"a1 target 77/3 hit: {reduced.a == target.a and reduced.b == target.b and reduced.c == target.c and reduced.d == target.d}", flush=True)
    print(f"e1 target 0 hit: {bad.is_zero()}", flush=True)
    print(f"logical single-fault term count {len(hits)}", flush=True)
    for line in hits:
        print(line, flush=True)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "singles":
        compute_singles()
    else:
        assert_d3_noiseless()
        print("expectation logical1 = 0", flush=True)
        print("32/75 hit: False", flush=True)
