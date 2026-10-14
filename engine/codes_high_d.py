"""High-distance stabilizer codes compiled into the staircase basis.

Distances for n <= 11 are the minimum Hamming weight of a non-trivial logical,
found by an exact weight scan over F_2. The distance-9 code is the
concatenation of the [[5,1,3]] code with itself. A [[3,1,3]] outer code does
not exist, so three 7-qubit blocks cannot multiply both distances by 3.
"""

from __future__ import annotations

from lift_n import Pauli, commutes, from_string, identity, mul, phase_name, product_subset, rank_mod2, scale, symplectic_vector, to_string
from lift_n import build_n_qubit_gammas
from presets import compile_fast_pauli_to_F, compile_pauli, format_word

FIVE_STABS = ("XZZXI", "IXZZX", "XIXZZ", "ZXIXZ")
FIVE_X = "XXXXX"
FIVE_Z = "ZZZZZ"
STEANE_STABS = ("IIIXXXX", "IXXIIXX", "XIXIXIX", "IIIZZZZ", "IZZIIZZ", "ZIZIZIZ")
EIGHT_STABS = ("XXXXXXXX", "ZZZZZZZZ", "IXIXYZYZ", "IXZYIXZY", "IYXZXZIY")
DODECA_STABS = (
    "XIIIIZZZIXX",
    "IXIIIZZYZIY",
    "IIXIIZZXXZI",
    "IIIXIZYIXXY",
    "IIIIXZXIZZX",
    "IIZIIXYIYZZ",
    "IZZIIIXZIXZ",
    "IZIIZIZXXIZ",
    "ZZZIZIIZXZX",
    "IZIZIIZIZXX",
)


def _ops(strings: tuple[str, ...] | list[str], phases: tuple[int, ...] | None = None) -> list[Pauli]:
    if phases is None:
        return [from_string(text) for text in strings]
    return [from_string(text, phase) for text, phase in zip(strings, phases, strict=True)]


def _group(generators: list[Pauli]) -> set[tuple[int, int, int]]:
    """All (phase, x, z) products of the generators."""
    found = {(0, 0, 0)}
    for gen in generators:
        extra = []
        for phase, x, z in found:
            prod = mul(Pauli(phase, x, z, gen.n), gen)
            extra.append((prod.phase, prod.x, prod.z))
        found.update(extra)
    return found


def _syndrome(op: Pauli, generators: list[Pauli]) -> int:
    bits = 0
    for i, gen in enumerate(generators):
        if not commutes(op, gen):
            bits |= 1 << i
    return bits


def _weight_scan(n: int, max_weight: int):
    """Yield (weight, x, z) for non-identity Paulis up to max_weight. Phase +1."""
    if max_weight > n:
        max_weight = n
    # Each qubit is I,X,Y,Z. Enumerate by assigning letters on a chosen support.
    from itertools import combinations

    letters = (1, 2, 3)  # X=x bit, Z=z bit, Y=both. Encoded below.
    for weight in range(1, max_weight + 1):
        for support in combinations(range(n), weight):
            # bit position: qubit index q=0 is leftmost = bit n-1-q? 
            # combinations range(n) I'll treat as bit index directly (bit 0 = rightmost).
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
                yield weight, x, z


def verify_code_distance(
    stabilizers: list[Pauli],
    logical_xs: list[Pauli],
    logical_zs: list[Pauli],
    scan_weight: int | None = None,
) -> dict[str, int | bool]:
    n = stabilizers[0].n
    k = len(logical_xs)
    if len(logical_zs) != k:
        raise ValueError("logical X and Z counts differ")
    for i, left in enumerate(stabilizers):
        for right in stabilizers[i + 1 :]:
            if not commutes(left, right):
                raise ValueError("stabilizers do not commute")
    rank = rank_mod2([symplectic_vector(op) for op in stabilizers], 2 * n)
    if rank != n - k:
        raise ValueError(f"stabilizer rank {rank} != {n - k}")
    for i, x_op in enumerate(logical_xs):
        for stab in stabilizers:
            if not commutes(x_op, stab) or not commutes(logical_zs[i], stab):
                raise ValueError("logical fails to commute with a stabilizer")
        for j, z_op in enumerate(logical_zs):
            should = i == j
            anti = not commutes(x_op, z_op)
            if anti != should:
                raise ValueError("logical commutation table is wrong")
        for j, other in enumerate(logical_xs):
            if i < j and not commutes(x_op, other):
                raise ValueError("logical X operators anticommute")
    group = _group(stabilizers)
    group_xz = {(x, z) for _phase, x, z in group}
    limit = n if scan_weight is None else scan_weight
    best = n + 1
    for weight, x, z in _weight_scan(n, limit):
        op = Pauli(0, x, z, n)
        if any(not commutes(op, stab) for stab in stabilizers):
            continue
        if (x, z) in group_xz:
            continue
        best = weight
        break
    if best == n + 1:
        # No logical at or below the scan. The given logicals still upper-bound d.
        best = min(op.weight() for op in logical_xs + logical_zs)
        exact = False
    else:
        exact = True
    return {"n": n, "k": k, "d": best, "rank": rank, "exact": exact, "t": (best - 1) // 2}


def _minimum_logicals(stabilizers: list[Pauli], k: int, scan: int) -> tuple[list[Pauli], list[Pauli]]:
    """Find k anticommuting logical pairs of small weight."""
    n = stabilizers[0].n
    group = {(x, z) for _p, x, z in _group(stabilizers)}
    logicals: list[Pauli] = []
    for _weight, x, z in _weight_scan(n, scan):
        if (x, z) in group:
            continue
        op = Pauli(0, x, z, n)
        if any(not commutes(op, stab) for stab in stabilizers):
            continue
        logicals.append(op)
    xs: list[Pauli] = []
    zs: list[Pauli] = []
    for left in logicals:
        if len(xs) == k:
            break
        for right in logicals:
            if commutes(left, right):
                continue
            if any(not commutes(left, op) or not commutes(right, op) for op in xs + zs):
                continue
            xs.append(left)
            zs.append(right)
            break
    if len(xs) != k:
        raise ValueError(f"found {len(xs)} logical pairs, wanted {k}")
    return xs, zs


def five_qubit() -> dict:
    stabs = _ops(FIVE_STABS)
    lx = _ops((FIVE_X,))
    lz = _ops((FIVE_Z,))
    report = verify_code_distance(stabs, lx, lz, scan_weight=5)
    return {"name": "perfect-513", "stabilizers": stabs, "logical_x": lx, "logical_z": lz, **report}


def steane() -> dict:
    stabs = _ops(STEANE_STABS)
    lx = _ops(("XXXXXXX",))
    lz = _ops(("ZZZZZZZ",))
    report = verify_code_distance(stabs, lx, lz, scan_weight=7)
    return {"name": "steane-713", "stabilizers": stabs, "logical_x": lx, "logical_z": lz, **report}


def subsystem_4x32() -> dict:
    stabs = _ops(("YZIIIII", "ZYIIIII", "IIXZZXI", "IIIXZZX", "IIXIXZZ", "IIZXIXZ"), (0, 2, 0, 0, 0, 0))
    lx = _ops(("IIXXXXX",))
    lz = _ops(("IIZZZZZ",))
    report = verify_code_distance(stabs, lx, lz, scan_weight=7)
    return {"name": "subsystem-4x32", "stabilizers": stabs, "logical_x": lx, "logical_z": lz, **report}


def clock_seat() -> dict:
    stabs = _ops(EIGHT_STABS)
    lx, lz = _minimum_logicals(stabs, 3, scan=4)
    report = verify_code_distance(stabs, lx, lz, scan_weight=4)
    return {"name": "clock-seat-833", "stabilizers": stabs, "logical_x": lx, "logical_z": lz, **report}


def extended_d5() -> dict:
    stabs = _ops(DODECA_STABS)
    lx, lz = _minimum_logicals(stabs, 1, scan=5)
    report = verify_code_distance(stabs, lx, lz, scan_weight=5)
    return {"name": "extended-d5", "stabilizers": stabs, "logical_x": lx, "logical_z": lz, **report}


def _lift_outer(outer: Pauli, block: int, inner_x: Pauli, inner_z: Pauli, blocks: int) -> Pauli:
    """Replace one outer qubit by the inner logical, and leave other blocks as identity."""
    n_in = inner_x.n
    acc = identity(n_in * blocks)
    for q in range(outer.n):
        letter = outer.letter(outer.n - 1 - q)
        if letter == "I":
            continue
        local = inner_x if letter == "X" else inner_z if letter == "Z" else mul(inner_x, inner_z)
        placed = _place_block(local, q, blocks)
        acc = mul(acc, placed)
    acc = scale(outer.phase, acc)
    return acc


def _place_block(local: Pauli, block: int, blocks: int) -> Pauli:
    n_in = local.n
    # Block 0 is the leftmost qubits. Bit 0 is the rightmost qubit of the full string.
    # Leftmost block occupies the highest bits.
    shift = (blocks - 1 - block) * n_in
    return Pauli(local.phase, local.x << shift, local.z << shift, n_in * blocks)


def clock_z24_d9() -> dict:
    """[[25,1,9]] by concatenating [[5,1,3]] with itself.

    A non-trivial logical must put a non-trivial inner logical on at least
    d_outer = 3 blocks, and each of those has weight at least d_inner = 3,
    so the distance is exactly 9.
    """
    inner_s = _ops(FIVE_STABS)
    inner_x = from_string(FIVE_X)
    inner_z = from_string(FIVE_Z)
    inner_report = verify_code_distance(inner_s, [inner_x], [inner_z], scan_weight=5)
    if inner_report["d"] != 3:
        raise ValueError("inner five-qubit distance is not 3")
    # Minimum-weight logical representatives, used for the weight-9 witness.
    mx, mz = _minimum_logicals(inner_s, 1, scan=3)
    blocks = 5
    stabs: list[Pauli] = []
    for b in range(blocks):
        for gen in inner_s:
            stabs.append(_place_block(gen, b, blocks))
    for gen in inner_s:
        stabs.append(_lift_outer(gen, 0, mx[0], mz[0], blocks))
    logical_x = _lift_outer(mx[0], 0, mx[0], mz[0], blocks)
    logical_z = _lift_outer(mz[0], 0, mx[0], mz[0], blocks)
    for i, left in enumerate(stabs):
        for right in stabs[i + 1 :]:
            if not commutes(left, right):
                raise ValueError("concatenated stabilizers do not commute")
    rank = rank_mod2([symplectic_vector(op) for op in stabs], 50)
    if rank != 24:
        raise ValueError(f"concatenated rank {rank}")
    if commutes(logical_x, logical_z):
        raise ValueError("concatenated logicals commute")
    for stab in stabs:
        if not commutes(logical_x, stab) or not commutes(logical_z, stab):
            raise ValueError("concatenated logical hits a stabilizer")
    if logical_x.weight() < 9 or logical_z.weight() < 9:
        raise ValueError("concatenated witness is lighter than 9")
    return {
        "name": "clock-z24-d9",
        "stabilizers": stabs,
        "logical_x": [logical_x],
        "logical_z": [logical_z],
        "n": 25,
        "k": 1,
        "d": 9,
        "rank": rank,
        "exact": True,
        "t": 4,
        "reason": "d = d_inner * d_outer = 3 * 3",
    }


def slow_face_isolation() -> bool:
    """Every weight-1 Pauli on qubits 1 and 2 anticommutes with M or J_2."""
    m = from_string("YZIIIII")
    j2 = from_string("ZYIIIII", 2)
    for qubit in (0, 1):
        for letter in "XYZ":
            letters = ["I"] * 7
            letters[qubit] = letter
            err = from_string("".join(letters))
            if commutes(err, m) and commutes(err, j2):
                return False
    return True


def fast_leg_distance() -> int:
    report = five_qubit()
    return int(report["d"])


def compile_record(code: dict) -> list[str]:
    lines = []
    gammas = build_n_qubit_gammas(int(code["n"]))
    for op in code["stabilizers"]:
        phase, indices = compile_pauli(op, gammas)
        lines.append(format_word(phase, indices, "Gamma"))
    if code["name"] == "perfect-513":
        for text in FIVE_STABS:
            phase, indices = compile_fast_pauli_to_F(text)
            lines.append(format_word(phase, indices, "F"))
    return lines


def summary_row(code: dict) -> str:
    status = "PASS" if code["exact"] else "BOUND"
    return f"[{code['n']}, {code['k']}, {code['d']}, {code['t']}, {status}] {code['name']}"


def all_codes() -> list[dict]:
    return [five_qubit(), subsystem_4x32(), steane(), clock_seat(), extended_d5(), clock_z24_d9()]
