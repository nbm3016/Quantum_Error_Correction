"""Compile a staircase word into H, S, and CNOT, and check it in Q(i, sqrt(2))."""

from __future__ import annotations

from fractions import Fraction

from lift_n import Pauli, build_n_qubit_gammas, identity, mul, product_subset, to_string
from non_clifford import QISqrt2, one
from symplectic import G, apply_op, fmt_op

Gate = tuple


def _half_root() -> QISqrt2:
    return QISqrt2(0, 0, Fraction(1, 2), 0)


def _apply_gate(vec: list[QISqrt2], gate: Gate) -> list[QISqrt2]:
    n = (len(vec)).bit_length() - 1
    kind = gate[0]
    if kind == "PHASE":
        factor = (one(), QISqrt2(0, 1), QISqrt2(-1), QISqrt2(0, -1))[gate[1]]
        return [factor * amp for amp in vec]
    out = [QISqrt2() for _ in vec]
    if kind in {"H", "S", "SDG", "Z", "X"}:
        qubit = gate[1]
        bit = n - qubit
        mask = 1 << bit
        if kind == "H":
            scale = _half_root()
            for i in range(len(vec)):
                if i & mask:
                    continue
                j = i | mask
                a = vec[i]
                b = vec[j]
                out[i] = scale * (a + b)
                out[j] = scale * (a - b)
            return out
        for i, amp in enumerate(vec):
            if kind == "X":
                out[i ^ mask] = out[i ^ mask] + amp
            elif kind == "Z":
                sign = QISqrt2(-1) if (i & mask) else one()
                out[i] = sign * amp
            elif kind == "S":
                sign = QISqrt2(0, 1) if (i & mask) else one()
                out[i] = sign * amp
            else:
                sign = QISqrt2(0, -1) if (i & mask) else one()
                out[i] = sign * amp
        return out
    control, target = gate[1], gate[2]
    cmask = 1 << (n - control)
    tmask = 1 << (n - target)
    for i, amp in enumerate(vec):
        j = i ^ tmask if (i & cmask) else i
        out[j] = out[j] + amp
    return out


def _simulate(n: int, gates: list[Gate], column: int) -> list[QISqrt2]:
    vec = [QISqrt2() for _ in range(1 << n)]
    vec[column] = one()
    for gate in gates:
        vec = _apply_gate(vec, gate)
    return vec


def _local(letter: str, qubit: int) -> tuple[list[Gate], list[Gate]]:
    """U and U† with U Z U† equal to the local Pauli, as application-order lists."""
    if letter == "I" or letter == "Z":
        return [], []
    if letter == "X":
        return [("H", qubit)], [("H", qubit)]
    if letter == "Y":
        # Y = (S H) Z (H S†). Application order runs the dagger first.
        forward = [("H", qubit), ("S", qubit)]
        backward = [("SDG", qubit), ("H", qubit)]
        return forward, backward
    raise ValueError(letter)


def synthesize_pauli(op: Pauli) -> list[Gate]:
    n = op.n
    support = [qubit for qubit in range(1, n + 1) if op.letter(n - qubit) != "I"]
    if not support:
        return []
    phase_gate: list[Gate] = [("PHASE", op.phase)] if op.phase else []
    root = support[0]
    u_dag: list[Gate] = []
    u_fwd: list[Gate] = []
    for qubit in support:
        fwd, back = _local(op.letter(n - qubit), qubit)
        u_dag.extend(back)
        u_fwd.extend(fwd)
    ladder = [("CNOT", qubit, root) for qubit in support[1:]]
    undo = list(reversed(ladder))
    return phase_gate + u_dag + undo + [("Z", root)] + ladder + u_fwd


def synthesize_word(indices: list[int], n: int = 7) -> tuple[Pauli, list[Gate], str]:
    gammas = build_n_qubit_gammas(n)
    op = product_subset(gammas, tuple(indices))
    gates = synthesize_pauli(op)
    return op, gates, _qasm(n, gates)


def _qasm(n: int, gates: list[Gate]) -> str:
    lines = ["OPENQASM 2.0;", 'include "qelib1.inc";', f"qreg q[{n}];"]
    for gate in gates:
        if gate[0] == "PHASE":
            lines.append(f"// global phase {('1','i','-1','-i')[gate[1]]}")
        elif gate[0] == "H":
            lines.append(f"h q[{gate[1] - 1}];")
        elif gate[0] == "S":
            lines.append(f"s q[{gate[1] - 1}];")
        elif gate[0] == "SDG":
            lines.append(f"sdg q[{gate[1] - 1}];")
        elif gate[0] == "Z":
            lines.append(f"z q[{gate[1] - 1}];")
        elif gate[0] == "X":
            lines.append(f"x q[{gate[1] - 1}];")
        else:
            lines.append(f"cx q[{gate[1] - 1}],q[{gate[2] - 1}];")
    return "\n".join(lines)


def _as_qi(value: G) -> QISqrt2:
    return QISqrt2(value.re, value.im, 0, 0)


def equivalent(op: Pauli, gates: list[Gate]) -> bool:
    """The circuit's columns equal the Pauli matrix, up to one global unit phase."""
    n = op.n
    columns = []
    for basis in range(1 << n):
        got = _simulate(n, gates, basis)
        columns.append(got)
    # Compare with apply_op on each basis vector.
    from symplectic import basis_vector

    ratios = []
    for basis in range(1 << n):
        image = apply_op((op.phase, to_string(op)), basis_vector(basis, 1 << n))
        got = columns[basis]
        ratio = None
        for src, amp in zip(image, got, strict=True):
            if src.is_zero() and amp.is_zero():
                continue
            if src.is_zero() or amp.is_zero():
                return False
            candidate = amp / _as_qi(src)
            if ratio is None:
                ratio = candidate
            elif candidate != ratio:
                return False
        if ratio is not None:
            ratios.append(ratio)
    return len(ratios) > 0 and all(item == one() for item in ratios)


def format_synth(indices: list[int], n: int = 7) -> str:
    op, gates, qasm = synthesize_word(indices, n)
    counts = {"h": 0, "s": 0, "sdg": 0, "z": 0, "cx": 0}
    for gate in gates:
        if gate[0] == "PHASE":
            continue
        key = {"H": "h", "S": "s", "SDG": "sdg", "Z": "z", "CNOT": "cx", "X": "x"}[gate[0]]
        counts[key] = counts.get(key, 0) + 1
    ok = equivalent(op, gates)
    lines = [
        f"pauli {fmt_op((op.phase, to_string(op)))}",
        f"gates h={counts['h']} s={counts['s']} sdg={counts['sdg']} z={counts['z']} cx={counts['cx']}",
        qasm,
        "operator PASS" if ok else "operator FAIL",
    ]
    return "\n".join(lines)
