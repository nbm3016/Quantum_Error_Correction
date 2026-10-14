"""Exact checks for the 7-qubit router. No floats."""

from __future__ import annotations

import unittest

from address import (
    address_ops,
    all_sign_tuples,
    charge_q,
    dihedral,
    e_eigenvalue,
    flip_signs,
    gammas,
    gamma8_eigenvalue,
    phi,
    psi,
)
from rooms import f9_bridge_ok, factor_identities, pair_rank_report, room_lines, simplest_rank9, walk_rank
from router import agrees_with_matrix
from syndrome import recover
from symplectic import (
    G,
    apply_op,
    comm_sign,
    mul_op,
    parse_pauli,
    scale,
    squares_to,
    transpose_op,
    vector_equal,
)


def _expect(name: str, ok: bool, detail: str = "") -> None:
    if not ok:
        raise AssertionError(name if not detail else f"{name}: {detail}")


def run_certificate() -> str:
    gens = gammas()
    words = (
        "XXXXXXX",
        "XXXXXXY",
        "XXXXXYI",
        "XXXXXZI",
        "XXXXYII",
        "XXXXZII",
        "XXXYIII",
        "XXXZIII",
        "XXYIIII",
        "XXZIIII",
        "XYIIIII",
        "XZIIIII",
        "YIIIIII",
        "ZIIIIII",
    )
    _expect("fourteen words", [op[1] for op in gens] == list(words) and all(op[0] == 0 for op in gens))
    for i, left in enumerate(gens):
        _expect(f"square {i}", squares_to(left) == 1)
        for j in range(i + 1, 14):
            _expect(f"anticommute {i},{j}", comm_sign(left, gens[j]) == -1)

    triple = dihedral()
    _expect("P", triple["P"] == (3, "XXXXXXZ"))
    _expect("Omega", triple["Omega"] == (2, "XXXXXXZ") and squares_to(triple["Omega"]) == 1)
    _expect("C", triple["C"] == (0, "ZYZYZYX") and squares_to(triple["C"]) == 1)
    _expect("E", triple["E"] == (3, "YZYZYZY") and squares_to(triple["E"]) == -1)
    _expect("Omega C = - C Omega", mul_op(triple["Omega"], triple["C"]) == scale(2, mul_op(triple["C"], triple["Omega"])))
    for k, op in enumerate(gens):
        conjugated = mul_op(mul_op(triple["C"], op), triple["C"])
        _expect(f"transpose {k}", conjugated == transpose_op(op))

    ops = address_ops()
    _expect("M J2", mul_op(ops["M"], ops["J_2"]) == (2, "XXIIIII"))
    _expect("D B", mul_op(ops["D"], ops["B"]) == (0, "IIIXXII"))
    _expect("A Cbit", mul_op(ops["A"], ops["C_bit"]) == (2, "IIIIIXZ"))
    y3 = (0, "IIYIIII")
    _expect("Sigma", ops["Sigma"] == scale(2, mul_op(y3, mul_op(ops["B"], ops["A"]))))
    _expect(
        "Gamma8 product",
        mul_op(mul_op(mul_op(mul_op(ops["M"], ops["J_2"]), ops["A"]), ops["B"]), ops["Sigma"]) == gens[8],
    )

    twin_expect = {
        (11, 13): scale(3, ops["M"]),
        (10, 12): scale(3, ops["J_2"]),
        (1, 2): scale(1, ops["A"]),
        (4, 6): scale(1, ops["B"]),
        (0, 3): scale(3, ops["C_bit"]),
        (5, 7): scale(3, ops["D"]),
    }
    for (i, j), op in twin_expect.items():
        _expect(f"twin {i},{j}", mul_op(gens[i], gens[j]) == op)
    isolators = {
        (12, 13): (1, "XIIIIII"),
        (10, 11): (1, "IXIIIII"),
        (8, 9): (1, "IIXIIII"),
        (6, 7): (1, "IIIXIII"),
        (4, 5): (1, "IIIIXII"),
        (2, 3): (1, "IIIIIXI"),
        (0, 1): (1, "IIIIIIZ"),
    }
    for (i, j), op in isolators.items():
        _expect(f"isolator {i},{j}", mul_op(gens[i], gens[j]) == op)

    for signs in all_sign_tuples():
        for gauge in ("C", "L"):
            vec = psi(signs, gauge)
            if gauge == "C":
                _expect("gauge C origin", vec[0] == G(1))
            for k, op in enumerate(gens):
                image = apply_op(op, vec)
                dest = flip_signs(signs, k)
                expect = [phi(k, signs, gauge) * value for value in psi(dest, gauge)]
                _expect(f"phase {gauge} {k} {signs}", vector_equal(image, expect))
        base = psi(signs, "C")
        for name, op in ops.items():
            image = apply_op(op, base)
            sign = {
                "M": signs[0],
                "J_2": signs[1],
                "D": signs[2],
                "B": signs[3],
                "A": signs[4],
                "C_bit": signs[5],
                "Sigma": signs[6],
            }[name]
            factor = G(sign)
            _expect(name, vector_equal(image, [factor * value for value in base]))
        e_image = apply_op(triple["E"], base)
        _expect("E eigenvalue", vector_equal(e_image, [e_eigenvalue(signs) * value for value in base]))
        g8 = apply_op(gens[8], base)
        _expect("Gamma8 eigenvalue", vector_equal(g8, [gamma8_eigenvalue(signs) * value for value in base]))
        factor = G(-1) if charge_q(signs) else G(1)
        _expect("gauge L rescale", vector_equal(psi(signs, "L"), [factor * value for value in base]))

    _expect("sample route", agrees_with_matrix((1, 1, 1, 1, 1, 1, 1), [0, 3, 8, 12], "C"))
    _expect("sample route L", agrees_with_matrix((1, -1, 1, 1, -1, 1, -1), [0, 3, 8, 12], "L"))
    recovery, _syndrome, _flips = recover(parse_pauli("IIXIIII"), (1, 1, 1, 1, 1, 1, 1))
    restored = apply_op(recovery, apply_op(parse_pauli("IIXIIII"), psi((1, 1, 1, 1, 1, 1, 1), "C")))
    _expect("recovery", vector_equal(restored, psi((1, 1, 1, 1, 1, 1, 1), "C")))

    failures = factor_identities()
    _expect("fast words", not failures, ",".join(failures))
    lines = room_lines()
    _expect("sixteen rooms", len(lines) == 16)
    _expect("F9 bridge", f9_bridge_ok(lines))
    single = [walk_rank(vec)[0] for _signs, vec in lines]
    _expect("single-line rank 6", single == [6] * 16, str(set(single)))
    checked, rank7 = pair_rank_report(lines)
    _expect("two-line rank 7", checked > 0 and rank7 == checked, f"{rank7}/{checked}")
    terms, _vec, rank, pivots, relation = simplest_rank9(lines)
    _expect("rank 9 witness", rank == 9 and relation is not None and len(pivots) == 9)
    _expect("witness lines", len(terms) >= 1)
    if relation is not None and all(coeff.is_zero() for coeff in relation):
        raise AssertionError("empty relation")
    _expect("expansion", run_expansion())
    return "AGC QEC FULL SUITE CERTIFICATE PASS"


def run_expansion() -> bool:
    import hashlib
    from pathlib import Path

    from codes_high_d import all_codes, fast_leg_distance, slow_face_isolation
    from decoder import decode_text
    from lift_n import address_independent, build_n_qubit_dihedral, phase_name, to_string
    from non_clifford import format_t
    from presets import compile_fast_pauli_to_F, compile_pauli_to_gamma
    from synth import format_synth

    letters = "IXYZ"
    for a in letters:
        for b in letters:
            for c in letters:
                for d in letters:
                    for e in letters:
                        for f in letters:
                            for gch in letters:
                                compile_pauli_to_gamma(a + b + c + d + e + f + gch)
    for a in letters:
        for b in letters:
            for c in letters:
                for d in letters:
                    for e in letters:
                        compile_fast_pauli_to_F(a + b + c + d + e)
    triple = build_n_qubit_dihedral(7)
    _expect("dihedral P", phase_name(triple["P"].phase) + to_string(triple["P"]) == "-iXXXXXXZ")
    _expect("dihedral Omega", phase_name(triple["Omega"].phase) + to_string(triple["Omega"]) == "-1XXXXXXZ")
    _expect("dihedral C", to_string(triple["C"]) == "ZYZYZYX")
    _expect("dihedral E", phase_name(triple["E"].phase) + to_string(triple["E"]) == "-iYZYZYZY")
    _expect("address 15", address_independent(15))
    codes = {code["name"]: code for code in all_codes()}
    _expect("five d", codes["perfect-513"]["d"] == 3)
    _expect("steane d", codes["steane-713"]["d"] == 3)
    _expect("seat d", codes["clock-seat-833"]["d"] == 3)
    _expect("dodeca d", codes["extended-d5"]["d"] == 5)
    _expect("clock d", codes["clock-z24-d9"]["d"] == 9 and codes["clock-z24-d9"]["n"] == 25)
    _expect("slow face", slow_face_isolation())
    _expect("fast leg", fast_leg_distance() == 3)
    for name, errors in (("perfect-513", "X1"), ("steane-713", "Z4"), ("extended-d5", "Y2")):
        _err, _rec, _word, ok = decode_text(name, errors)
        _expect(f"decode {name}", ok)
    text = format_t((1, 1, 1, 1, 1, 1, 1), ["T1", "G0", "T3", "T1"])
    _expect("t norm", "norm PASS" in text)
    _expect("synth", "operator PASS" in format_synth([0, 3, 8]))
    from exact_extend import run_extend
    from stim_load import assert_d3_counts, assert_d3_noiseless, assert_ring_identity

    _expect("extend", run_extend() == "EXACT EXTEND PASS")
    assert_d3_counts()
    _expect("d3 counts", True)
    assert_ring_identity()
    _expect("d3 ring", True)
    assert_d3_noiseless()
    _expect("d3 noiseless logical", True)
    baseline = Path(__file__).resolve().parents[1] / "core" / "complete_baseline.json"
    if baseline.is_file():
        digest = hashlib.sha256(baseline.read_bytes()).hexdigest()
        _expect(
            "baseline",
            digest == "e79808eb82123bbbd5a98cfa23d5c26ffd02677915cc94e28cfb5fc59257eabc",
            digest,
        )
    return True


class RouterTests(unittest.TestCase):
    def test_certificate(self) -> None:
        self.assertEqual(run_certificate(), "AGC QEC FULL SUITE CERTIFICATE PASS")


if __name__ == "__main__":
    unittest.main()
