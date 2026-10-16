"""Address register, seven extractors, and the commutation record.

Run from this directory: python cli.py <command>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from address import (
    address_ops,
    charge_q,
    e_eigenvalue,
    fmt_blocks,
    fmt_signs,
    fmt_vector,
    gamma8_eigenvalue,
    parse_signs,
    psi,
)
from router import format_route, generator_words, parse_word
from syndrome import format_image, format_syndrome
from symplectic import fmt_g, fmt_op

ROOT = Path(__file__).resolve().parent
PARENT = ROOT.parent
CORE_SHA = "e79808eb82123bbbd5a98cfa23d5c26ffd02677915cc94e28cfb5fc59257eabc"
WORK_SHA = "1d9b54a788142b9d2c818d236d55e8a7a71b841c0e032107bcc052f23abe707a"
COUNTS = "continuous_knobs = 0, n_LOCK = 35, n_promoted = 0, n_OPEN = 15"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _audit_baselines() -> list[str]:
    targets = (
        (PARENT / "core" / "complete_baseline.json", CORE_SHA, "core"),
        (PARENT / "project" / "complete_baseline.json", None, "project"),
    )
    lines = []
    for path, expected, label in targets:
        if not path.is_file():
            lines.append(f"{label} complete_baseline.json absent")
            continue
        digest = _sha256(path)
        if label == "core":
            status = "MATCH" if digest == expected else "DIFFER"
            lines.append(f"core complete_baseline.json {status} {digest}")
        else:
            if digest == WORK_SHA:
                status = "MATCH working"
            elif digest == CORE_SHA:
                status = "MATCH core"
            else:
                status = "DIFFER"
            lines.append(f"project complete_baseline.json {status} {digest}")
    return lines


def _audit_knobs() -> str:
    path = PARENT / "falsifier" / "isolate_updated.json"
    if not path.is_file():
        return "isolate_updated.json absent"
    data = json.loads(path.read_text(encoding="utf-8"))
    locks = [item for item in data.get("locks", []) if item.get("status") == "LOCK"]
    knob = next((item for item in locks if item.get("name") == "continuous_knobs"), None)
    value = None if knob is None else knob.get("value")
    return f"isolate read-only locks {len(locks)} continuous_knobs {value}"


def cmd_status() -> int:
    print("workspace", ROOT)
    print("siblings read-only")
    for line in _audit_baselines():
        print(line)
    print(_audit_knobs())
    print(COUNTS)
    print("generators")
    for index, word in enumerate(generator_words()):
        print(f"Gamma_{index} {word}")
    print("address")
    for name, op in address_ops().items():
        print(f"{name} {fmt_op(op)}")
    return 0


def cmd_address(signs_text: str, gauge: str) -> int:
    signs = parse_signs(signs_text)
    print(f"gauge {gauge}")
    print(f"signs {fmt_signs(signs)}")
    print(fmt_blocks(signs))
    print(f"Q {charge_q(signs)}")
    print(f"E {fmt_g(e_eigenvalue(signs))}")
    print(f"Gamma_8 {fmt_g(gamma8_eigenvalue(signs))}")
    print(fmt_vector(psi(signs, gauge)))
    return 0


def cmd_route(start: str, word: str, gauge: str) -> int:
    print(format_route(parse_signs(start), parse_word(word), gauge))
    return 0 if "FAIL" not in format_route(parse_signs(start), parse_word(word), gauge) else 1


def cmd_map() -> int:
    print(format_image())
    print(COUNTS)
    return 0


def cmd_syndrome(error: str, start: str) -> int:
    print(format_syndrome(error, parse_signs(start)))
    return 0


def cmd_walk() -> int:
    from rooms import format_witness, pair_rank_report, room_lines, simplest_rank9, walk_rank

    lines = room_lines()
    ranks = [walk_rank(vec)[0] for _signs, vec in lines]
    print(f"single-line ranks {min(ranks)}..{max(ranks)} count {len(ranks)}")
    checked, rank7 = pair_rank_report(lines)
    print(f"two-line s_D flip with coefficient i: {rank7}/{checked} have rank 7")
    print(format_witness(simplest_rank9(lines)))
    return 0


def cmd_verify() -> int:
    from test_qec import run_certificate

    print(run_certificate())
    return 0


def cmd_compile(preset: str | None, strings: str | None) -> int:
    from codes_high_d import compile_record
    from presets import compile_fast_pauli_to_F, compile_pauli_to_gamma, format_word

    if preset:
        # Local import avoids building every code for a string compile.
        from codes_high_d import (
            clock_seat,
            extended_d5,
            five_qubit,
            steane,
            subsystem_4x32,
        )

        built = {
            "perfect-513": five_qubit,
            "subsystem-4x32": subsystem_4x32,
            "steane-713": steane,
            "clock-seat-833": clock_seat,
            "extended-d5": extended_d5,
            "clock-z24-d9": None,
        }[preset]
        if preset == "clock-z24-d9":
            from codes_high_d import clock_z24_d9

            code = clock_z24_d9()
        else:
            code = built()
        print(f"[[{code['n']}, {code['k']}, {code['d']}]] {code['name']}")
        for line in compile_record(code):
            print(line)
        return 0
    if not strings:
        raise SystemExit("compile-code needs --preset or --strings")
    for text in strings.split(","):
        word = text.strip()
        phase, indices = compile_pauli_to_gamma(word)
        print(format_word(phase, indices, "Gamma"), word)
        if len(word) == 5:
            f_phase, f_indices = compile_fast_pauli_to_F(word)
            print(format_word(f_phase, f_indices, "F"))
    return 0


def cmd_high_distance() -> int:
    from codes_high_d import all_codes, fast_leg_distance, slow_face_isolation, summary_row

    print("n k d t status name")
    for code in all_codes():
        print(summary_row(code))
    print("slow-face", "PASS" if slow_face_isolation() else "FAIL")
    print("fast-leg-d", fast_leg_distance())
    return 0


def cmd_decode(name: str, errors: str) -> int:
    from decoder import decode_text
    from lift_n import phase_name, to_string
    from presets import format_word
    from symplectic import fmt_op

    error, recovery, word, ok = decode_text(name, errors)
    phase, indices = word
    print(f"error {fmt_op((error.phase, to_string(error)))}")
    print(f"recovery {fmt_op((recovery.phase, to_string(recovery)))}")
    print(format_word(phase, indices, "Gamma"))
    print("restore PASS" if ok else "restore FAIL")
    return 0 if ok else 1


def cmd_scale(n: int) -> int:
    from lift_n import (
        address_independent,
        build_n_qubit_dihedral,
        build_n_qubit_gammas,
        commutes,
        mul,
        phase_name,
        to_string,
    )

    gammas = build_n_qubit_gammas(n)
    if len(gammas) != 2 * n:
        raise SystemExit("generator count")
    for i, left in enumerate(gammas):
        square = mul(left, left)
        if not square.is_scalar() or square.phase != 0:
            raise SystemExit(f"Gamma_{i} is not an involution")
        for right in gammas[i + 1 :]:
            if commutes(left, right):
                raise SystemExit("generators must anticommute")
    triple = build_n_qubit_dihedral(n)
    print(f"n {n}")
    print(f"generators {2 * n}")
    for key in ("P", "Omega", "C", "E"):
        print(f"{key} {phase_name(triple[key].phase)}{to_string(triple[key])}")
    if n % 2 == 1 and n >= 3:
        print("address", "PASS" if address_independent(n) else "FAIL")
    else:
        print("address skipped")
    return 0


def cmd_t(start: str, gates: str) -> int:
    from non_clifford import format_t

    text = format_t(parse_signs(start), [piece.strip() for piece in gates.split(",") if piece.strip()])
    print(text)
    return 0 if "norm PASS" in text else 1


def cmd_synth(word: str) -> int:
    from synth import format_synth

    text = format_synth(parse_word(word))
    print(text)
    return 0 if "operator PASS" in text else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cli.py")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("status")
    sub.add_parser("map")
    address = sub.add_parser("address")
    address.add_argument("--signs", required=True)
    address.add_argument("--gauge", choices=("C", "L"), default="C")
    route = sub.add_parser("route")
    route.add_argument("--start", required=True)
    route.add_argument("--word", required=True)
    route.add_argument("--gauge", choices=("C", "L"), default="C")
    syndrome = sub.add_parser("syndrome")
    syndrome.add_argument("--error", required=True)
    syndrome.add_argument("--start", default="+1,+1,+1,+1,+1,+1,+1")
    sub.add_parser("walk-witness")
    sub.add_parser("verify")
    compile_code = sub.add_parser("compile-code")
    compile_code.add_argument("--preset", default=None)
    compile_code.add_argument("--strings", default=None)
    sub.add_parser("high-distance")
    decode = sub.add_parser("decode")
    decode.add_argument("--code", required=True)
    decode.add_argument("--errors", required=True)
    scale = sub.add_parser("scale-n")
    scale.add_argument("--qubits", type=int, required=True)
    t_gate = sub.add_parser("t-gate")
    t_gate.add_argument("--start", required=True)
    t_gate.add_argument("--gates", required=True)
    synth = sub.add_parser("synth")
    synth.add_argument("--word", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "status":
        return cmd_status()
    if args.command == "map":
        return cmd_map()
    if args.command == "address":
        return cmd_address(args.signs, args.gauge)
    if args.command == "route":
        return cmd_route(args.start, args.word, args.gauge)
    if args.command == "syndrome":
        return cmd_syndrome(args.error, args.start)
    if args.command == "walk-witness":
        return cmd_walk()
    if args.command == "verify":
        return cmd_verify()
    if args.command == "compile-code":
        return cmd_compile(args.preset, args.strings)
    if args.command == "high-distance":
        return cmd_high_distance()
    if args.command == "decode":
        return cmd_decode(args.code, args.errors)
    if args.command == "scale-n":
        return cmd_scale(args.qubits)
    if args.command == "t-gate":
        return cmd_t(args.start, args.gates)
    if args.command == "synth":
        return cmd_synth(args.word)
    return 2


if __name__ == "__main__":
    sys.exit(main())
