"""O(1) address routing through the flip table and the phase table."""

from __future__ import annotations

from address import (
    apply_gammas,
    flip_signs,
    fmt_signs,
    gammas,
    phi,
    psi,
    route_phase,
)
from symplectic import G, fmt_g, vector_equal


def parse_word(text: str) -> list[int]:
    if text.strip() == "":
        return []
    indices = []
    for piece in text.split(","):
        k = int(piece.strip())
        if k < 0 or k > 13:
            raise ValueError(f"generator index out of range: {k}")
        indices.append(k)
    return indices


def trace(signs: tuple[int, ...], word: list[int], gauge: str) -> list[tuple[int, G, tuple[int, ...]]]:
    """Each step records the generator, the phase taken from the source line, and the arrival signs."""
    steps = []
    current = signs
    for k in word:
        factor = phi(k, current, gauge)
        current = flip_signs(current, k)
        steps.append((k, factor, current))
    return steps


def routed_vector(signs: tuple[int, ...], word: list[int], gauge: str) -> tuple[G, tuple[int, ...], list]:
    phase, final = route_phase(signs, word, gauge)
    from symplectic import scale_vec

    return phase, final, scale_vec(phase, psi(final, gauge))


def agrees_with_matrix(signs: tuple[int, ...], word: list[int], gauge: str) -> bool:
    phase, final, routed = routed_vector(signs, word, gauge)
    del phase, final
    return vector_equal(routed, apply_gammas(signs, word, gauge))


def format_route(signs: tuple[int, ...], word: list[int], gauge: str) -> str:
    phase, final, _routed = routed_vector(signs, word, gauge)
    lines = [
        f"gauge {gauge}",
        f"start {fmt_signs(signs)}",
        f"word {','.join(str(k) for k in word) if word else '(empty)'}",
    ]
    running = G(1)
    current = signs
    for k, factor, arrived in trace(signs, word, gauge):
        running = running * factor
        lines.append(
            f"Gamma_{k}: {fmt_signs(current)} -- {fmt_g(factor)} --> {fmt_signs(arrived)}"
        )
        current = arrived
    lines.append(f"accumulated {fmt_g(phase)}")
    lines.append(f"final {fmt_signs(final)}")
    ok = agrees_with_matrix(signs, word, gauge)
    lines.append("matrix agreement PASS" if ok else "matrix agreement FAIL")
    if running != phase:
        lines.append("phase accumulator FAIL")
    return "\n".join(lines)


def generator_words() -> list[str]:
    return [op[1] for op in gammas()]
