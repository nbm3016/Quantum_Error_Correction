# Exact extension

This note extends the router already in this folder. It does not replace it.
`continuous_knobs = 0`. `n_LOCK = 35`. `n_promoted = 0`. `n_OPEN = 15`.
No float is used. Claims below are labelled sourced or unverified.

## 1. Pauli faults and a symbolic rate

Result. `exact_extend.py` gives every prep, gate, and measurement one Pauli fault.
The rate `p` stays a symbol. The checked channel after `T` is identity with weight `1-p`
and `X`, `Y`, `Z` with weight `p/3` each.

Test. `run_extend` in `exact_extend.py`, called from `python cli.py verify`.

Next step this week. Put the same channel on the CNOT and the Hadamard in `synth.py`.

## 2. Cultivation past a bare injection

Sourced, arXiv:2409.17595, abstract: "We call our construction "magic state cultivation" because it gradually grows the size and reliability of one state."

Sourced, arXiv:2509.01224, abstract: "The resultant states to the d=3 and d=5 variant magic state cultivation circuits can be expressed as a sum of 4 and 8 Clifford ZX-digrams respectively."

Sourced, arXiv:2512.23799, abstract: "When applied to protocols based on code switching, magic state cultivation, and magic state distillation, our method yields a complexity polynomial in (i) the number of qubits and (ii) the non-stabilizerness, e.g., stabilizer rank or Pauli rank, of the target encoded magic state."

Sourced, arXiv:2604.15043, abstract: "The FT preparation of stabiliser and magic states, including the first stage of magic state cultivation, is analysed without approximations."

Unverified: this folder does not contain the colour-code circuits of those papers, and it does not reproduce their fidelities.

Result. The circuit here is an injection `T|+>` followed by `T†` and an `X` post-selection.
The conditional fidelity is exactly `1 - p/6 - p^2/9 + O(p^3)`.
A `Z` fault is rejected. The `X` and `Y` faults leave the linear term.

Test. `cultivation_fidelity` asserts those three coefficients.

Next step this week. Add one stabilizer check on a second qubit and recompute the series.

## 3. Fault distance

Sourced, arXiv:2604.15043: "To our knowledge, SyQMA is the first tool capable of determining the circuit-level fault distance of a non-Clifford circuit such as magic state preparation."

Sourced, same paper: "A state preparation program preserves the distance d of the code if all combinations of up to t = (d-1)/2 noise events can be successfully decoded by an ideal decoder."

Result. A Pauli inserted after one `T` is still a Pauli. The minimum undetected logical fault is the code distance: 3 on Steane, 5 on the `[[11,1,5]]` dodecacode.

Test. `steane_plus_t_fault_distance` and `dodeca_fault_distance`.

Next step this week. Move the fault to before `T` and keep the image `T P T†`, which is not a Pauli.

## 4. Logical Cliffords and one merge

Sourced, arXiv:2604.15043, Outlook: "For QEC, the decoding framework may be extended beyond state preparation, e.g., by computing channel fidelities between noiseless and noisy versions of QEC programs like QEC cycles or lattice surgery. Decoding could also incorporate correction programs beyond Pauli operators, including transversal logical Clifford gates or more complex dynamic programs."

Unverified: SyQMA did not publish that lattice-surgery calculation. The merge below is ours.

Result. Transversal `H` and `S` preserve the Steane stabilizer group, and transversal `H` sends `XXXXXXX` to `ZZZZZZZ`.
Two 2-qubit repetition codes with seam `IZZI` leave one logical qubit: `XXXX` and `ZIII`. The other `Z` logical differs by the product of the three stabilizers.

Test. `transversal_steane` and `lattice_merge`.

Next step this week. Repeat the seam check for two copies of the Steane code.

## 5. Exact series against Tsim and SOFT

Sourced, arXiv:2604.01059, abstract: "After the one-time compilation, one can sample detector or measurement shots in linear time in the number of Clifford gates and exponentially only in the number of non-Clifford gates."

Sourced, arXiv:2512.23037, Table I: at noise `0.001`, "Reliable simulation" logical error rate `4.59 × 10^{-9}`, discrepancy factor `7.65`, against empirical conjecture `6 × 10^{-10}`.

Sourced, same table: at noise `0.0005`, reliable simulation `1.57 × 10^{-10}`, discrepancy factor `7.85`, empirical conjecture `2 × 10^{-11}`.

Unverified: this folder did not run Tsim or SOFT, and the `-p/6` series is not their 42-qubit cultivation rate.

Result. On the gadget in section 2 the exact conditional infidelity starts at `p/6`. A Monte Carlo estimate of that linear term has sampling noise. SOFT's table says an empirical Clifford proxy missed the `d=5` cultivation rate by a factor of about 8 at `p = 10^{-3}`. The exact engine is the right tool for the linear coefficient. It is not a substitute for their shot count.

Test. The series assertion in `run_extend`.

Next step this week. Expand the same gadget through `p^4` and compare one published SOFT row only after the circuits are the same.

## 6. Decoding at scale

Sourced, arXiv:2604.15043: "We do not envision this version of SyQMA being used for large-scale simulations that involve many measurements and non-Clifford gates, but rather for the detailed analysis of error propagation (stochastic, biased and coherent), fault-tolerance (overall circuit-distance and syndrome-wise) and decoding (maximum-likelihood upper bound, syndrome postselection) in universal QEC gadgets of small to moderate size."

Sourced, arXiv:2512.23037: "Using only modest GPU resources, SOFT performs over 200 billion shots and achieves the first ground-truth simulation of the cultivation protocol at a non-trivial scale."

Result. An exact symbolic engine certifies fault distance and the leading power of `p` for a small gadget. It does not decode a distance-15 lattice or sample a 42-qubit cultivation. Correlated noise is exact when it is a Pauli channel on a small support. It stops being exact, in practice, when the fault set is larger than the weight scan.

Next step this week. Keep the weight scan, and do not add a float decoder.

## 7. Ten further checks

`continuous_knobs = 0`. `n_LOCK = 35`. `n_promoted = 0`. `n_OPEN = 15`.
No float is used in `exact_extend.py`. Each item is labelled sourced or unverified.
`run_extend` still returns `EXACT EXTEND PASS`.

### 7.1 Noise and gadgets

Sourced, arXiv:2609.18922, equation (8a): the one-qubit depolarising channel is "(1-p) ρ" plus "p/3 (XρX + YρY + ZρZ)".

Sourced, same paper, equation (8b): the two-qubit channel puts weight "p/15" on each of the 15 non-identity Paulis.

Result, this engine. Those two channels are exact polynomials. The Hadamard channel is the one-qubit channel. On `|0>`, then `H`, then that channel, the fidelity to `|+>` is `1 - (2/3) p`. Z dephasing has fidelity `1 - p` on `|+>`. A coherent Z rotation is kept as symbols `c` and `s` with `c^2 + s^2 = 1`. It is not a float angle.

Sourced, arXiv:2609.18922, equation (7): "T† X T = (X - Y) / sqrt(2)", "T† Y T = (X + Y) / sqrt(2)", with `T = diag(1, e^{i π/4})`.

Result, this engine. Those three conjugations hold in `Z[i, 1/sqrt(2)]`, and so do the forward images `T X T† = (X + Y) / sqrt(2)`, `T Y T† = (Y - X) / sqrt(2)`, and `T Z T† = Z`.

Sourced, arXiv:2609.18922: "an ideal input, and a fixed pattern of faults inside the check, X_3 in the 'fold' layers and Y_9 in the 'unfold' layer ... chosen because it is the smallest pattern that passes post-selection with a logical error." Then: "The data error before the exit layer is Q = X_0 X_3 Y_9." Then: "The accepted state is therefore logically wrong: two faults lead to an accepted logical error, hence the fault distance is d_fault = 2." Then: "No detector fires with probability 1/4, and every accepted output carries the logical error."

Unverified: this folder did not load that double-checking circuit, so it did not recompute the probability 1/4. The quoted pair is their pair.

Result, this engine, separate circuit. On one round of Steane extraction the weight-2 set is `Z` on data qubit 0 before the round, and `Z` on the ancilla after the second CNOT of `IIIZZZZ`. All six detectors stay quiet. The data error is `ZIIIIZZ`, which is a logical. The same pair is rejected after the flag in item 10.

Sourced, arXiv:2609.18922: "The escape stage is not included." And: "The escape, grafting and decoding stages are outside the scope of this manuscript." And, Discussion: "Extensions to the escape and grafting stages of cultivation are left for future work."

Sourced, arXiv:2604.27058, abstract: "Clifft enables, to our knowledge, the first exact end-to-end simulation of magic state cultivation including the escape stage, over hundreds of billions of shots."

Unverified: this folder did not run Clifft and did not graft a colour code.

Result, this engine. The smallest graft here grows one qubit to the 3-qubit repetition code by two CNOTs onto fresh `|0>` qubits. Logical `X` becomes `XXX`. Logical `Z` stays `ZII`. New stabilizers are `ZZI` and `ZIZ`. That is not their escape stage.

Next step. Attach the same two channels inside `synth.py` only when a caller asks for a noisy synthesis.

### 7.2 The published d=3 series

Sourced, companion code, URL `https://github.com/kh428/exact_ler_for_msc`, file `data/series/d3_coefficients.csv`, raw rows:

`2,109433/100,32/75,32/75`

`3,203757811/6750,2058319/4500,2009039/4500`

The columns are `k`, `a_k`, `e_k`, `L_k`. So `L_2 = 32/75` and `L_3 = 2009039/4500`. The acceptance coefficients on those rows are `a_0 = 1`, `a_1 = 77/3`, `a_2 = 109433/100`. The variable is `x = p/(1-p)`.

Sourced, arXiv:2609.18922, equation (4): "P_L^{(3)}(x) = (32/75) x^2 + (2009039/4500) x^3 + O(x^4)".

Sourced, same paper, Table 1 caption: "Whole-circuit d=5 results." The displayed acceptance at the third listed point is `0.143955584096`. That row is d=5, not d=3. The paper says the stored values are exact fractions. This note does not reconstruct that fraction from the displayed digits.

Sourced, Appendix G: "on single-threaded C++ code executed on an Apple Macbook (M4)". Then: "The p = 10^{-3} repeat, which reused the cached endpoint tables of Appendix F.2, took 1.710 worker core-hours and the complete degree-ten calculation 9.857 worker core-hours."

Unverified: `python cli.py verify` does not run that contraction. One run of that command on this Windows session finished in 57.79 seconds and printed `AGC QEC FULL SUITE CERTIFICATE PASS`. That duration is the certificate, not 1.710 worker core-hours, and this note names no hardware for the machine that ran it.

Next step. Recompute one modular prime from their `recompute_prime.py` only in a separate checkout, then compare one residue. Do not paste their series into this certificate as if it had been derived here.

### 7.3 Numbers this engine does not claim

Sourced, arXiv:2512.23037, Table I, noise `0.001`: reliable simulation `4.59 × 10^{-9}`, discrepancy factor `7.65`, empirical conjecture `6 × 10^{-10}`.

Sourced, arXiv:2603.14670, abstract: "coherent noise error thresholds, when computed at the circuit level (i.e taking into account full syndrome circuits) for distances up to d=9, are systematically overestimated (by a factor of about 4) by a Pauli-twirling approximation of the noise."

Unverified: this engine did not compute a d=9 threshold and did not reproduce the factor of about 4. The phrase "about 4" is theirs.

Result, this engine. For `U = c I - i s Z` on `|+>`, the coherent `<Y>` is `2 c s`. The Pauli twirl of that rotation is the dephasing channel `c^2 ρ + s^2 Z ρ Z`, whose `<Y>` is 0. The exact gap is `2 c s`. That is the whole claim.

Next step. Keep the symbol `c, s`. Do not fit a threshold.

### 7.4 Direct T simulation, smallest gap

Sourced, arXiv:2409.17595: "Ideally, we would directly simulate the T gate circuits."

Sourced, arXiv:2512.23037: "reliably assessing their validity and performance remains a crucial bottleneck."

Sourced, arXiv:2604.27058, abstract: "These simulations show that escape-stage failures suppress the discrepancy between the true T-gate circuit and its S-proxy at low decoder-gap thresholds, while at high thresholds the full-protocol behavior approaches the larger discrepancy observed in the cultivation stages alone."

The sentence "persists at the full-protocol level" is a paraphrase. The raw sentence says the full-protocol behavior approaches the larger discrepancy at high thresholds. Unverified as a paraphrase.

Result, this engine, one qubit. The noisy state is the Pauli channel after `T`. The true check is `T†` then `X = +`. The proxy check is `S†` then `X = +`. Conditional fidelity to `|T>`:

true check: `1 - p/6 + O(p^2)`

S proxy: `1 + (-5/3 + sqrt(2)) p + O(p^2)`

The ratio of the leading infidelities is `10 - 6 sqrt(2)`. This is not their factor 7, and it is not a full protocol.

Next step. Add one CNOT and one flag to this pair of checks and recompute the same ratio.

### 7.5 Coherent noise, damping, leakage

Result. Coherent noise uses the symbols in item 7.3. Amplitude damping at the single rate `γ = 1/2` is inside `Z[i, 1/sqrt(2)]`, because `sqrt(γ) = sqrt(1-γ) = 1/sqrt(2)`. On `|T>` the two Kraus probabilities are `3/4` and `1/4`.

Blocker. A generic damping rate has `sqrt(1-γ)` outside this ring. Leakage needs a third level. This Pauli engine has no third level, so leakage is not simulated.

Next step. Leave leakage blocked until the state carries an explicit leaked basis vector with exact amplitudes.

### 7.6 Decoders

Sourced, arXiv:2512.07737, abstract: "This combination of requirements remains unmet by a machine-learning decoder, nor by any decoder for promising resource-efficient codes such as the color code."

Unverified: this engine did not train a decoder and did not decode a colour code.

Result, this engine. On the 3-qubit repetition code, the syndrome of one `X` on qubit 0 has two explanations. The weight-1 likelihood minus the weight-2 likelihood is `p - 3 p^2 + 2 p^3`. The linear term is `+p`, so maximum likelihood picks the weight-1 recovery. That is the small case. It is not AlphaQubit 2.

Next step. Keep this likelihood polynomial. Do not add a float network.

### 7.7 One cheaper T

Result, this engine. One accepted injection has leading infidelity `p/6`. Two independent accepted injections have leading infidelity `p/3`. Dropping one of the two injections lowers that coefficient by `1/6`. No larger algorithm is costed.

Next step. Replace the second injection by a Clifford and recompute the same linear coefficient.

### 7.8 Hardware

Literature only. This note makes no hardware claim for the machine that runs `cli.py verify`.

Sourced, arXiv:2609.18922: Apple Macbook (M4), single-threaded C++, 1.710 worker core-hours for the `p = 10^{-3}` repeat, 9.857 worker core-hours for degree ten.

Sourced, arXiv:2512.23037: "We simulate the d=5 noisy cultivation circuit and collect over 200 billion samples, using 16 NVIDIA H800 GPUs within 20 days."

Next step. If a later run records a wall time, label the machine and the command in the same sentence.

### 7.9 Nodal observation

The nodes are the four Pauli branches `I`, `X`, `Y`, `Z` after `T`, each with weight `1-p` or `p/3`. Post-selection is the observation: keep the `+` outcome of `X` after `T†`.

Result, this engine. Acceptance is `1` for `I`, `1/2` for `X`, `1/2` for `Y`, and `0` for `Z`. The observation discards `Z` and keeps `X` and `Y`. An accepted branch is therefore not proof that the pre-check state was `|T>`.

Sourced, arXiv:2609.18922, on their d=3 check, not on this one-qubit gadget: "two faults lead to an accepted logical error, hence the fault distance is d_fault = 2." Their post-selection likewise accepts a logical error. Unverified here, because their circuit was not run.

Next step. Enumerate the two-fault Steane pair of item 7.1 as its own four-outcome channel, still with exact weights.

### 7.10 Question, then a check

Question. On the unflagged Steane extraction, do the two faults in item 7.1 leave every detector quiet and flip the logical?

Answer. Yes. Detectors are `(0,0,0,0,0,0)`. The data Pauli is `ZIIIIZZ`.

Fix. Prepare a flag in `|+>`. Around that ancilla fault, apply CNOT from the flag to the ancilla, let the fault sit between the two CNOTs, and apply the CNOT again. Measure the flag in `X` and postselect on `+`.

Re-verified in `run_extend`. The flag fires, so the same logical no longer passes post-selection. The data error is still the logical if the flag is ignored. The new check is what rejects it.

Next step. Ask the same yes/no question for one more ancilla slot, and add a flag only where the answer is yes.
