# QEC map

Writing this page does not lock. Counts stay `continuous_knobs = 0`, `n_LOCK = 35`, `n_promoted = 0`, `n_OPEN = 15`.

The data register is the seven-qubit address `(s_M, s_J, s_D, s_B, s_A, s_c, s_Sigma)`. The checks are the seven extractors: six twin products of the gamma generators, then `Gamma_8`. A syndrome bit is the `F_2` commutation of a fault Pauli with one extractor. That record says which address signs flipped. It is not an operator added to the slow face.

The slow face stays `M`, `B_{11}`, and `B_{13}`, with product `-i I_4`. The face is not the metric. Those three operators are not renamed data, check, and syndrome.

Bin: `CONSISTENT`. Baseline written: no. Lock changed: no. `B[2] = 32/75` remains uncomputed.
