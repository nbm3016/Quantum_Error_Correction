# Quantum error correction

Private record of the exact QEC engine. It is not the AGC ledger and it does not lock anything.

Three folders:

| Folder | Role |
|---|---|
| `engine/` | The exact calculator. `Z[i, 1/sqrt(2)]`. No floats. |
| `record/` | Circuits and the outputs of runs that were actually made. |
| `frame/` | The three-node face this work is not allowed to overwrite. |

Run the certificate from `engine/`:

```bash
python cli.py verify
```

`B[2] = 32/75` is open until a weight-2 pair sum is computed with `run_statevector` and `score_sv`.
