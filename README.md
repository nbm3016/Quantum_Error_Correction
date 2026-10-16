# Quantum error correction

Exact QEC engine. It is not the AGC ledger and it does not lock anything.

Three folders:

| Folder | Role |
|---|---|
| `engine/` | The exact calculator. Data is the seven-qubit address. Checks are the seven extractors. The syndrome is their commutation record. |
| `record/` | Circuits and the outputs of runs that were actually made. |
| `frame/` | The three-node face this work is not allowed to overwrite. |

Run the certificate from `engine/`:

```bash
python cli.py verify
python cli.py map
```

`B[2] = 32/75` is open until a weight-2 pair sum is computed with `run_statevector` and `score_sv`.
