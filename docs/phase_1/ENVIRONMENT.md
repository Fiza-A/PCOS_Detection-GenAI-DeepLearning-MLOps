# Environment audit

Inspected using the available `python` executable (Python 3.11.15). Package presence is reported for that executable, not assumed for any future virtual environment.

| Package | Available | Planned phase |
|---|---|---|
| pandas | Yes | Phase 1 audit and data handling |
| NumPy | Yes | Phase 1 audit |
| scikit-learn | Yes | Later split/modeling; not used for fitting in Phase 1 |
| SDV | No | Phase 2 candidate, install/version check then |
| CTGAN | No | Phase 2 via selected stable package |
| PyTorch | No | Phase 4 only if selected |
| TensorFlow | No | Not selected/needed currently |
| SHAP | No | Phase 5 |
| MLflow | No | Phase 6 |
| Prefect | No | Phase 7 |
| FastAPI | Yes | Phase 8 |
| Evidently | No | Phase 10 |
| prometheus-client | No | Phase 10 |
| pytest | No | Requested test runner unavailable in inspected interpreter |

No packages were installed. The root `requirements.txt` lists only pandas and NumPy for this audit. The requested `.venv` does not exist in the active workspace.
