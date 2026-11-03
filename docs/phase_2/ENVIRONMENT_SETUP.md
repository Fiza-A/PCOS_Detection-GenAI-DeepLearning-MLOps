# Phase 2 environment setup (no training performed)

## Runtime

- OS: Windows 10, 64-bit AMD64.
- Python: 3.11.15.
- Project environment: `1PCOS_SEM7/.venv`.
- SDV: 1.38.5.
- PyTorch: 2.15.0+cpu (transitive requirement through SDV's CTGAN implementation); `torch.cuda.is_available()` returned `False`.
- pandas 2.3.3; NumPy 2.4.6; SciPy 1.17.1; scikit-learn 1.9.1; PyYAML 6.0.3.

All imports resolved from `.venv/Lib/site-packages`. No system Python packages were installed. CPU-only PyTorch is sufficient for the SDV CTGAN and TVAE imports tested here; no CUDA runtime is available or needed for this environment check.

## Installation issue and resolution

The first `.venv` creation attempt invoked `ensurepip` and failed, leaving no environment-local pip. The first package install then stalled while pip was unpacking PyTorch. Inspection showed the install had populated many dependency directories but had not yielded a usable pip command. The environment was recovered by bootstrapping pip into the existing local environment with the host pip `--python .venv` option, then completing the dependency installation. The available wheel resolved to CPU PyTorch 2.15.0+cpu. The prior attempt did not train models or process dataset partitions.

## Recreate

From the project root, with Python 3.11 available and package-index access:

```powershell
python -m venv --without-pip .venv
python -m pip --python .venv install pip
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

The `--without-pip`/bootstrap sequence is used because `ensurepip` failed for the available Python distribution. The pinned versions in root `requirements.txt` describe the verified environment. On another platform, resolve and verify the platform-specific PyTorch wheel before relying on byte-identical environment recreation.

## Verification

```powershell
.venv\Scripts\python.exe -c "import sys, pandas, numpy, scipy, sklearn, sdv, torch; from sdv.single_table import CTGANSynthesizer, TVAESynthesizer; print(sys.executable); print(pandas.__version__, numpy.__version__, scipy.__version__, sklearn.__version__, sdv.__version__, torch.__version__); print(sdv.__file__, torch.__file__); print('CUDA:', torch.cuda.is_available()); print(CTGANSynthesizer, TVAESynthesizer)"
.venv\Scripts\python.exe -m pip check
```

The recorded verification imported pandas, NumPy, SciPy, scikit-learn, SDV, `CTGANSynthesizer`, and `TVAESynthesizer`; module paths pointed inside this project `.venv`. PyTorch reported `2.15.0+cpu`, and CUDA was unavailable. No generator `.fit()` or `.sample()` method was called.

## Scope guard

This document records environment setup only. No data split, preprocessing fit, CTGAN/TVAE training, synthetic generation, synthetic-data evaluation, or quality-gate result was created.
