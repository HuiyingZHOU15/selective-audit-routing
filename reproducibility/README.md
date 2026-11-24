# Reproducibility package

This directory exposes the existing Phase 1--6 implementation through two reproducibility tracks. It does not introduce a second training or evaluation implementation.

## Track A: reproduce reported artifacts without GPU training

This track validates the frozen inputs, rebuilds the three data-derived figures, regenerates the anonymous LaTeX manuscript, and compiles the PDF when `pdflatex` is available.

```powershell
cd C:\agent\selective-audit-routing
py -m venv .venv-reproduce
.\.venv-reproduce\Scripts\python.exe -m pip install -r reproducibility\requirements-results.txt
.\.venv-reproduce\Scripts\python.exe reproducibility\reproduce_results.py
```

PowerShell shortcut:

```powershell
.\reproducibility\reproduce_results.ps1
```

Outputs are written to `reproducibility/output/`:

- `figures/`: editable SVG and opaque 800-dpi TIFF figures;
- `latex/`: generated LaTeX source, review JPEGs, and PDF when TeX is installed;
- `reproduction_manifest.json`: SHA-256 hashes for every required input and generated artifact.

Use `--verify-only` to check inputs and hashes without generating files. Use `--skip-pdf` when TeX is unavailable.

## Track B: rerun the official training matrix

Install `requirements-training.txt` in a CUDA-compatible environment, then run:

```powershell
.\reproducibility\run_official_training.ps1 -RunsRoot D:\pcaob_phase5_runs
```

To run one official seed only:

```powershell
.\reproducibility\run_official_training.ps1 -RunsRoot D:\pcaob_phase5_runs -Seed 20261114
```

The wrapper invokes the existing `src/run_phase5_official.py` with the frozen 24-run manifest. The manifest itself prevents test or calibration loading during training.

After checkpoints are available, validation or retrospective test predictions can be regenerated with:

```powershell
.\reproducibility\run_phase6_evaluation.ps1 `
  -OfficialRoot D:\pcaob_phase5_runs `
  -OutputRoot D:\pcaob_phase6_validation

.\reproducibility\run_phase6_evaluation.ps1 `
  -OfficialRoot D:\pcaob_phase5_runs `
  -OutputRoot D:\pcaob_phase6_test `
  -Partition test `
  -ConfirmRetrospectiveTest
```

The explicit confirmation flag preserves the manuscript's evidential status: the test set is retrospective, not untouched.

## Canonical implementation map

| Stage | Existing implementation |
|---|---|
| Frozen experimental matrix | `configs/adaptation_phase5_official_matrix_v1_1.csv` |
| Training scheduler | `src/run_phase5_official.py` |
| Unified model trainer | `src/train_adaptation_phase1.py` |
| Phase 6 evaluation | `src/evaluate_adaptation_phase6_v1_1.py` |
| Report-level bootstrap aggregation | `src/aggregate_phase6_results_v1_1.py` |
| Final Phase 6 tables | `src/finalize_phase6b_v1_1.py` |
| Near-duplicate sensitivity | `scripts/run_phase6_posthoc_sensitivity_v1_1.py` |
| Submission figures | `scripts/build_isafm_revision_figures.py` |
| HTML-to-LaTeX manuscript | `scripts/build_isafm_submission_latex.py` |

## Scope and data status

- The frozen modeling table contains 6,607 findings grouped by `report_id` into train, validation, calibration, and test partitions.
- The 890-finding result is a retrospective clean-test evaluation under the original pipeline.
- The 863-finding result is a post-hoc near-duplicate sensitivity analysis.
- Masking is exploratory and is not a same-model causal intervention.
- Checkpoints and Hugging Face caches are intentionally excluded because they are large and machine-specific.

## Before public release

Replace local paths in examples with repository-relative instructions, add the repository license and DOI, confirm whether the source PCAOB CSV may be redistributed, and complete the author-approved data-availability statement.

