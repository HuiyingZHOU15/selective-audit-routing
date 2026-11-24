from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


EXPECTED_SPLITS = {"train": 3982, "validation": 992, "calibration": 666, "test": 967}


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve()
    parser = argparse.ArgumentParser(
        description="Verify frozen inputs and reproduce manuscript figures, LaTeX, and PDF."
    )
    parser.add_argument("--project-root", type=Path, default=here.parents[1])
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--skip-pdf", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def required_inputs(root: Path) -> list[Path]:
    return [
        root / "PAPER.html",
        root / "data/processed/model_dataset_v2.csv",
        root / "configs/adaptation_evaluation_phase1.freeze.json",
        root / "configs/adaptation_phase5_official_matrix_v1_1.csv",
        root / "configs/phase6_retrospective_evaluation_v1_1.freeze.json",
        root
        / "results/phase6b_retrospective_test_aggregation_v1_1/phase6_test_contrasts_v1_1.csv",
        root
        / "results/phase6_posthoc_near_duplicate_sensitivity_v1_1/phase6_test_contrast_near_duplicate_sensitivity_v1_1.csv",
        root
        / "results/phase6b_retrospective_test_final_v1_1/phase6_test_frequency_tier_summary_v1_1.csv",
        root
        / "results/phase6b_retrospective_test_aggregation_v1_1/phase6_test_condition_summary_v1_1.csv",
        root
        / "results/phase6b_retrospective_test_aggregation_v1_1/phase6_test_seed_results_v1_1.csv",
        root
        / "results/phase6b_retrospective_test_final_v1_1/phase6_test_per_class_seed_results_v1_1.csv",
        root
        / "results/phase6b_retrospective_test_final_v1_1/phase6_test_frequency_tier_seed_results_v1_1.csv",
        root / "scripts/build_isafm_revision_figures.py",
        root / "scripts/build_isafm_submission_latex.py",
        root
        / "submission/isafm_major_revision_2026-11-21/03_figure_files/graphical_abstract_800dpi.tiff",
        root
        / "submission/isafm_major_revision_2026-11-21/03_figure_files/accuracy_efficiency_tradeoff_800dpi.tiff",
        root
        / "submission/isafm_major_revision_2026-11-21/03_figure_files/exploratory_lexical_masking_800dpi.tiff",
    ]


def validate_inputs(root: Path, paths: list[Path]) -> dict[str, str]:
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing required inputs:\n" + "\n".join(missing))

    import pandas as pd

    dataset = pd.read_csv(root / "data/processed/model_dataset_v2.csv", usecols=["split"])
    observed = {str(key): int(value) for key, value in dataset["split"].value_counts().items()}
    if len(dataset) != 6607 or observed != EXPECTED_SPLITS:
        raise RuntimeError(
            f"Frozen dataset mismatch: rows={len(dataset)}, splits={observed}; "
            f"expected rows=6607, splits={EXPECTED_SPLITS}."
        )

    contrasts = pd.read_csv(
        root
        / "results/phase6b_retrospective_test_aggregation_v1_1/phase6_test_contrasts_v1_1.csv"
    )
    if len(contrasts) != 5:
        raise RuntimeError(f"Expected five prespecified contrasts, found {len(contrasts)}.")

    return {str(path.relative_to(root)): sha256(path) for path in paths}


def load_figure_module(root: Path):
    source = root / "scripts/build_isafm_revision_figures.py"
    spec = importlib.util.spec_from_file_location("isafm_reproduction_figures", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load figure script: {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def copy_static_figures(root: Path, figure_output: Path) -> None:
    source = root / "submission/isafm_major_revision_2026-11-21/03_figure_files"
    figure_output.mkdir(parents=True, exist_ok=True)
    for stem in (
        "graphical_abstract",
        "accuracy_efficiency_tradeoff",
        "exploratory_lexical_masking",
    ):
        for suffix in ("_800dpi.tiff", ".svg"):
            shutil.copy2(source / f"{stem}{suffix}", figure_output / f"{stem}{suffix}")


def generate_figures(root: Path, figure_output: Path) -> None:
    copy_static_figures(root, figure_output)
    module = load_figure_module(root)
    module.OUT = figure_output
    module.main()


def stage_manuscript(root: Path, output_root: Path, figure_output: Path) -> Path:
    stage = output_root / "_stage"
    stage_figures = (
        stage / "submission/isafm_major_revision_2026-11-21/03_figure_files"
    )
    stage_figures.mkdir(parents=True, exist_ok=True)
    shutil.copy2(root / "PAPER.html", stage / "PAPER.html")
    for path in figure_output.iterdir():
        if path.is_file():
            shutil.copy2(path, stage_figures / path.name)
    return stage


def build_latex(root: Path, stage: Path, latex_output: Path) -> None:
    latex_output.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(root / "scripts/build_isafm_submission_latex.py"),
        str(stage),
        str(latex_output),
    ]
    subprocess.run(command, check=True)


def compile_pdf(latex_output: Path) -> bool:
    pdflatex = shutil.which("pdflatex")
    if pdflatex is None:
        print("pdflatex not found; LaTeX source was generated but PDF compilation was skipped.")
        return False
    command = [pdflatex, "-interaction=nonstopmode", "-halt-on-error", "main_anonymous.tex"]
    subprocess.run(command, cwd=latex_output, check=True)
    subprocess.run(command, cwd=latex_output, check=True)
    return True


def output_hashes(output_root: Path) -> dict[str, str]:
    hashes = {}
    for directory in (output_root / "figures", output_root / "latex"):
        if not directory.exists():
            continue
        for path in sorted(directory.rglob("*")):
            if path.is_file():
                hashes[str(path.relative_to(output_root))] = sha256(path)
    return hashes


def main() -> int:
    args = parse_args()
    root = args.project_root.resolve()
    output_root = (
        args.output_root.resolve()
        if args.output_root
        else root / "reproducibility/output"
    )
    inputs = required_inputs(root)
    input_hashes = validate_inputs(root, inputs)
    print(f"Verified {len(inputs)} frozen inputs and the 6,607-row split profile.")
    if args.verify_only:
        return 0

    figure_output = output_root / "figures"
    latex_output = output_root / "latex"
    generate_figures(root, figure_output)
    stage = stage_manuscript(root, output_root, figure_output)
    build_latex(root, stage, latex_output)
    pdf_compiled = False if args.skip_pdf else compile_pdf(latex_output)

    manifest = {
        "schema_version": "1.0",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "project_root": str(root),
        "dataset_rows": 6607,
        "expected_splits": EXPECTED_SPLITS,
        "pdf_compiled": pdf_compiled,
        "inputs_sha256": input_hashes,
        "outputs_sha256": output_hashes(output_root),
    }
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "reproduction_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Reproduction completed: {output_root}")
    print(f"Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

