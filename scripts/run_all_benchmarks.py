#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
REPRESENTATIVE_CHEM_MANIFEST = (
    ROOT / "configs" / "representative_subsets" / "chembench_representative14.json"
)
REPRESENTATIVE_GRN_EXAMPLES = (
    ROOT / "configs" / "representative_subsets" / "grnbench_representative5.json"
)


def _runner(name: str) -> Path:
    mapping = {
        "newton": ROOT / "scripts" / "run_newton_llm_autoscilab_budget.py",
        "chem": ROOT / "scripts" / "run_chembench_llm_autoscilab_budget.py",
        "grn": ROOT / "scripts" / "run_grn_prompt_budget.py",
    }
    return mapping[name]


def _append_common_args(
    cmd: list[str],
    *,
    workers: int,
    limit: int | None,
    out_dir: Path | None,
) -> list[str]:
    cmd.extend(["--workers", str(workers)])
    if limit is not None:
        cmd.extend(["--limit", str(limit)])
    if out_dir is not None:
        cmd.extend(["--out-dir", str(out_dir)])
    return cmd


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convenience launcher for the paper release benchmarks."
    )
    parser.add_argument(
        "--benchmark",
        choices=["newton", "chem", "grn", "all"],
        default="all",
        help="Which benchmark family to run.",
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the exact benchmark commands without starting any work.",
    )
    parser.add_argument(
        "--skip-preflight",
        action="store_true",
        help="Skip the OpenAI-compatible /models reachability and model-ID check.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Use one model for every selected benchmark (default: OPENAI_MODEL).",
    )
    parser.add_argument(
        "--main-url",
        default=None,
        help="OpenAI-compatible endpoint for every selected benchmark (default: OPENAI_BASE_URL).",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=ROOT / "results" / "paper_release_runs",
        help="Root output directory for generated result folders.",
    )
    parser.add_argument(
        "--representative-subset",
        action="store_true",
        help=(
            "Use the compact, coverage-balanced ChemBench or GRNBench manifest. "
            "Supported with --benchmark chem or --benchmark grn."
        ),
    )
    parser.add_argument("--newton-model", default=None)
    parser.add_argument("--newton-main-url", default=None)
    parser.add_argument("--newton-pysr-iters", type=int, default=800)
    parser.add_argument("--newton-budgets", type=int, nargs="+", default=[10, 20, 50])
    parser.add_argument("--newton-manifest", type=Path, default=None)
    parser.add_argument("--chem-main-model", default=None)
    parser.add_argument("--chem-main-url", default=None)
    parser.add_argument("--chem-ensemble-model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--chem-ensemble-url", default="http://localhost:8001/v1")
    parser.add_argument("--chem-max-per-iter", type=int, default=5)
    parser.add_argument("--chem-budgets", type=int, nargs="+", default=[40, 60, 80])
    parser.add_argument("--chem-manifest", type=Path, default=None)
    parser.add_argument("--grn-main-model", default=None)
    parser.add_argument("--grn-main-url", default=None)
    parser.add_argument("--grn-max-per-iter", type=int, default=5)
    parser.add_argument("--grn-budgets", type=int, nargs="+", default=[10, 20, 50])
    parser.add_argument("--grn-examples-file", type=Path, default=None)
    args = parser.parse_args()

    if args.workers < 1:
        parser.error("--workers must be at least 1")
    if args.representative_subset and args.benchmark not in {"chem", "grn"}:
        parser.error(
            "--representative-subset is supported only with --benchmark chem or grn"
        )
    if args.representative_subset and (
        args.chem_manifest is not None or args.grn_examples_file is not None
    ):
        parser.error(
            "--representative-subset cannot be combined with an explicit Chem/GRN manifest"
        )

    default_model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    default_url = os.environ.get("OPENAI_BASE_URL")

    if not args.dry_run:
        args.out_root.mkdir(parents=True, exist_ok=True)

    jobs: list[tuple[str, list[str]]] = []

    if args.benchmark in {"newton", "all"}:
        cmd = [sys.executable, str(_runner("newton"))]
        cmd.extend(["--model", args.newton_model or args.model or default_model])
        cmd.extend(["--budgets", *map(str, args.newton_budgets)])
        cmd.extend(["--pysr-iters", str(args.newton_pysr_iters)])
        newton_url = args.newton_main_url or args.main_url or default_url
        if newton_url:
            cmd.extend(["--main-url", newton_url])
        if args.newton_manifest:
            cmd.extend(["--manifest", str(args.newton_manifest)])
        if args.skip_preflight:
            cmd.append("--skip-preflight")
        _append_common_args(
            cmd,
            workers=args.workers,
            limit=args.limit,
            out_dir=args.out_root / "newton",
        )
        jobs.append(("newton", cmd))

    if args.benchmark in {"chem", "all"}:
        cmd = [sys.executable, str(_runner("chem"))]
        cmd.extend(["--main-model", args.chem_main_model or args.model or default_model])
        cmd.extend(["--budgets", *map(str, args.chem_budgets)])
        cmd.extend(["--max-per-iter", str(args.chem_max_per_iter)])
        cmd.extend(["--ensemble-model", args.chem_ensemble_model])
        cmd.extend(["--ensemble-url", args.chem_ensemble_url])
        chem_url = args.chem_main_url or args.main_url or default_url
        if chem_url:
            cmd.extend(["--main-url", chem_url])
        chem_manifest = (
            REPRESENTATIVE_CHEM_MANIFEST
            if args.representative_subset
            else args.chem_manifest
        )
        if chem_manifest:
            cmd.extend(["--manifest", str(chem_manifest)])
        if args.skip_preflight:
            cmd.append("--skip-preflight")
        _append_common_args(
            cmd,
            workers=args.workers,
            limit=args.limit,
            out_dir=args.out_root / "chem",
        )
        jobs.append(("chem", cmd))

    if args.benchmark in {"grn", "all"}:
        cmd = [sys.executable, str(_runner("grn"))]
        cmd.extend(["--main-model", args.grn_main_model or args.model or default_model])
        cmd.extend(["--budgets", *map(str, args.grn_budgets)])
        cmd.extend(["--max-per-iter", str(args.grn_max_per_iter)])
        grn_url = args.grn_main_url or args.main_url or default_url
        if grn_url:
            cmd.extend(["--main-url", grn_url])
        grn_examples_file = (
            REPRESENTATIVE_GRN_EXAMPLES
            if args.representative_subset
            else args.grn_examples_file
        )
        if grn_examples_file:
            cmd.extend(["--examples-file", str(grn_examples_file)])
        if args.skip_preflight:
            cmd.append("--skip-preflight")
        _append_common_args(
            cmd,
            workers=args.workers,
            limit=args.limit,
            out_dir=args.out_root / "grn",
        )
        jobs.append(("grn", cmd))

    for name, cmd in jobs:
        print(f"[paper-release] running {name}: {' '.join(cmd)}")
        if not args.dry_run:
            subprocess.run(cmd, cwd=str(ROOT), check=True)


if __name__ == "__main__":
    main()
