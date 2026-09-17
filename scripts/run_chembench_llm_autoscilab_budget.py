#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from run_chembench_comparison import run_llm_pipeline
from autoscilab.benchmarking import (
    atomic_write_json,
    completed_result,
    ensure_run_config,
    file_sha256,
    preflight_openai_endpoint,
    safe_endpoint,
)

ROOT = Path(__file__).parent.parent
DEFAULT_MANIFEST = ROOT / 'configs' / 'noise_studies' / 'chembench_llm_autoscilab_noise27.json'


def _load_manifest(path: Path, limit: int | None = None) -> list[dict]:
    rows = json.loads(path.read_text())
    return rows[:limit] if limit is not None else rows


def _run_one(
    task: dict,
    budget: int,
    args: argparse.Namespace,
    out_dir: Path,
    run_fingerprint: str,
) -> dict:
    load_dotenv(override=False)
    run_dir = out_dir / task['id']
    try:
        result = run_llm_pipeline(
            task['domain'], task['difficulty'], task['law_version'],
            budget=budget, model=args.main_model, noise=0.0,
            use_domain_tags=False, hypothesis_grammar_source='universal',
            strong_model=None, strong_model_calls=0,
            ensemble_mode=False, ensemble_k=5, ensemble_model=args.ensemble_model,
            ensemble_url=args.ensemble_url, ensemble_every=2,
            main_url=args.main_url, max_completion_tokens=4096,
            ensemble_adaptive=True, ensemble_k_max=20,
            ensemble_stability_threshold=0.1, ensemble_confidence_gate=0.7,
            confidence_threshold=1.1, results_dir=run_dir,
            max_experiments_per_iter=args.max_per_iter,
        )
        if result.get('error'):
            raise RuntimeError(str(result['error']))
        row = {
            'task_id': task['id'], 'domain': task['domain'], 'difficulty': task['difficulty'],
            'law_version': task['law_version'], 'budget': budget, 'noise': 0.0,
            'method': 'mei_v5', 'status': 'completed', 'gt_rmsle': result.get('gt_rmsle'),
            'exact_accuracy': 1.0 if result.get('exact') else 0.0,
            'n_experiments': result.get('n_experiments'), 'n_llm_calls': result.get('n_llm_calls'),
            'duration_s': result.get('duration_s'), 'equation': result.get('equation'),
            'termination': result.get('termination'), 'consultant_calls': result.get('consultant_calls'),
            'max_points_per_iter': args.max_per_iter,
            'run_fingerprint': run_fingerprint,
        }
    except Exception as exc:
        row = {
            'task_id': task['id'], 'domain': task['domain'], 'difficulty': task['difficulty'],
            'law_version': task['law_version'], 'budget': budget, 'noise': 0.0,
            'method': 'mei_v5', 'status': 'error', 'gt_rmsle': None, 'exact_accuracy': 0.0,
            'n_experiments': 0, 'n_llm_calls': 0, 'duration_s': 0.0, 'equation': None,
            'termination': None, 'consultant_calls': 0,
            'error': f'{type(exc).__name__}: {exc}\n{traceback.format_exc()}',
            'max_points_per_iter': args.max_per_iter,
            'run_fingerprint': run_fingerprint,
        }
    run_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(run_dir / 'result.json', row)
    target = 'error.json' if row['status'] == 'error' else 'summary.json'
    atomic_write_json(run_dir / target, row)
    return row


def _aggregate(rows: list[dict]) -> dict:
    rmsles = [float(r['gt_rmsle']) for r in rows if isinstance(r.get('gt_rmsle'), (int, float))]
    exacts = [float(r.get('exact_accuracy') or 0.0) for r in rows]
    durations = [float(r['duration_s']) for r in rows if isinstance(r.get('duration_s'), (int, float))]
    return {
        'n_rows': len(rows), 'mean_gt_rmsle': mean(rmsles) if rmsles else None,
        'exact_accuracy': mean(exacts) if exacts else None,
        'mean_duration_s': mean(durations) if durations else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description='Run ChemBench LLM-AutoSciLab budget study on a fixed manifest.')
    parser.add_argument('--manifest', type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument('--budgets', type=int, nargs='+', default=[40, 60, 80])
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--main-model', default=os.environ.get('OPENAI_MODEL', 'gpt-4o-mini'))
    parser.add_argument('--max-per-iter', type=int, default=5)
    parser.add_argument('--main-url', default=os.environ.get('OPENAI_BASE_URL'))
    parser.add_argument('--ensemble-model', default='Qwen/Qwen2.5-7B-Instruct')
    parser.add_argument('--ensemble-url', default='http://localhost:8001/v1')
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--out-dir', type=Path, default=None)
    parser.add_argument('--skip-preflight', action='store_true')
    args = parser.parse_args()

    if args.main_url and not args.skip_preflight:
        api_key = (
            os.environ.get('DEEPINFRA_API_KEY') or os.environ.get('OPENAI_API_KEY')
            if 'deepinfra.com' in args.main_url.lower()
            else os.environ.get('OPENAI_API_KEY', 'local')
        )
        preflight_openai_endpoint(args.main_url, args.main_model, api_key or 'local')

    tasks = _load_manifest(args.manifest, args.limit)
    out_dir = args.out_dir or ROOT / 'results' / 'paper_release_runs' / 'chem'
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    by_budget = {}
    print(f'[ChemBudget] tasks={len(tasks)} workers={args.workers} model={args.main_model} out={out_dir}')
    for budget in args.budgets:
        budget_dir = out_dir / f'b{budget}'
        budget_dir.mkdir(parents=True, exist_ok=True)
        fingerprint = ensure_run_config(
            budget_dir,
            {
                'benchmark': 'chembench',
                'method': 'mei_v5',
                'model': args.main_model,
                'main_url': safe_endpoint(args.main_url),
                'manifest_sha256': file_sha256(args.manifest),
                'budget': budget,
                'noise': 0.0,
                'max_per_iter': args.max_per_iter,
                'max_completion_tokens': 4096,
                'use_domain_tags': False,
                'hypothesis_grammar_source': 'universal',
            },
        )
        result_files = {task['id']: budget_dir / task['id'] / 'result.json' for task in tasks}
        rows_by_id = {
            task['id']: prior
            for task in tasks
            if (prior := completed_result(result_files[task['id']], fingerprint)) is not None
        }
        pending = [task for task in tasks if task['id'] not in rows_by_id]
        print(
            f'[ChemBudget] budget={budget} -> {budget_dir} '
            f'(resumed={len(rows_by_id)} pending={len(pending)})'
        )
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(_run_one, task, budget, args, budget_dir, fingerprint): task
                for task in pending
            }
            for fut in as_completed(futures):
                task = futures[fut]
                try:
                    row = fut.result()
                except Exception as exc:
                    tb = traceback.format_exc()
                    row = {
                        'task_id': task['id'], 'domain': task['domain'], 'difficulty': task['difficulty'],
                        'law_version': task['law_version'], 'budget': budget, 'noise': 0.0,
                        'method': 'mei_v5', 'status': 'error', 'gt_rmsle': None, 'exact_accuracy': 0.0,
                        'n_experiments': 0, 'n_llm_calls': 0, 'duration_s': 0.0, 'equation': None,
                        'termination': None, 'consultant_calls': 0, 'error': f'{type(exc).__name__}: {exc}\n{tb}',
                        'run_fingerprint': fingerprint,
                    }
                    err_dir = budget_dir / task['id']
                    err_dir.mkdir(parents=True, exist_ok=True)
                    atomic_write_json(err_dir / 'error.json', row)
                    atomic_write_json(err_dir / 'result.json', row)
                rows_by_id[task['id']] = row
                rows = list(rows_by_id.values())
                rows.sort(key=lambda r: (r['domain'], r['difficulty'], r['law_version']))
                atomic_write_json(budget_dir / 'summary.json', rows)
                current_agg = _aggregate([r for r in rows if r.get('status') != 'error'])
                current_agg['n_errors'] = sum(r.get('status') == 'error' for r in rows)
                atomic_write_json(budget_dir / 'aggregate.json', current_agg)
                print(f"[ChemBudget] {task['id']} budget={budget}: {row['status']}")
        rows = list(rows_by_id.values())
        rows.sort(key=lambda r: (r['domain'], r['difficulty'], r['law_version']))
        atomic_write_json(budget_dir / 'summary.json', rows)
        agg = _aggregate([r for r in rows if r.get('status') != 'error'])
        agg['n_errors'] = sum(r.get('status') == 'error' for r in rows)
        by_budget[str(budget)] = agg
        atomic_write_json(budget_dir / 'aggregate.json', agg)
        all_rows.extend(rows)

    root_summary = {
        'benchmark': 'chembench', 'method': 'mei_v5', 'model': args.main_model,
        'manifest': str(args.manifest), 'budgets': args.budgets, 'workers': args.workers,
        'by_budget': by_budget,
    }
    atomic_write_json(out_dir / 'summary.json', all_rows)
    atomic_write_json(out_dir / 'aggregate.json', root_summary)
    print(f'[ChemBudget] wrote {out_dir}')


if __name__ == '__main__':
    main()
