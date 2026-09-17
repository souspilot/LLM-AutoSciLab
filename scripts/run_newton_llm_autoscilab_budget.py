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

from run_paper_sweep import get_api_key, run_one as run_newton_one
from autoscilab.benchmarking import (
    atomic_write_json,
    completed_result,
    ensure_run_config,
    file_sha256,
    preflight_openai_endpoint,
    safe_endpoint,
)

ROOT = Path(__file__).parent.parent
DEFAULT_MANIFEST = ROOT / 'configs' / 'noise_studies' / 'newtonbench_llm_autoscilab_noise108.json'


def _load_manifest(path: Path, limit: int | None = None) -> list[dict]:
    rows = json.loads(path.read_text())
    return rows[:limit] if limit is not None else rows


def _normalize(row: dict, task: dict, budget: int) -> dict:
    return {
        'task_id': task['id'],
        'domain': task['domain'],
        'difficulty': task['difficulty'],
        'law_version': task['law_version'],
        'seed': task['seed'],
        'budget': budget,
        'noise': 0.0,
        'method': 'llm_autoscilab_newton',
        'status': row.get('status'),
        'gt_rmsle': row.get('gt_rmsle'),
        'exact_accuracy': row.get('exact_accuracy'),
        'oracle_calls': row.get('oracle_calls'),
        'llm_calls': row.get('llm_calls'),
        'duration_s': row.get('duration_s'),
        'best_equation': row.get('best_equation'),
        'law_str': row.get('law_str'),
        'error': row.get('error'),
        'result_dir': row.get('results_dir') or row.get('result_dir'),
    }


def _aggregate(rows: list[dict]) -> dict:
    completed = [r for r in rows if r.get('status') and r.get('status') != 'error']
    rmsles = [float(r['gt_rmsle']) for r in completed if isinstance(r.get('gt_rmsle'), (int, float))]
    exacts = [float(r.get('exact_accuracy') or 0.0) for r in completed]
    durations = [float(r['duration_s']) for r in completed if isinstance(r.get('duration_s'), (int, float))]
    return {
        'n_rows': len(rows),
        'n_completed': len(completed),
        'n_errors': sum(r.get('status') == 'error' for r in rows),
        'mean_gt_rmsle': mean(rmsles) if rmsles else None,
        'exact_accuracy': mean(exacts) if exacts else None,
        'mean_duration_s': mean(durations) if durations else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description='Run NewtonBench LLM-AutoSciLab budget study on a fixed manifest.')
    parser.add_argument('--manifest', type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument('--budgets', type=int, nargs='+', default=[10, 20, 50])
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--model', default=os.environ.get('OPENAI_MODEL', 'gpt-4o-mini'))
    parser.add_argument('--main-url', default=os.environ.get('OPENAI_BASE_URL'))
    parser.add_argument('--pysr-iters', type=int, default=800)
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--out-dir', type=Path, default=None)
    parser.add_argument('--skip-preflight', action='store_true')
    args = parser.parse_args()

    if args.main_url and not args.skip_preflight:
        preflight_openai_endpoint(args.main_url, args.model, get_api_key(args.model, args.main_url))

    tasks = _load_manifest(args.manifest, args.limit)
    out_dir = args.out_dir or ROOT / 'results' / 'paper_release_runs' / 'newton'
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict] = []
    by_budget: dict[str, dict] = {}

    print(f'[NewtonBudget] tasks={len(tasks)} workers={args.workers} model={args.model} out={out_dir}')

    for budget in args.budgets:
        budget_dir = out_dir / f'b{budget}'
        budget_dir.mkdir(parents=True, exist_ok=True)
        max_per_iter = max(1, budget // 5)
        eq_every = max_per_iter
        fingerprint = ensure_run_config(
            budget_dir,
            {
                'benchmark': 'newtonbench',
                'method': 'llm_autoscilab_newton',
                'model': args.model,
                'main_url': safe_endpoint(args.main_url),
                'manifest_sha256': file_sha256(args.manifest),
                'budget': budget,
                'noise': 0.0,
                'max_per_iter': max_per_iter,
                'eq_every': eq_every,
                'pysr_iters': args.pysr_iters,
            },
        )
        result_files = {
            task['id']: budget_dir / 'tasks' / task['id'] / 'result.json'
            for task in tasks
        }
        raw_rows = {
            task['id']: prior
            for task in tasks
            if (prior := completed_result(result_files[task['id']], fingerprint)) is not None
        }
        pending = [task for task in tasks if task['id'] not in raw_rows]
        print(
            f'[NewtonBudget] budget={budget} -> {budget_dir} '
            f'(resumed={len(raw_rows)} pending={len(pending)})'
        )
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(
                    run_newton_one,
                    task['domain'],
                    task['difficulty'],
                    task['law_version'],
                    task['seed'],
                    args.model,
                    budget,
                    max_per_iter,
                    eq_every,
                    0.0,
                    args.pysr_iters,
                    budget_dir / 'tasks' / task['id'],
                    args.main_url,
                    result_files[task['id']],
                    fingerprint,
                ): task
                for task in pending
            }
            for fut in as_completed(futures):
                task = futures[fut]
                try:
                    raw = fut.result()
                except Exception as exc:
                    raw = {
                        'status': 'error', 'gt_rmsle': None, 'exact_accuracy': None,
                        'oracle_calls': 0, 'llm_calls': 0, 'duration_s': 0,
                        'best_equation': None, 'law_str': None,
                        'error': f'{type(exc).__name__}: {exc}\n{traceback.format_exc()}', 'result_dir': None,
                        'run_fingerprint': fingerprint,
                    }
                    atomic_write_json(result_files[task['id']], raw)
                raw_rows[task['id']] = raw
                rows = [_normalize(raw_rows[t['id']], t, budget) for t in tasks if t['id'] in raw_rows]
                rows.sort(key=lambda r: (r['domain'], r['difficulty'], r['law_version'], r['seed']))
                atomic_write_json(budget_dir / 'summary.json', rows)
                atomic_write_json(budget_dir / 'aggregate.json', _aggregate(rows))
                row = _normalize(raw, task, budget)
                print(f"[NewtonBudget] {task['id']} budget={budget}: {row['status']}")
        rows = [_normalize(raw_rows[t['id']], t, budget) for t in tasks if t['id'] in raw_rows]
        rows.sort(key=lambda r: (r['domain'], r['difficulty'], r['law_version'], r['seed']))
        atomic_write_json(budget_dir / 'summary.json', rows)
        agg = _aggregate(rows)
        by_budget[str(budget)] = agg
        atomic_write_json(budget_dir / 'aggregate.json', agg)
        all_rows.extend(rows)

    root_summary = {
        'benchmark': 'newtonbench', 'method': 'llm_autoscilab_newton', 'model': args.model,
        'manifest': str(args.manifest), 'budgets': args.budgets, 'workers': args.workers,
        'by_budget': by_budget,
    }
    atomic_write_json(out_dir / 'summary.json', all_rows)
    atomic_write_json(out_dir / 'aggregate.json', root_summary)
    print(f'[NewtonBudget] wrote {out_dir}')


if __name__ == '__main__':
    main()
