#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).parent.parent))

from autoscilab.benchmarking import atomic_write_json

ROOT = Path(__file__).parent.parent
DEFAULT_EXAMPLES = ROOT / 'configs' / 'noise_studies' / 'grnbench_prompt_noise18.json'
EXPERIMENT_RUNNER = ROOT / 'scripts' / 'run_llm_autoscilab_grn_graph_examples.py'


def _load_examples(path: Path, limit: int | None = None) -> list[dict]:
    rows = json.loads(path.read_text())
    return rows[:limit] if limit is not None else rows


def _aggregate(rows: list[dict]) -> dict:
    completed = [r for r in rows if r.get('status') == 'completed']
    f1s, exacts, signs, durations = [], [], [], []
    for row in completed:
        ev = row.get('final_graph_eval') or {}
        if isinstance(ev.get('edge_f1'), (int, float)):
            f1s.append(float(ev['edge_f1']))
        if isinstance(ev.get('exact_graph_accuracy'), (int, float)):
            exacts.append(float(ev['exact_graph_accuracy']))
        if isinstance(ev.get('sign_accuracy'), (int, float)):
            signs.append(float(ev['sign_accuracy']))
        if isinstance(row.get('duration_s'), (int, float)):
            durations.append(float(row['duration_s']))
    return {
        'n_rows': len(rows), 'n_completed': len(completed), 'n_errors': sum(r.get('status') == 'error' for r in rows),
        'edge_f1': mean(f1s) if f1s else None, 'exact_graph_accuracy': mean(exacts) if exacts else None,
        'sign_accuracy': mean(signs) if signs else None, 'mean_duration_s': mean(durations) if durations else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description='Run GRN prompt-ablation budget study on a fixed manifest.')
    parser.add_argument('--examples-file', type=Path, default=DEFAULT_EXAMPLES)
    parser.add_argument('--budgets', type=int, nargs='+', default=[10, 20, 50])
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--main-model', default=os.environ.get('OPENAI_MODEL', 'gpt-4o-mini'))
    parser.add_argument('--main-url', default=os.environ.get('OPENAI_BASE_URL'))
    parser.add_argument('--max-per-iter', type=int, default=5)
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--out-dir', type=Path, default=None)
    parser.add_argument('--skip-preflight', action='store_true')
    args = parser.parse_args()

    examples = _load_examples(args.examples_file, args.limit)
    out_dir = args.out_dir or ROOT / 'results' / 'paper_release_runs' / 'grn'
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    by_budget = {}
    print(f'[GRNBudget] tasks={len(examples)} workers={args.workers} model={args.main_model} out={out_dir}')
    for budget in args.budgets:
        budget_dir = out_dir / f'b{budget}'
        budget_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable, str(EXPERIMENT_RUNNER), '--examples-file', str(args.examples_file),
            '--experiment-mode', 'prompt', '--workers', str(args.workers), '--main-model', args.main_model,
            '--budget', str(budget), '--noise', '0.0', '--max-per-iter', str(args.max_per_iter),
            '--out-dir', str(budget_dir),
        ]
        if args.main_url:
            cmd.extend(['--main-url', args.main_url])
        if args.limit is not None:
            cmd.extend(['--limit', str(args.limit)])
        if args.skip_preflight:
            cmd.append('--skip-preflight')
        print(f'[GRNBudget] budget={budget} -> {budget_dir}')
        subprocess.run(cmd, cwd=str(ROOT), check=True)
        rows = json.loads((budget_dir / 'summary.json').read_text())
        for row in rows:
            row['budget'] = budget
            row['noise'] = 0.0
            row['method'] = 'llm_autoscilab_grn'
        atomic_write_json(budget_dir / 'summary.json', rows)
        agg = _aggregate(rows)
        by_budget[str(budget)] = agg
        atomic_write_json(budget_dir / 'aggregate.json', agg)
        all_rows.extend(rows)

    root_summary = {
        'benchmark': 'grnbench', 'method': 'llm_autoscilab_grn', 'model': args.main_model,
        'manifest': str(args.examples_file), 'budgets': args.budgets, 'workers': args.workers,
        'by_budget': by_budget,
    }
    atomic_write_json(out_dir / 'summary.json', all_rows)
    atomic_write_json(out_dir / 'aggregate.json', root_summary)
    print(f'[GRNBudget] wrote {out_dir}')


if __name__ == '__main__':
    main()
