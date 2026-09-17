# LLM-AutoSciLab 

This release contains the minimal code needed to run the paper's three benchmark families with our method:

- `NewtonBench`
- `ActiveSciBench-Chem`
- `ActiveSciBench-GRN`

The packaged tree includes:

- `autoscilab/`: method, loops, acquisition, and oracle implementations
- `configs/`: fixed manifests used by the benchmark runners
- `newtonbench_vendor/`: minimal vendored Newton benchmark modules required by the Newton oracle
- `scripts/`: benchmark entry points plus a convenience launcher

## Environment

Use Python 3.11 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Required environment variables:

- `OPENAI_API_KEY`: required for the default `gpt-4o-mini` runs
- `TOGETHER_API_KEY`: only required if you use Together-hosted non-OpenAI models

For an OpenAI-compatible local server (vLLM, SGLang, llama.cpp, etc.), use:

```bash
export OPENAI_API_KEY=dummy
export OPENAI_BASE_URL=http://127.0.0.1:8000/v1
export OPENAI_MODEL='the/model-id-reported-by-your-server'
```

[`env.local.example.sh`](env.local.example.sh) contains the same setup plus the
CUDA, Triton, JAX, and proxy settings used on a module-based GPU cluster.

`OPENAI_MODEL` is optional if the model is supplied with `--model` (all benchmark
families) or the family-specific model flags. The key may be a dummy value for a
server that does not authenticate.

Optional overrides:

- `--main-url`: point the main model at an OpenAI-compatible local or remote endpoint
- `--ensemble-url`: ChemBench ensemble endpoint, default `http://localhost:8001/v1`

## Entry Points

The simplest interface is the wrapper:

```bash
python scripts/run_all_benchmarks.py --benchmark all --workers 1 --limit 1
```

That runs a small manifest slice from all three benchmark families and writes results under:

```text
results/paper_release_runs/
```

Run a single family:

```bash
python scripts/run_all_benchmarks.py --benchmark newton --workers 1
python scripts/run_all_benchmarks.py --benchmark chem --workers 1
python scripts/run_all_benchmarks.py --benchmark grn --workers 1
```

## Local reproduction and resume

After starting the local model server and sourcing your cluster environment,
run the included author configurations with:

```bash
python scripts/run_all_benchmarks.py \
  --benchmark all \
  --model "$OPENAI_MODEL" \
  --workers 1
```

The defaults use the release manifests and settings: 108 Newton tasks at budgets
10/20/50, 27 Chem tasks at budgets 40/60/80, and 18 GRN tasks at budgets
10/20/50. The "budget" here is the experiment/oracle-call setting used by this
paper release; it is unrelated to NewtonBench's optional monetary-cost mode.

For faster model-development iterations, use the coverage-balanced
representative subsets:

```bash
python scripts/run_all_benchmarks.py \
  --benchmark chem \
  --representative-subset \
  --workers 2 \
  --out-root results/representative

python scripts/run_all_benchmarks.py \
  --benchmark grn \
  --representative-subset \
  --workers 4 \
  --out-root results/representative
```

The Chem subset contains 14 of the 27 release tasks. It covers the four main
substrate-law families (Michaelis-Menten, ping-pong, Hill, and substrate
inhibition), all novel mechanism families present in the release manifest,
and competitive, uncompetitive, noncompetitive/mixed, product, Arrhenius, pH,
cooperative, and activation effects. The GRN subset contains one task for each
of the five network motifs and spans easy/medium/hard difficulty plus all three
law versions. Both keep the original budget grids and benchmark settings.
These subsets are intended for iteration and regression checks; use the full
manifests for final paper-comparable aggregate numbers.

The manifests are
`configs/representative_subsets/chembench_representative14.json` and
`configs/representative_subsets/grnbench_representative5.json`. Use a separate
`--out-root` from full-manifest runs because manifest identity is part of the
resume fingerprint.

Use `--limit 1` first for a smoke test. Remove it for the complete manifests;
the already completed first task is reused. Increase `--workers` only if the
local inference server is configured for useful concurrent request batching.
Add `--dry-run` to inspect the exact child commands without starting any work.
Before launching workers, each runner checks `<OPENAI_BASE_URL>/models` and
verifies the requested model ID. Use `--skip-preflight` only for a compatible
gateway that does not expose that endpoint.

Runs are crash-safe and resumable by default under `results/paper_release_runs/`:

- every completed task is written atomically and skipped on the next invocation;
- failed tasks are retried;
- successful LLM responses are cached inside their task directory, so an
  interrupted task replays completed calls instead of generating them again;
- summaries are checkpointed after each completed task;
- a recorded configuration fingerprint prevents results from different models,
  endpoints, manifests, or benchmark settings from being mixed.

Resume by running the same command again. To run a different configuration, set
a different `--out-root`. `LLM_REQUEST_TIMEOUT_SECONDS` controls the per-request
timeout and defaults to 1800 seconds for long local generations.

## Direct Benchmark Runners

Newton:

```bash
python scripts/run_newton_llm_autoscilab_budget.py \
  --model gpt-4o-mini \
  --budgets 10 20 50 \
  --workers 4
```

Chem:

```bash
python scripts/run_chembench_llm_autoscilab_budget.py \
  --main-model gpt-4o-mini \
  --budgets 40 60 80 \
  --workers 4
```

GRN:

```bash
python scripts/run_grn_prompt_budget.py \
  --main-model gpt-4o-mini \
  --budgets 10 20 50 \
  --workers 4
```


For a lightweight live run, use:

```bash
python scripts/run_all_benchmarks.py --benchmark all --workers 1 --limit 1
```
