# NovaX Autoresearch Program

This autoresearch run is for optimizing NovaX itself, not the original
nanochat `train.py` experiment. The goal is to make NovaX faster than PyTorch
where its design gives it an advantage, while preserving broad benchmark
performance and correctness.

## Objective

Optimize the NovaX library so that GPU benchmarks improve over the current best
NovaX baseline and, over time, NovaX wins or ties more cases against PyTorch.

The main strengths to exploit are:

- Lazy expression graphs and elementwise kernel fusion.
- Kernel compile caching.
- Bucketed GPU memory reuse.
- Low-overhead CUDA launch paths through PyCUDA.
- Specialized fused kernels such as matmul + bias + ReLU.
- Workloads where PyTorch eager mode pays multiple kernel launches or Python
  dispatch overhead for a chain that NovaX can represent as one graph.

## Files In Scope

Read these before starting:

- `README.md`
- `docs/concepts.md`
- `docs/getting-started.md`
- `novax/core.py`
- `novax/dispatch.py`
- `novax/ops/launcher.py`
- `novax/utils/mempool.py`
- `novax/ops/gpu/*.py`
- `tests/*.py`
- `benchmarks/novax_gpu_benchmark.py`

You may modify files under `novax/` and add focused tests under `tests/`.
You may update docs only when behavior changes. Do not edit
`benchmarks/novax_gpu_benchmark.py` during an experiment unless the human
explicitly asks for benchmark changes.

## Metric

The benchmark runner is:

```bash
python benchmarks/novax_gpu_benchmark.py --profile research
```

It prints a human-readable table and summary lines such as:

```text
benchmarks_ok: 31
benchmarks_error: 0
pytorch_wins: 4
geomean_novax_vs_pytorch: 1.834221
baseline_comparable: 31
improved_tests: 2
regressed_tests: 1
research_score: 42.713901
qualified: yes
```

When `--baseline-json` is supplied, the benchmark compares current NovaX times
against that baseline. A run qualifies only when:

- At least one comparable benchmark is faster by the improvement threshold.
- The number of regressions stays within the regression budget.
- The weighted research score is positive.

Default thresholds in the runner are 3 percent improvement, 5 percent
regression, and a regression budget of the larger of 2 tests or 10 percent of
comparable tests. This matches the desired behavior: keep an experiment when it
finds a real faster timing in one or more tests and does not slow down many
others.

PyTorch comparison is still important context. Prefer changes that reduce
`geomean_novax_vs_pytorch`, increase `pytorch_wins`, or turn PyTorch losses into
ties. The keep/discard decision, however, is based on the baseline comparison so
that local timing noise and broad regressions are controlled.

## Setup

1. Choose a fresh run tag, for example `may27-novax`.
2. Create a branch from the current working branch:

```bash
git checkout -b autoresearch/<tag>
```

3. Verify dependencies in the active Python environment:

```bash
python -m pip install -e ".[gpu]" torch pytest
```

4. Run the tests:

```bash
python -m pytest -q
```

5. Establish the current-best benchmark baseline:

```bash
python benchmarks/novax_gpu_benchmark.py --profile research --write-json autoresearch/best.json > autoresearch/baseline.log 2>&1
```

6. Initialize `autoresearch/results.tsv` if it does not exist. Use tab
separation:

```text
commit	research_score	qualified	improved	regressed	errors	pytorch_wins	geomean	status	description
```

Do not commit `autoresearch/results.tsv`, `autoresearch/*.log`, or
`autoresearch/*.json` benchmark artifacts unless the human asks for them.

## Experiment Loop

Repeat indefinitely until interrupted by the human.

1. Record the starting commit:

```bash
git rev-parse --short HEAD
```

2. Pick one concrete optimization idea. Keep each experiment small enough that
the diff can be understood and reverted.

3. Modify the library. Favor optimizations that match NovaX's architecture:

- Extend fusion so unary roots can fuse with binary child graphs.
- Improve GPU broadcasting for vector bias and scalar operands without falling
  back to CPU.
- Avoid unnecessary host transfers, synchronizations, temporary tensors, and
  repeated CUDA source generation.
- Reuse buffers safely through `mempool`.
- Tune reduction kernels and block sizing.
- Improve matmul or fused matmul+bias+activation kernels.
- Add capture/replay or graph-level execution only if correctness and fallback
  behavior are clean.

4. Run correctness tests:

```bash
python -m pytest -q
```

If tests fail, fix the experiment or discard it.

5. Commit the experiment before benchmarking:

```bash
git add novax tests docs
git commit -m "autoresearch: <short experiment description>"
```

6. Run the research benchmark against the current best JSON:

```bash
python benchmarks/novax_gpu_benchmark.py --profile research --baseline-json autoresearch/best.json --write-json autoresearch/last.json > autoresearch/run.log 2>&1
```

On Windows PowerShell, extract the key lines with:

```powershell
Select-String -Path autoresearch/run.log -Pattern "^(benchmarks_error|pytorch_wins|geomean_novax_vs_pytorch|improved_tests|regressed_tests|research_score|qualified):"
```

On bash-like shells, use:

```bash
grep -E "^(benchmarks_error|pytorch_wins|geomean_novax_vs_pytorch|improved_tests|regressed_tests|research_score|qualified):" autoresearch/run.log
```

7. Decide:

- Keep if `qualified: yes`, `benchmarks_error` did not increase unexpectedly,
  and tests passed.
- For tiny wins near noise level, rerun the same benchmark once. Keep only if
  the result remains qualified or the improvement is clearly meaningful.
- Discard if `qualified: no`, there are broad regressions, correctness fails,
  or the change adds complexity without a durable speedup.

8. Log the result in `autoresearch/results.tsv`.

For a keep, copy the latest benchmark to the new best baseline:

```bash
copy autoresearch\last.json autoresearch\best.json
```

or on bash-like shells:

```bash
cp autoresearch/last.json autoresearch/best.json
```

For a discard, reset only the experiment commit you just created, after checking
that there are no unrelated uncommitted user changes:

```bash
git status --short
git reset --hard HEAD~1
```

## Logging

Append one row per experiment to `autoresearch/results.tsv`:

```text
commit	research_score	qualified	improved	regressed	errors	pytorch_wins	geomean	status	description
a1b2c3d	0.000000	no	0	0	0	2	2.315441	keep	baseline
b2c3d4e	42.713901	yes	2	1	0	3	2.102901	keep	fuse unary roots over binary expressions
c3d4e5f	-18.200000	no	1	4	0	3	2.250012	discard	larger reduction block size
d4e5f6g	0.000000	no	0	0	3	2	0.000000	crash	experimental cuda graph capture
```

## Guardrails

- Never optimize by changing the benchmark or hiding work from synchronization.
- Never sacrifice numerical correctness for timing.
- Do not install new runtime dependencies unless the human approves.
- Do not remove CPU fallback behavior while improving GPU performance.
- Do not keep changes that only help one tiny benchmark while causing broad
  slowdowns elsewhere.
- Prefer simple, local improvements over clever rewrites unless the measured
  speedup is large.

## What To Try First

The current codebase likely has wins available in these areas:

- Make lazy fusion include unary operations at the root, not just binary
  subgraphs.
- Add GPU broadcasting for shape-compatible bias vectors and scalars.
- Ensure common expression chains avoid intermediate kernel launches.
- Add tests for GPU broadcast and fused graph correctness.
- Improve memory lifecycle around temporary outputs so repeated benchmark loops
  reuse buffers instead of allocating fresh buffers.
- Benchmark matmul tile choices beyond 16x16 if the device supports them.

Keep moving. The loop is autonomous: propose, implement, test, benchmark, keep
or discard, then continue.
