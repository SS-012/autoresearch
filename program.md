# NovaX Autoresearch Program

This autoresearch run is for optimizing NovaX itself, not the original
nanochat `train.py` experiment. The goal is to make NovaX's differentiated GPU
execution path blazingly fast: lazy graph fusion, static-shape graph
capture/replay, square/specialized matmul, and fused matmul epilogues. Broad
benchmark coverage still runs as a guardrail, but the primary score is now the
focused edge where NovaX can structurally beat PyTorch.

## Objective

Optimize the NovaX library so that the focused differentiated-path GPU
benchmarks improve over the current best NovaX baseline and, over time, NovaX's
edge cases become dramatically faster than PyTorch.

The focused benchmark scope is:

- `matmul` cases, especially small/medium square shapes where NovaX can avoid
  excess eager overhead or use a narrow fast path.
- `fusion` cases, where lazy expression graphs collapse multiple PyTorch eager
  launches into one NovaX kernel.
- `fused_mm` cases, especially matmul + bias + activation epilogues.
- `training_lane` cases, where a static captured forward/backward/optimizer
  step can beat PyTorch eager training by keeping the whole step GPU-resident.
- `inference_capture_*` cases, where static repeated work can replay through
  CUDA graphs.

The main strengths to exploit are:

- Lazy expression graphs and elementwise kernel fusion.
- Kernel compile caching.
- Bucketed GPU memory reuse.
- Low-overhead CUDA launch paths through PyCUDA.
- Specialized fused kernels such as matmul + bias + ReLU.
- CUDA graph capture/replay for repeated static workloads.
- Workloads where PyTorch eager mode pays multiple kernel launches or Python
  dispatch overhead for a chain that NovaX can represent as one graph.

Do not spend autoresearch budget trying to beat PyTorch on every isolated eager
elementwise, activation, bandwidth, or reduction case. Those cases remain useful
for catching severe regressions, but they are not the primary objective unless a
change also strengthens the focused differentiated path.

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

It prints a human-readable table and focused summary lines such as:

```text
benchmarks_ok: 31
benchmarks_error: 0
pytorch_wins: 7
geomean_novax_vs_pytorch: 0.681234
overall_geomean_novax_vs_pytorch: 1.046350
baseline_scope: differentiated
baseline_comparable: 11
improved_tests: 2
regressed_tests: 1
overall_regressed_tests: 4
research_score: 42.713901
qualified: yes
```

When `--baseline-json` is supplied, the benchmark compares current NovaX times
against that baseline. The primary comparison and `research_score` now use only
the differentiated-path cases listed above. A run qualifies only when:

- At least one comparable focused benchmark is faster by the improvement
  threshold.
- The number of focused regressions stays within the regression budget.
- The focused weighted research score is positive.

Default thresholds in the runner are 3 percent improvement, 5 percent
regression, and a regression budget of the larger of 2 tests or 10 percent of
focused comparable tests. This matches the desired behavior: keep an experiment
when it makes NovaX's edge faster without degrading many focused edge cases.

PyTorch comparison is still important context. In the benchmark output,
`geomean_novax_vs_pytorch`, `pytorch_wins`, `pytorch_ties`, and `pytorch_losses`
now describe the focused differentiated path. The full-suite comparison is
reported separately as `overall_geomean_novax_vs_pytorch`,
`overall_pytorch_wins`, `overall_pytorch_ties`, and `overall_pytorch_losses`.
Use the overall metrics as guardrail context, not as the primary optimization
target.

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

The `pytorch_wins` and `geomean` columns are focused differentiated-path values.

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

3. Modify the library. Favor optimizations that strengthen NovaX's focused edge:

- Extend fusion so unary roots can fuse with binary child graphs.
- Improve GPU broadcasting for vector bias and scalar operands without falling
  back to CPU.
- Avoid unnecessary host transfers, synchronizations, temporary tensors, and
  repeated CUDA source generation.
- Reuse buffers safely through `mempool`.
- Improve square/specialized matmul or fused matmul+bias+activation kernels.
- Add or improve capture/replay and graph-level execution when correctness and
  fallback behavior are clean.
- Consider Triton/CUDA/CUTLASS/cuBLASLt-style fused kernels for stable hot
  shapes rather than broad eager micro-optimizations.
- Do not retry direct exact64 matmul dispatch reordering as a standalone
  Python-overhead edit. It passed correctness but did not improve the intended
  exact64 row, and the 3-run focused gate failed with `inference_capture` as
  the worst regression.
- Do not retry caller-side shape gating for the current fused-mm fast-path
  helper dispatch as a standalone wrapper optimization. It qualified once, but
  confirmation failed and the `fused_mm_128` target signal did not hold.
- For the fast training lane, prefer schedule-level changes that reduce graph
  nodes, memory traffic, or vendor-library work. Avoid isolated per-thread
  coarsening or local arithmetic rewrites unless a microprobe and full median
  gate both show a durable target win.
- Treat exact output-backward/ReLU fusion as proven locally but not keepable as
  a standalone launcher under the current coupled focused gate. Revisit it only
  as part of a shape-keyed whole-step workspace, generated train-step plan, or
  explicitly training-isolated scoring mode.
- Captured-only exact output-backward plus ReLU fusion is a keep only under a
  fresh paired-control baseline or a stable regenerated `best.json`. It
  improves the train-step row by about `1.30x`; do not reject or promote future
  train-step variants solely against an old lucky saved-best artifact.
- Do not retry the local exact train-lane microvariants that failed event
  probes after that keep: shared-memory `pred-target` tiling inside exact
  output-backward, split exact output/hidden-gradient kernels, restrict or
  launch-bounds hints on that kernel, exact no-bounds captured first-layer
  matmul+bias+ReLU, or exact no-bounds captured output matmul+bias.
- Build on the kept captured input-gradient update schedule: fusing `d_w1/d_b1`
  production with the `w1/b1` update qualified and moved the train row to about
  `0.0322ms`. Future train-step plans should fuse update work where gradients
  are produced while still returning API-visible gradient buffers.
- Do not retry standalone block-geometry retuning of that exact
  input-gradient/update kernel. The `32x8` and `64x4` probe improved component
  CUDA-event timing, but the `32x8` candidate failed confirmation and did not
  preserve a durable train-row win.
- The kept captured output-shape `w2/b2` update specialization is useful
  cleanup, but it did not create a new headline train-step jump. Do not spend
  more standalone effort on tiny optimizer-tail kernels unless they are part of
  a larger generated/static train-step plan.
- Do not retry exact by-parameter variants of the current four-parameter SGD
  update as standalone launchers. The isolated update improved locally, but the
  captured training row did not improve and the focused gate regressed badly.
- Do not retry vectorizing the current segmented four-parameter SGD update as
  a standalone edit. It improved isolated CUDA-event timing but did not improve
  the captured training benchmark and failed the focused gate.
- Build on the kept `2cc6e24` result only with care: `float2` vectorization is
  useful for the exact same-shape fusion-chain launchers, but the earlier
  `float4` route failed the full focused gate. Any future vector-width change
  needs a direct event probe and a 3-run median benchmark gate.
- Do not retry alias-hint-only edits for those exact `float2` fusion-chain
  kernels as standalone changes. Adding `__restrict__` looked slightly faster
  in an event probe but failed the full focused gate with four regressions.
- Do not retry exact `float2` fusion-chain FMA/block-size tweaks without a
  much larger repeatable local margin; repeated event probes did not preserve
  the apparent chain5 win.
- Do not retry simple MSE inverse-scale or block-size changes as standalone
  fast-training-lane edits; isolated probes did not cleanly beat the current
  512-thread atomic-loss kernel.

Avoid unfocused tweaks to isolated eager elementwise, activation, or reduction
kernels unless they are required by the focused path or remove a severe
guardrail regression.

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

For serious keep/revert decisions, especially when a target improvement is large
but unrelated focused rows are noisy, use the 3-run median gate:

```bash
python benchmarks/novax_gpu_benchmark.py --profile research --suite-repeats 3 --baseline-json autoresearch/best.json --write-json autoresearch/last.json > autoresearch/run.log 2>&1
```

If the saved `best.json` was produced by a single lucky run, first create a
stable no-change baseline with the same median gate, then compare the candidate
against that stable artifact. Do not lower the bar silently; record which
baseline was used in `results.tsv`.

On Windows PowerShell, extract the key lines with:

```powershell
Select-String -Path autoresearch/run.log -Pattern "^(benchmarks_error|focus_cases|pytorch_wins|geomean_novax_vs_pytorch|overall_geomean_novax_vs_pytorch|baseline_scope|improved_tests|regressed_tests|overall_regressed_tests|research_score|qualified):"
```

On bash-like shells, use:

```bash
grep -E "^(benchmarks_error|focus_cases|pytorch_wins|geomean_novax_vs_pytorch|overall_geomean_novax_vs_pytorch|baseline_scope|improved_tests|regressed_tests|overall_regressed_tests|research_score|qualified):" autoresearch/run.log
```

7. Decide:

- Keep if `qualified: yes`, `benchmarks_error` did not increase unexpectedly,
  tests passed, and any overall-suite regressions are understood.
- For tiny wins near noise level, rerun the same benchmark once. Keep only if
  the result remains qualified or the improvement is clearly meaningful.
- For strong target wins with unrelated regressions, prefer the 3-run median
  gate over ad hoc single-run tiebreakers.
- Discard if `qualified: no`, focused regressions exceed the budget,
  correctness fails, or the change adds complexity without a durable focused
  speedup.
- If a change qualifies on the focused metric but causes large non-focus
  regressions, prefer narrowing or gating the change instead of discarding the
  focused idea outright.

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
- For the focused fusion-chain kernels, explore only narrow, shape-guarded
  follow-ups that preserve the current `float2` fast path and do not broaden
  dispatch matching.

Keep moving. The loop is autonomous: propose, implement, test, benchmark, keep
or discard, then continue.
