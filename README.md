# memval

A reproducible evaluation harness that measures whether durable encrypted
memory actually helps an agent. One shared agent loop answers a fixed 25-task
battery under three conditions: no memory, memlawb (a crypto-blind encrypted
memory backend), and signet (a DID-rooted encrypted state system). Every
memory-dependent task is also run with retrieval disabled, so the report can
prove the memory delta is causal rather than asserted.

The point is evidence, not vibes: a hiring manager can open the committed
sample report and see the three-way comparison plus the control check without
running anything.

## Requirements

- Python 3.11+ and [uv](https://docs.astral.sh/uv/)
- For the memory conditions: [Bun](https://bun.sh) and local checkouts of the
  memlawb and signet repositories
- A reachable OpenAI-compatible upstream, or the bundled fake upstream for
  fixture runs
- Optional, for `memval judge`: `TYPESAFE_API_KEY` in the environment

## Quickstart

```bash
uv sync

# Point the harness at a model. memval.config.json maps model names to
# upstream routes; the key is read from the named environment variable and
# never stored in the config.
cat > memval.config.json <<'EOF'
{
  "models": {
    "my-model": {
      "base": "https://api.example.com/v1",
      "key_env": "EXAMPLE_API_KEY"
    }
  }
}
EOF

# Everything is loopback or local subprocess; the only network traffic is the
# gateway's upstream relay.
uv run memval run --config memval.config.json --model my-model
```

There is no shipped default model or provider. `--model` is required whenever
the config names more than one route.

### Fixture mode (no API key, no real model)

`tests/fixtures/fake_upstream.py` is a deterministic heuristic stand-in for a
chat model. Run it and point a config at it:

```bash
uv run python tests/fixtures/fake_upstream.py --port 8901 &
cat > fixture.config.json <<'EOF'
{"models": {"fake": {"base": "http://127.0.0.1:8901", "key_env": null}}}
EOF
uv run memval run --config fixture.config.json --model fake --fixture
```

`--fixture` marks the report header so a generated sample is never mistaken
for a real-model result.

### Commands

```bash
memval run [--config <file>] [--model <name>] [--conditions none,memlawb,signet]
           [--tasks recall-01,leak-03] [--results <file.jsonl>] [--seed N]
           [--fixture]
memval report <results.jsonl>
memval judge <results.jsonl> [--tasks-dir PATH] [--endpoint URL]
memval calibrate <results.jsonl> [--sample 30 | --labels <file>]
memval check-isolation <memlawb|signet>
```

- `run` executes the battery, appends one JSONL record per
  task x condition x variant cell, and writes a Markdown report next to the
  results file. Exit is nonzero only when the harness could not run: bad
  config, missing upstream, no tasks matched. A condition whose backend fails
  preflight is recorded as `not_run`, never as a zero pass rate.
- `report` re-renders any complete or partial results file. When a judge
  sidecar exists next to the results file the report gains a Judge analysis
  section; without one it renders exactly as before.
- `judge` runs the optional Jev analysis pass over a finished results file
  (below). It never mutates results and can be re-run to fill gaps.
- `calibrate` emits a blinded human-labeling worksheet for a stratified
  sample of judged cells, or scores a filled worksheet against the sidecar.
- `check-isolation` stands a backend up under a fresh temporary root, writes
  and recalls a fact, tears it down, then proves a fresh root recalls nothing
  and no artifacts (store files, custody, passphrase files, transcripts)
  survive teardown.

A full three-condition run is 25 tasks x 3 conditions x (live + control) for
the memory conditions, so roughly 125 cells before multi-turn tool overhead.
Each cell makes several upstream calls. Use `--conditions`, `--tasks`, and
resume (below) rather than firing a full battery blind.

### Resuming

`--results some.jsonl` appends to and resumes a partial run: cells that
recorded `pass`, `fail`, or `not_run` are skipped; `error` cells re-execute.
Kill a run any time and restart it against the same file.

## The battery

`tasks/` holds 25 JSON tasks in three families:

- `recall-*` (10): plant a preference or fact in leg 1, ask for it after the
  boundary.
- `follow-*` (9): multi-turn instructions whose consequences must survive
  into the closing leg.
- `leak-*` (6): two fictional users share the agent; user 2's session must
  not surface user 1's facts.

Each task file declares `id`, `type`, `sessions` (legs of scripted user
turns), `planted_facts`, `expected`, `control`, and optionally `users`.

The session boundary is the mechanism that makes memory matter: between legs
the harness wipes the agent's message context entirely while the backend
stores keep running. Nothing in leg 2's prompt carries the planted facts, and
a validator (`echo_guard` in `tasks.py`) rejects any task whose post-boundary
turns echo a fact verbatim. A no-memory agent cannot pass a recall task by
re-reading the prompt, because the prompt is gone.

## Conditions

All three run the same agent loop, system prompt, generation settings, and
tool-call budget. Only the memory wiring differs:

- **none**: no tools are advertised at all. This is the floor and should fail
  every memory-dependent task.
- **memlawb**: a per-task `memlawb mcp` stdio process scoped to a fresh
  namespace (`user:eval/<task-id>`), backed by a per-run store. The agent gets
  the native save/recall/search/list/delete surface.
- **signet**: a per-task `signet mcp` process under a fresh custody created
  by `signet init`. The agent-facing surface is read-only
  (recall/search/list); durable writes come from `signet learn`, which the
  harness runs on each leg's transcript at the boundary. Calls to tools
  outside the advertised surface are rejected as tool errors, so the
  read-only promise is enforced at dispatch, not just in the prompt. That
  asymmetry, agent-discretionary saves versus automatic capture, is part of
  what the comparison measures and is disclosed in the report.

Isolation tasks respawn the MCP process per fictional user because both
backends bind their scope at process start: memlawb via `MEMLAWB_NAMESPACE`,
signet via `SIGNET_HOME` custody.

## Scoring

Scoring is mechanical and is the scorer of record; the optional Jev layer
below is analysis, not the outcome.

- `exact`: normalized final answer equals `expected.value`.
- `contains_all`: every `expected.keywords` entry appears in the normalized
  final answer at a left token boundary with no trailing digit (so `12pm`
  cannot satisfy `2pm`, but `9:30am` still satisfies `9:30`).
- `contains_none`: no `expected.forbidden` token appears in *any* assistant
  message of the closing leg, so a mid-session leak cannot pass on a clean
  final line.

Normalization is case-folding plus whitespace collapse. Cell outcomes are
`pass`, `fail`, `error` (harness or upstream fault, re-run on resume), and
`not_run` (condition could not start).

## Judge layer (optional)

`memval judge` adds a supplementary analysis pass on top of the mechanical
outcome. It sends each scored cell's blinded transcripts to the Jev System
One endpoint in one batched call and records four typed answers per cell:

- `task_success`: a 0-4 rubric score with confidence,
- `memory_used`: a probability the agent actually used stored memory,
- `confabulated`: a probability the agent cited memory that was never
  planted (the task's `planted_facts` are the ground truth, so the
  question means the same thing under all three conditions),
- `failure_class`: one of `no-failure`, `retrieval-miss`,
  `wrong-memory`, `confabulation`, `tool-error`, `refused`.

These are calibrated probabilities from a judge model, not ground truth.
They never change the mechanical outcome and never feed the control check.

**Artifacts.** Transcripts persist to `<results-stem>.transcripts/` and
judge output to `<results-stem>.judge.jsonl`; both sit beside the results
file. Re-running `judge` skips cells that already judged cleanly and
retries `judge_error` cells.

**Blinding.** The judge never sees condition labels: transcripts are
stripped of backend names and tool names are normalized to generic memory
verbs. Blinding is not total, by construction. `none` cells have no tool
calls and memlawb's agent-visible writes have no signet counterpart, so
the judge can still infer the condition family from call shapes. The
calibration pass below measures the residual per-condition bias rather
than pretending it is zero.

**Task-type notes.** The bundle is generic, and without context a correct
non-disclosure on an isolation task reads to the judge as a wrong or
ambiguous answer. `judge_state` therefore prepends a per-type note for
isolation cells (a refusal to leak counts as success, and retrieval used
to exclude a fact still counts as memory use). The notes are hashed into
`bundle_version`, so changing them mints a new judge generation.

**Flags.** Answers the judge is unsure about are flagged: score and
failure-class answers under 0.6 confidence, noul answers with probability
inside [0.35, 0.65]. Flagged cells stay out of the report's judge means
until a human adjudicates them through calibration; if more than 15% of
cells are flagged the judge section marks itself invalid.

**Calibration.** `memval calibrate <results> --sample 30` writes a blinded
worksheet (opaque row ids, neutral transcript names, condition join key
kept harness-side) plus every flagged cell. Fill in `human_score`,
`human_memory_used`, and `human_confabulated`, then run
`memval calibrate <results> --labels <file>` to get agreement-within-one
rung, per-condition bias, confidence reliability, and confabulation
precision/recall against a 0.8 agreement bar. Labeled flagged cells get
an adjudication record in the sidecar, clearing the flag. Re-run
calibration whenever the battery or rubric changes; there is no standing
per-run audit.

**Cost.** One batched call is roughly 700 input tokens and under half a
second; a full 125-cell pass costs fractions of a cent and about a minute.

## The control

Every memory-dependent task ships a control variant: the same cell with
retrieval blinded. Read tools stay advertised (so the agent's behavior is
unchanged) but return empty results; writes still land. At the end of each
control cell the harness writes a sentinel entry through the backend and
reads it back unblinded.

A control is only meaningful if both halves are proven: reads served nothing
(observed blinded read calls) and a write provably landed (the sentinel).
Outcomes:

- `collapsed`: live passed, control dropped, at least one read was provably
  blinded, and the sentinel landed. This is the evidence that the memory
  delta is causal.
- `no_drop`: the control scored the same as a passing live run; the task is
  flagged `INVALID` in the report because it was not measuring memory.
- `inconclusive`: the harness could not prove both halves.

For isolation tasks the polarity is reversed: passing under blinded reads is
correct behavior, and validity rides on the witness (the plant provably
landed under its own scope before the boundary).

## Security model

What each layer actually protects, and what it does not:

- **The passphrase file.** memlawb gets `MEMLAWB_PASSPHRASE_FILE`, a path to a
  0600 file created inside the per-run root. The passphrase value never
  appears in the environment block, on a command line, in logs, or in
  results. This protects the key material from process-listing and env-scrape
  disclosure, not from something that can already read the run root.
- **The crypto-blind store.** The memlawb and signet stores see only
  ciphertext; plaintext and derived keys stay in the MCP client process.
  Compromising a run-root store leaks ciphertext and namespace structure,
  not contents.
- **Neither protects against** an operator reading the live MCP stdio stream,
  the transcripts while the run root exists, or memory returned to a
  correctly-scoped agent. Isolation tasks test default-surface leakage only:
  they ask "does user B's session surface user A's facts unprompted", not
  "can a determined user B address user A's scope by name".

Child processes get an explicit environment allow-list rather than the
inherited environment, so ambient credentials (`AWS_*`, stray `PORT`, and so
on) cannot redirect a per-run store at real data. MCP child stdout is the
protocol channel; any non-protocol byte on it is treated as fatal. The
gateway binds 127.0.0.1 only, requires https upstreams except loopback
fixtures, injects the upstream key from the named environment variable at
relay time, and propagates upstream status codes verbatim.

## Results and reports

`results/` is git-ignored except one committed sample report. Each JSONL
record carries the cell key, outcome, tool-call count, blinded-read count,
sentinel and witness results, the final answer (truncated), model, and seed.
Per-cell transcripts persist next to the results file under
`<results-stem>.transcripts/` so `memval judge` can score them after the
run root is gone. The Markdown report renders the three-way live table,
totals, the control table with witness marks, tool-call witness counts (a
memory condition with zero tool calls did not exercise the backend), the
disclosures block, and the judge section when a sidecar exists.

Reproducibility is "comparable, not bitwise identical": temperature 0 and
seed passthrough are sent, but upstream serving is not deterministic even
then. Two runs against the fake upstream produce identical report structure;
a real model will drift in phrasing while the pass/fail pattern should hold.

## Private components

memlawb and signet are private repositories. This repo references only their
public interfaces (CLI subcommands, MCP tool names, environment variables);
nothing here exposes their internals. The default checkout paths are
`~/projects/memlawb` and `~/projects/signet`; override with
`--memlawb-checkout` / `--signet-checkout`.

## Development

```bash
uv run pytest                                   # full suite
uv run pytest tests/test_tasks.py               # battery schema gate
uv run memval check-isolation memlawb           # backend enable/disable check
uv run memval check-isolation signet
```

The isolation tests skip cleanly when bun or the private checkouts are
absent, so the suite stays green for outside contributors.
