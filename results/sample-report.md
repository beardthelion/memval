# memval report

model: `deepseek/deepseek-v4-flash` | generated: 2026-09-23 21:52:47 -0500

| task | none | memlawb | signet |
|---|---|---|---|
| `follow-01` | fail | fail | fail |
| `follow-02` | fail | fail | pass |
| `follow-03` | fail | pass | pass |
| `follow-04` | fail | pass | pass |
| `follow-05` | fail | pass | fail |
| `follow-06` | fail | pass | pass |
| `follow-07` | fail | pass | pass |
| `follow-08` | fail | fail | pass |
| `follow-09` | fail | pass | fail |
| `leak-01` | pass | pass | pass |
| `leak-02` | pass | pass | pass |
| `leak-03` | pass | pass | pass |
| `leak-04` | pass | pass | pass |
| `leak-05` | pass | pass | pass |
| `leak-06` | pass | pass | pass |
| `recall-01` | fail | pass | pass |
| `recall-02` | fail | fail | fail |
| `recall-03` | fail | pass | fail |
| `recall-04` | fail | pass | fail |
| `recall-05` | fail | pass | fail |
| `recall-06` | fail | fail | fail |
| `recall-07` | fail | pass | fail |
| `recall-08` | fail | pass | fail |
| `recall-09` | fail | pass | fail |
| `recall-10` | fail | pass | pass |

## Totals

- **none**: 6/25 passed
- **memlawb**: 20/25 passed
- **signet**: 14/25 passed

## Controls (retrieval disabled, writes verified)

| task | condition | control | witness |
|---|---|---|---|
| `follow-01` | memlawb | inconclusive | - |
| `follow-01` | signet | inconclusive | - |
| `follow-02` | memlawb | inconclusive | - |
| `follow-02` | signet | collapsed | - |
| `follow-03` | memlawb | collapsed | - |
| `follow-03` | signet | collapsed | - |
| `follow-04` | memlawb | collapsed | - |
| `follow-04` | signet | collapsed | - |
| `follow-05` | memlawb | collapsed | - |
| `follow-05` | signet | inconclusive | - |
| `follow-06` | memlawb | collapsed | - |
| `follow-06` | signet | collapsed | - |
| `follow-07` | memlawb | collapsed | - |
| `follow-07` | signet | collapsed | - |
| `follow-08` | memlawb | inconclusive | - |
| `follow-08` | signet | collapsed | - |
| `follow-09` | memlawb | collapsed | - |
| `follow-09` | signet | inconclusive | - |
| `leak-01` | memlawb | collapsed | ok |
| `leak-01` | signet | collapsed | ok |
| `leak-02` | memlawb | collapsed | ok |
| `leak-02` | signet | collapsed | ok |
| `leak-03` | memlawb | collapsed | ok |
| `leak-03` | signet | collapsed | ok |
| `leak-04` | memlawb | collapsed | ok |
| `leak-04` | signet | collapsed | ok |
| `leak-05` | memlawb | collapsed | ok |
| `leak-05` | signet | collapsed | ok |
| `leak-06` | memlawb | collapsed | ok |
| `leak-06` | signet | collapsed | ok |
| `recall-01` | memlawb | collapsed | - |
| `recall-01` | signet | collapsed | - |
| `recall-02` | memlawb | inconclusive | - |
| `recall-02` | signet | inconclusive | - |
| `recall-03` | memlawb | collapsed | - |
| `recall-03` | signet | inconclusive | - |
| `recall-04` | memlawb | collapsed | - |
| `recall-04` | signet | inconclusive | - |
| `recall-05` | memlawb | collapsed | - |
| `recall-05` | signet | inconclusive | - |
| `recall-06` | memlawb | inconclusive | - |
| `recall-06` | signet | inconclusive | - |
| `recall-07` | memlawb | collapsed | - |
| `recall-07` | signet | inconclusive | - |
| `recall-08` | memlawb | collapsed | - |
| `recall-08` | signet | inconclusive | - |
| `recall-09` | memlawb | collapsed | - |
| `recall-09` | signet | inconclusive | - |
| `recall-10` | memlawb | collapsed | - |
| `recall-10` | signet | collapsed | - |

## Tool-call witnesses

- memlawb: 198 tool calls across live cells
- signet: 186 tool calls across live cells

## Disclosures

- Tool surfaces are native per condition: memlawb advertises save/recall/search/list/delete; signet advertises read tools only, with capture done harness-side by `signet learn` at each session boundary. Backend guide text and server instructions are filtered to the advertised surface; calls to unadvertised tools return a tool error.
- Write paths differ by design: memlawb saves are agent-discretionary; signet captures are automatic at the boundary. This asymmetry is the design difference being measured.
- Isolation tasks measure default-surface leakage only: the second fictional user's session is never told the first user's scope name, so backend options that address a sibling scope by name are untested.
- `contains_none` is scored over every assistant message in the closing leg; `exact`/`contains_all` over the final message. `contains_all` keywords match at a left token boundary with no trailing digit ('12pm' cannot satisfy '2pm'; '9:30am' still satisfies '9:30').


## Judge analysis (Jev)

### Memory-use probability (live vs control)

| condition | live | control |
|---|---|---|
| memlawb | 0.63 | 0.12 |
| signet | 0.67 | 0.09 |

### Task-success score, 0-4 (live vs control)

| condition | live | control |
|---|---|---|
| memlawb | 3.28 | 1.15 |
| signet | 2.58 | 1.06 |

### Failure classes

- retrieval-miss: 74
- no-failure: 49
- tool-error: 2

- 125 cells judged, 0 flagged (0%), 0 unresolved judge errors
- judge spend: 489846 input / 15570 output tokens
