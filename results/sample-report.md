# memval report

model: `deepseek/deepseek-v4-flash` | generated: 2026-09-23 16:05:16 -0500

| task | none | memlawb | signet |
|---|---|---|---|
| `follow-01` | fail | fail | pass |
| `follow-02` | fail | fail | pass |
| `follow-03` | fail | pass | pass |
| `follow-04` | fail | pass | pass |
| `follow-05` | fail | pass | fail |
| `follow-06` | fail | pass | pass |
| `follow-07` | fail | pass | fail |
| `follow-08` | fail | fail | fail |
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
| `recall-04` | fail | fail | pass |
| `recall-05` | fail | pass | fail |
| `recall-06` | fail | pass | fail |
| `recall-07` | fail | pass | fail |
| `recall-08` | fail | pass | fail |
| `recall-09` | fail | pass | fail |
| `recall-10` | fail | fail | fail |

## Totals

- **none**: 6/25 passed
- **memlawb**: 19/25 passed
- **signet**: 13/25 passed

## Controls (retrieval disabled, writes verified)

| task | condition | control | witness |
|---|---|---|---|
| `follow-01` | memlawb | inconclusive | - |
| `follow-01` | signet | collapsed | - |
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
| `follow-07` | signet | inconclusive | - |
| `follow-08` | memlawb | inconclusive | - |
| `follow-08` | signet | inconclusive | - |
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
| `recall-04` | memlawb | inconclusive | - |
| `recall-04` | signet | collapsed | - |
| `recall-05` | memlawb | collapsed | - |
| `recall-05` | signet | inconclusive | - |
| `recall-06` | memlawb | collapsed | - |
| `recall-06` | signet | inconclusive | - |
| `recall-07` | memlawb | collapsed | - |
| `recall-07` | signet | inconclusive | - |
| `recall-08` | memlawb | collapsed | - |
| `recall-08` | signet | inconclusive | - |
| `recall-09` | memlawb | collapsed | - |
| `recall-09` | signet | inconclusive | - |
| `recall-10` | memlawb | inconclusive | - |
| `recall-10` | signet | inconclusive | - |

## Tool-call witnesses

- memlawb: 195 tool calls across live cells
- signet: 220 tool calls across live cells

## Disclosures

- Tool surfaces are native per condition: memlawb advertises save/recall/search/list/delete; signet advertises read tools only, with capture done harness-side by `signet learn` at each session boundary. Backend guide text is filtered to the advertised surface.
- Write paths differ by design: memlawb saves are agent-discretionary; signet captures are automatic at the boundary. This asymmetry is the design difference being measured.
- Isolation tasks measure default-surface leakage only: the second fictional user's session is never told the first user's scope name, so backend options that address a sibling scope by name are untested.
- `contains_none` is scored over every assistant message in the closing leg; `exact`/`contains_all` over the final message.
