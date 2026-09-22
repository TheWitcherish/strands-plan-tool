# strands-plan-tool

When your agent needs three tool calls in a row, you wait for the model three times — not
because it is thinking, but because it has to be *asked* again before it can make the next
call.

This adds one tool that lets the model hand over a whole plan instead. A dependency chain
of depth D costs **one** model round trip instead of D.

```bash
uv add strands-plan-tool
```

## 30 seconds

```python
from strands import Agent
from strands_plan_tool.strands_adapter import WorkflowPlanPlugin

agent = Agent(tools=[read_action_log, get_character, roll_check], plugins=[WorkflowPlanPlugin()])
agent("Resolve this combat round.")
```

That registers one extra tool, `submit_workflow_plan`. Tool choice stays automatic: the
model plans when it helps and ignores the tool when it does not.

Two runnable examples, no credentials needed:

```bash
uv run python -m examples.game_master    # 12 steps, depth 6, three-wide fan-out
uv run demo.py                           # 3 steps, straight chain
```

## Writing a plan

A plan is steps, dependencies, and which results come back. Three fields carry all of it:

| Field | What it does |
|---|---|
| `args` | Literal arguments, known before the plan runs |
| `bind` | JSONata expressions evaluated against `{"steps": {<id>: <result>}}` |
| `returns` | The only step ids whose results travel back to the model |

Dependencies are **inferred from the bindings** — reading `steps.log.actions[0].actor` *is*
the edge, so you never declare it twice. `after` exists for the rare edge a binding does
not express.

```python
from strands_plan_tool import PlanStep, WorkflowPlan, execute_plan

plan = WorkflowPlan(
    steps=(
        PlanStep(id="log", tool="read_action_log"),
        PlanStep(id="hero", tool="get_character", bind={"name": "steps.log.actions[0].actor"}),
        PlanStep(
            id="check",
            tool="roll_check",
            bind={
                "actor": "steps.hero.name",
                "skill": "steps.log.actions[0].skill",
                "modifier": "steps.hero.skills.stealth",
            },
        ),
    ),
    returns=("check",),
    final=True,
)

result = await execute_plan(plan, my_tool_invoker)
```

`log` and `hero` did the work and never entered the model's context. Note that `check` binds
from *two* upstream steps — `hero` for the modifier and `log` for the skill name — so its
in-degree is 2 and both edges were inferred from the bindings alone.

## A real shape: one RPG combat round

`/examples/rpg.py` is a Game Master resolving a round for three players. It is the useful
example because a round is not a straight line:

```
L0  read_action_log                        1 step
L1  get_character   x3   (fan-out)         3 steps
L2  roll_check      x3                     3 steps
L3  resolve_action  x3                     3 steps
L4  tally_round          (join, in-degree 3)
L5  narrate_round
```

Twelve steps, depth 6, three wide. Run it:

```
plan: 12 steps, depth 6, widest level 3

  L0 ok  log        read_action_log    0.0ms
  L1 ok  c_vesna    get_character      0.3ms
  L1 ok  c_alder    get_character      0.1ms
  L1 ok  c_grimm    get_character      0.1ms
  ...
  L4 ok  tally      tally_round        0.1ms
  L5 ok  narration  narrate_round      0.2ms

  levels (model round trips replaced): 6
  returned to the model:       ['narration', 'tally']
  results the model never saw: 10

  Round resolved at the collapsed bridge at Kaer Trolde: 24 damage dealt. The troll still stands.
```

Ten of the twelve results never reached the model. It asked for a resolved round and got a
resolved round.

### The join is why it is JSONata and not a path syntax

`tally_round` wants one *array* of damage values, but each value lives in a different
step. A binding builds the array inline:

```python
PlanStep(
    id="tally",
    tool="tally_round",
    args={"target": "bridge troll"},
    bind={"damages": "[steps.a_vesna.damage, steps.a_alder.damage, steps.a_grimm.damage]"},
)
```

No glue tool, no extra round trip. A plain path syntax cannot express that; an expression
language can. `$sum`, `$count`, filters and predicates are all available the same way —
which is also how you shrink a large tool result to the one number the model needs.

### What the model actually does with it

Running the same example against a live model (`--live`) is the interesting part. It does
**not** blindly plan everything. Reproducibly, across runs:

```
tools the model called:  ['read_action_log', 'submit_workflow_plan']
did it plan?             True
model round trips:       2        (a plain loop needs ~7)
```

It calls `read_action_log` directly first — it cannot know how many characters to fetch
until it has seen the actions — then plans the eleven mechanical steps that follow. Nobody
prompted that split. **The model uses the loop where judgement is needed and the plan where
it is not**, which is exactly the division the tool is for.

Two practical notes from those runs:

- The model tends to over-populate `returns`, handing back every step instead of the two
  that matter. Saying "name in `returns` only the steps whose results you actually need"
  in your system prompt tightens it.
- Document your tools' **return shapes** in their docstrings. Tool schemas describe inputs
  only, so a model binding to `steps.hero.sign` when the field is `weakness` is your
  documentation's fault, not the model's.

## When it pays off

| Shape | Use a plan? |
|---|---|
| One tool call | No — the plan schema costs more tokens than it saves |
| Two or more *mechanically* dependent calls | Yes — this is the whole point |
| A lookup, then a fan-out over its results | Yes — the most common winning shape |
| Independent calls in one turn | No — already concurrent in the agent loop |
| Each result needs your judgement before the next | No — call them one at a time |

Break-even is **depth 2**. Below that, do not plan.

## Measured on a live model

Bedrock, `eu.anthropic.claude-sonnet-4-5-20250929-v1:0` in `eu-central-1`, 2026-09-22. Same
model, tools and prompt in both arms; each tool sleeps 250ms so the gap is attributable to
round trips, not tool speed. Medians of 2 trials.

```
depth |            loop            |            plan            | verdict
      |  wall   trips   tokens     |  wall   trips   tokens  pl%|
------+----------------------------+----------------------------+---------------
  1   |  3.27s   2.0     1309    |  3.36s   2.0     3532   0| model declined
  2   |  5.72s   3.0     2494    |  3.38s   1.0     1994 100| plan +41%
  3   |  7.33s   4.0     3930    |  4.12s   1.0     2157 100| plan +44%
  4   | 10.00s   5.0     5646    |  4.86s   1.0     2285 100| plan +51%
  5   | 12.82s   6.0     8128    |  5.69s   1.0     2518 100| plan +56%
```

Round trips stay flat at 1 while the loop's grow with depth — by depth 5, about half the wall
clock and under a third of the tokens. At depth 1 the model **declined to plan** (`pl% = 0`)
unprompted. Reproduce with
`AWS_PROFILE=... AWS_REGION=... uv run python -m bench.breakeven`.

## Options

```python
WorkflowPlanPlugin(
    plannable=["*", "!delete_files"],  # which tools a plan may call
    honor_final=True,  # a final plan ends the turn
    max_steps=32,  # ceiling per plan
)
```

**`plannable`** uses the same pattern syntax as `HumanInTheLoop.allowed_tools`.
**Approval-gated tools cannot be planned** — a direct tool call refuses interrupts, so such a
tool is rejected at validation and the model is told to call it directly. The gate keeps
working; it just is not batchable. If your agent has interventions attached this argument is
required, and construction fails loudly rather than silently batching something that should
have prompted a human.

**`honor_final`** ends the turn on a plan that set `final`, instead of sampling the model
again to restate results it already has. That removes the second round trip and most of the
token cost. The trade is visible: the closing message becomes JSON rather than prose, so pass
`False` to keep the prose. The exit only arms when every step succeeded.

## Safety

JSONata is a query language: no imports, no filesystem, no network, no subprocess. Its one
dynamic surface is `$eval` — which evaluates further JSONata, never Python — and it is off by
default, so a plan's bindings can be read completely before anything runs.
`JsonataBinder(allow_eval=True)` opts back in and gives up that inspectability.

Cycles, duplicate ids, dangling references and step-count overruns are rejected **before**
any tool runs, so a malformed plan costs one cheap repair round trip and leaves no side
effects. A binding that fails is a step failure, never a substituted `None`.

## What it does not change

**Not the agent loop.** The model emits exactly one tool call; from the event loop's side
that is an ordinary tool call — dispatch, one result, carry on.

**Not the model's autonomy.** The *model* writes the plan, not you — unlike an
author-defined DAG, fixed before the agent ever runs. What it gives up is re-deliberating
between the steps it chose to batch: the trade you make writing a shell pipeline instead of
running commands interactively.

## Development

The engine imports no agent framework. Tool invocation sits behind a `ToolInvoker` protocol,
which is why steps run in-process with no hop to the model provider between them, and where
the Strands adapter attaches. Ordering is Kahn's algorithm over in-degree counts, and
execution is barrier-free: each step starts the instant its own dependencies land.

```bash
uv sync --extra dev
uv run ruff check . && uv run ruff format --check .
uv run mypy src tests examples bench demo.py demo_bedrock.py
uv run pytest -q
```

MIT licensed.
