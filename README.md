# strands-plan-tool

Ever noticed that when your agent needs three tool calls in a row, you wait for the
model three times? Not because the model is thinking — because it has to be *asked*
again before it can make the next call.

This package lets the model hand over one plan instead.

```bash
uv run demo.py
```

## What it does

The model writes a `WorkflowPlan`: which tools to call, what depends on what, and how
each step's arguments derive from earlier results. JSONata expressions carry data from
one step into the next. The plan runs locally in one batch, and only the steps the plan
named in `returns` travel back to the model.

A dependency chain of depth D costs **one** model round trip instead of D.

```python
from strands_plan_tool import PlanStep, WorkflowPlan, execute_plan

plan = WorkflowPlan(
    steps=(
        PlanStep(id="board", tool="read_notice_board"),
        PlanStep(
            id="beast", tool="consult_bestiary", bind={"beast": "steps.board.contracts[0].beast"}
        ),
        PlanStep(
            id="satchel",
            tool="pack_satchel",
            bind={"sign": "steps.beast.weakness", "oil": "steps.beast.oil"},
        ),
    ),
    returns=("satchel",),
    final=True,
)

result = await execute_plan(plan, my_tool_invoker)
```

`board` and `beast` did the work and never entered the model's context.

## What it does not do

**It does not change the agentic loop.** The model emits exactly one tool call. From
the event loop's side that is an ordinary single tool call: dispatch, get a result,
call the model again. Everything the plan does happens inside that one call.

**It does not make the agent static.** The *model* writes the plan, not you. Planning is
one tool among many with tool choice left automatic, so the model elects to plan, per
turn, or ignores it and calls tools one at a time. Contrast an author-defined DAG,
which is fixed before the agent ever runs.

**It does not remove judgement — it defers it.** Inside a plan the model does not
re-deliberate between steps. That is the trade, and it is the same one you make writing
a shell pipeline instead of running commands interactively. Plan when the dependencies
are *mechanical*. Do not plan when each result needs a decision.

## Where it pays off, and where it does not

The win is **depth, not width**. Independent tool calls in one turn are already
concurrent in most agent runtimes; a plan buys nothing there. It buys everything on the
shape that is actually common — a lookup, then a fan-out over its results, which is
depth-2 today because the model must see the ids before it can request them.

A plan costs more output tokens to generate than a single tool call, so there is a
break-even depth. Measure it on your own workload rather than assuming a win.

## Safety

JSONata is a query language, not a programming language: no imports, no filesystem, no
network, no subprocess. Its one dynamic surface is `$eval`, which evaluates further
JSONata (never Python), and this package shuts it by default so a plan's bindings can
be read completely before anything executes.

```python
JsonataBinder()  # $eval raises PermissionError
JsonataBinder(allow_eval=True)  # opt back in, and give up static inspectability
```

A binding that fails is a **step failure**, never a substituted `None`. Handing a tool
an argument the plan did not intend is worse than reporting that the plan was wrong.

Cycles, duplicate ids, dangling references, and step-count overruns are all rejected
before any tool runs, so a malformed plan costs one cheap repair round trip and leaves
no side effects behind.

## Architecture

The engine imports no agent framework. Tool invocation sits behind a `ToolInvoker`
protocol:

```python
async def __call__(self, name: str, args: Mapping[str, JsonValue]) -> JsonValue: ...
```

That seam is why the latency claim holds — steps run in this process, with no hop back
to the model provider between them — and it is where an agent-framework adapter attaches.

The boundary models are Pydantic v2 and frozen, so a TypeScript port mirrors
`models.py` as Zod schemas with the same field names.

## Measured on a live model

Bedrock, `eu.anthropic.claude-sonnet-4-5-20250929-v1:0` in `eu-central-1`, 2026-09-17.
Two arms, same model, same tools, same prompt: `loop` is an ordinary agent, `plan` is the
same agent plus this plugin. Each tool sleeps 250ms, so the wall-clock gap is attributable
to model round trips rather than tool speed. Medians of 2 trials.

```
depth |            loop            |            plan            | verdict
      |  wall   trips   tokens     |  wall   trips   tokens  pl%|
------+----------------------------+----------------------------+---------------
  1   |  3.16s   2.0     1309    |  3.47s   2.0     3534   0| model declined
  2   |  5.28s   3.0     2496    |  3.32s   1.0     1994 100| plan +37%
  3   |  8.53s   4.0     3932    |  4.08s   1.0     2146 100| plan +52%
  4   | 10.62s   5.0     5588    |  4.56s   1.0     2290 100| plan +57%
  5   | 11.32s   6.0     7962    |  5.49s   1.0     2522 100| plan +51%
```

Reproduce it with `AWS_PROFILE=... AWS_REGION=... uv run python -m bench.breakeven`.

Three readings, and the second matters as much as the first:

- **Round trips stay flat at 1 while the loop's grow with depth.** That is the mechanism and
  the whole claim. Break-even is **depth 2**; by depth 5 a plan is about half the wall clock
  and roughly a third of the tokens (2522 against 7962).
- **At depth 1 the model declined to plan** (`pl% = 0`) and called the tool directly. Nobody
  told it to. A one-call task is not worth a plan and it judged that correctly, which is what
  "planning is elected, not imposed" looks like in a measurement.
- **A plan is not free.** At depth 1, where the model does plan if pushed, the schema costs
  more tokens than it saves (3534 against 1309). Below depth 2 planning is the wrong tool.

## Using it with Strands Agents

```python
from strands import Agent
from strands_plan_tool.strands_adapter import WorkflowPlanPlugin

agent = Agent(
    tools=[read_notice_board, consult_bestiary, pack_satchel],
    plugins=[WorkflowPlanPlugin()],
)
```

That registers one extra tool, `submit_workflow_plan`. Tool choice stays automatic, so the
model may plan or may keep calling tools one at a time.

### Approval-gated tools cannot be planned

Verified against `strands-agents` 1.55.1 on 2026-09-15. A direct tool call — the path a
plan uses — refuses interrupts at two points in `strands/tools/_caller.py`:

- if the agent is already interrupted, any direct call raises
  `RuntimeError("cannot directly call tool during interrupt")`;
- if a tool raises an interrupt during a direct call, the caller raises
  `RuntimeError("cannot raise interrupt in direct tool call")` — and first calls
  `_InterruptState.deactivate()`, which **clears** the agent's interrupts and context.

That second behaviour is why plannability is decided *before* execution: by the time the
error surfaces, the interrupt state is already gone, so catching it is too late. A tool that
can request approval is refused at validation, and the model is told to call it directly.
The gate keeps working; it just is not batchable.

Because this package cannot tell from outside which tools an intervention gates, attaching
to an agent that has interventions requires an explicit list, using the same pattern syntax
as `HumanInTheLoop.allowed_tools`:

```python
WorkflowPlanPlugin(plannable=["*", "!delete_files"])
```

Leave it out on an agent with interventions and construction fails loudly rather than
silently batching something that should have prompted a human.

## Scheduling

The plan is a **DAG**, not a tree — a step may have several dependencies and several
dependents, so in-degree above one is normal and tree structures are the wrong shape.
Ordering comes from Kahn's algorithm over in-degree counts. A DFS topological sort is the
one choice to avoid: it produces a single linear sequence, and running a sequence serialises
steps that never depended on each other.

Levels are how a plan is *described* — the level count is its depth, and that is the number
of model round trips it replaces. Levels are not how it is *run*: a level barrier makes
every step in level N+1 wait for the slowest step in level N even when it never depended on
it. Execution is therefore barrier-free, with each step awaiting exactly its own
dependencies and starting the instant its last one lands.

Cost is O(1) amortised per step and per edge, O(V + E) overall, which is optimal — a plan
cannot be validated without being read. Wall clock is bound by the critical path rather
than by the sum of per-level maxima.

## Development

```bash
uv sync --extra dev
uv run ruff check . && uv run ruff format --check .
uv run mypy src tests demo.py
uv run pytest -q
```

### A final plan ends the turn

A plan that sets `final` asserts its returned values answer the request. The plugin takes
the model at its word and ends the turn there, rather than sampling once more purely to have
the model restate the result. That is the difference between two round trips and one — and
between 5310 tokens and 2522 at depth 5, because the restatement is what carried the cost.

The trade is real and visible: the closing message becomes the plan's returned values as
JSON rather than prose.

```python
WorkflowPlanPlugin(honor_final=False)  # keep the prose, pay the extra round trip
```

The early exit is only armed when every step succeeded. Ending a turn on a partial result
would hide the failure from the model, which is the one thing the ledger exists to prevent.

## Status

The engine, its guards, the scheduler, and the Strands adapter are implemented and tested
(68 tests), and the latency claim is measured against a live Bedrock model — see the table
above. Two bugs the benchmark caught that the unit suite had not, both now covered by
regression tests:

- a `@tool` returning a dict arrives as a JSON *string* inside a content block, so the
  obvious binding (`steps.board.beast`) could not resolve until `unwrap_tool_result` decoded
  it — and a `status="error"` result was being passed through as if it were data;
- the SDK hands a nested Pydantic tool parameter through as the raw decoded dict, so
  `plan.steps` raised `AttributeError` against a live model while every test stayed green,
  because the tests called `run_plan` with an already-typed `WorkflowPlan`. Reported upstream
  as [harness-sdk#4383](https://github.com/strands-agents/harness-sdk/issues/4383);
  `coerce_plan` handles it locally either way.
