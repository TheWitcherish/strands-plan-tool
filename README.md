# strands-plan-tool

When your agent needs three tool calls in a row, you wait for the model three times — not
because it is thinking, but because it has to be *asked* again before it can make the next
call.

This adds one tool that lets the model hand over the whole chain at once. A dependency chain
of depth D costs **one** model round trip instead of D.

```bash
uv add strands-plan-tool
```

## A complete example

A Game Master resolving an attack: look up the monster, roll against its armour, apply the
damage. Three tools, each needing the one before it.

Needs AWS credentials and Bedrock model access — `export AWS_PROFILE=...` and
`export AWS_REGION=...`. Pass `Agent(model=BedrockModel(model_id=...))` to pick a model
explicitly; omitted here, so Strands uses its default.

```python
from strands import Agent, tool

from strands_plan_tool.strands_adapter import WorkflowPlanPlugin

BESTIARY = {"goblin": {"name": "goblin", "hp": 7, "armour": 15}}


@tool
def look_up_monster(name: str) -> dict[str, object]:
    """Look up a monster's stat block.

    Args:
        name: Monster name, e.g. "goblin".

    Returns:
        An object shaped {"name": str, "hp": int, "armour": int}.
    """
    return dict(BESTIARY[name.lower()])


@tool
def roll_attack(armour: int) -> dict[str, object]:
    """Roll an attack against an armour value.

    Args:
        armour: The target's armour, from look_up_monster's "armour" field.

    Returns:
        An object shaped {"roll": int, "hits": bool, "damage": int}.
    """
    roll = 18
    return {"roll": roll, "hits": roll >= armour, "damage": 6 if roll >= armour else 0}


@tool
def apply_damage(monster: str, damage: int) -> dict[str, object]:
    """Apply damage to a monster.

    Args:
        monster: Monster name, from look_up_monster's "name" field.
        damage: Damage dealt, from roll_attack's "damage" field.

    Returns:
        An object shaped {"hp_after": int, "defeated": bool}.
    """
    hp_after = max(BESTIARY[monster.lower()]["hp"] - damage, 0)
    return {"hp_after": hp_after, "defeated": hp_after == 0}


agent = Agent(
    system_prompt=(
        "You are a Game Master for a tabletop role-playing game.\n\n"
        "You have a submit_workflow_plan tool. When you already know which tools to call "
        "and each call's arguments come from an earlier call's result, submit one plan "
        "instead of calling the tools one at a time. Each step's `bind` maps an argument "
        'to a JSONata expression over {"steps": {<id>: <result>}}, for example '
        '{"armour": "steps.monster.armour"}. Read each tool\'s documented return shape.'
    ),
    tools=[look_up_monster, roll_attack, apply_damage],
    plugins=[WorkflowPlanPlugin()],
)

response = agent("The rogue attacks the goblin. Resolve the attack and tell me the outcome.")

print(response.message["content"][0]["text"])
print(f"model round trips: {response.metrics.cycle_count}  (a plain loop needs 4)")
```

```
Tool #1: submit_workflow_plan

{"lookup": {"name": "goblin", "hp": 7, "armour": 15},
 "attack": {"roll": 18, "hits": true, "damage": 6},
 "damage": {"hp_after": 1, "defeated": false}}
model round trips: 1  (a plain loop needs 4)
```

One tool call. Three tools ran. You wrote no plan — the model did.

The integration is the two lines you already saw: import `WorkflowPlanPlugin`, pass it in
`plugins=`. Tool choice stays automatic, so the model plans when it helps and ignores the
tool when it does not.

## Three things to know

**1. Document your tools' return shapes.** Tool schemas describe *inputs* only. A model
binding to `steps.lookup.ac` when your field is `armour` is your docstring's fault, not the
model's — so every tool above ends with a `Returns:` line naming the exact shape. This is the
single biggest factor in whether plans work for you.

**2. Tell the model when to plan.** One paragraph in the system prompt, as above. Without it
the model usually keeps calling tools one at a time — not wrong, just slower.

**3. A `final` plan ends the turn, so the answer is JSON.** The model marked its plan final,
meaning its results *are* the answer, so the turn ended there instead of sampling the model
again to restate them. That is the round trip you saved. It also means the final message is
appended rather than streamed, which is why the example prints it explicitly. Want prose?
`WorkflowPlanPlugin(honor_final=False)` — one more round trip, nicer output.

Also expect the JSON keys to vary: `lookup` and `attack` are step ids the *model* chose. Three
runs of this exact example produced `lookup`, `monster` and `goblin_stats` for the first step.
The values are stable; the names are its own.

## When not to bother

| Shape | Plan? |
|---|---|
| One tool call | No — the plan schema costs more tokens than it saves |
| Two or more *mechanically* dependent calls | Yes — this is the point |
| A lookup, then a fan-out over its results | Yes — the most common winning shape |
| Independent calls in one turn | No — already concurrent in the agent loop |
| Each result needs your judgement before the next | No — call them one at a time |

Break-even is depth 2.

## Measured

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
clock and under a third of the tokens. At depth 1 the model **declined to plan** unprompted
(`pl% = 0`), which is the correct call. Reproduce with
`AWS_PROFILE=... AWS_REGION=... uv run python -m bench.breakeven`.

## Safety

JSONata is a query language: no imports, no filesystem, no network, no subprocess. Its one
dynamic surface is `$eval` — which evaluates further JSONata, never Python — and it is off by
default, so a plan's bindings can be read completely before anything runs.

Cycles, duplicate ids, dangling references and step-count overruns are rejected **before** any
tool runs, so a bad plan costs one cheap repair round trip and leaves no side effects. A
binding that fails is a step failure, never a silently substituted `None`.

**Approval-gated tools cannot be planned.** A direct tool call refuses interrupts, so such a
tool is rejected at validation and the model is told to call it directly — the gate keeps
working, it just is not batchable. If your agent has interventions attached, name what is safe
to batch: `WorkflowPlanPlugin(plannable=["*", "!delete_files"])`. Leave it out and
construction fails loudly rather than silently skipping a human approval.

## Going further

A bigger shape — one RPG combat round for three players, in `examples/rpg.py`:

```
L0  read_action_log                      1 step
L1  get_character   x3   (fan-out)       3 steps
L2  roll_check      x3                   3 steps
L3  resolve_action  x3                   3 steps
L4  tally_round          (join, in-degree 3)
L5  narrate_round
```

```bash
uv run python -m examples.game_master          # offline, deterministic
uv run python -m examples.game_master --live   # the model writes its own plan
```

Twelve steps, depth 6, three wide, and ten of the twelve results never reach the model. The
join is where JSONata beats a path syntax: `tally_round` wants one *array* of damages living in
three different steps, so a binding builds it inline —
`{"damages": "[steps.a.damage, steps.b.damage, steps.c.damage]"}`. No glue tool, no extra
round trip.

Run it `--live` and the model does something worth seeing: it calls `read_action_log`
directly first, because it cannot know how many characters to fetch until it has seen the
actions, then plans the eleven mechanical steps that follow. It uses the loop where judgement
is needed and the plan where it is not.

You can also drive the engine without an agent at all — `execute_plan(plan, invoker)` takes
any async `(name, args) -> result` callable, which is how the tests and the offline example
run with no model and no credentials.

## Development

Tool invocation sits behind a `ToolInvoker` protocol, so the engine imports no agent
framework — steps run in-process with no hop to the model provider between them. Ordering is
Kahn's algorithm over in-degree counts and execution is barrier-free: each step starts the
instant its own dependencies land, so wall clock tracks the critical path.

```bash
uv sync --extra dev
uv run ruff check . && uv run ruff format --check .
uv run mypy src tests examples bench demo.py demo_bedrock.py
uv run pytest -q
```

MIT licensed.
