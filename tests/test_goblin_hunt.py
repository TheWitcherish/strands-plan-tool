from examples.goblin_hunt import LAIR, PLAN, local_invoker
from strands_plan_tool import StepStatus, execute_plan


async def test_the_mage_takes_only_the_road_to_the_goblins_lair() -> None:
    result = await execute_plan(PLAN, local_invoker)

    status = {outcome.id: outcome.status for outcome in result.ledger}
    taken = "castle" if LAIR == "castle" else "forest"
    untaken = "forest" if taken == "castle" else "castle"
    assert status[taken] is StepStatus.OK
    assert status[untaken] is StepStatus.NOT_TAKEN
    assert (status["attack"] is StepStatus.OK) == (taken == "castle")
