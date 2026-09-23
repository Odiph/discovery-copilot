"""`app/main.py::_cancel_and_await` -- independent review finding C.

`_async_main`'s `finally` block cancels `status_poll_task` and awaits it
BEFORE `pair.supervisor.stop()`, `pa.terminate()`, the embedded server's own
stop, and `store.close()`. A bare `await task` there would re-raise whatever
the task had already failed with (anything other than `CancelledError`),
aborting the `finally` block and skipping every step listed above -- on the
one code path whose whole purpose is an ordered, clean close (SESS-01).
These tests exercise `_cancel_and_await` in isolation, without needing a real
capture/STT/server stack around it.
"""

from __future__ import annotations

import asyncio
import logging

from app.main import _cancel_and_await


def test_a_task_that_already_failed_with_a_non_cancelled_exception_is_absorbed_not_reraised(caplog):
    async def _boom() -> None:
        raise RuntimeError("simulated status-poll-loop bug")

    async def scenario() -> None:
        task = asyncio.create_task(_boom())
        await asyncio.sleep(0)  # let the task actually fail before we cancel/await it
        # Must not raise -- this is the exact call site shape used in the
        # `finally` block: cancel (a no-op on an already-finished task) then
        # await, absorbing the RuntimeError instead of letting it propagate.
        await _cancel_and_await(task, task_name="test-status-poll-task")

    with caplog.at_level(logging.WARNING):
        asyncio.run(scenario())  # would raise RuntimeError before the fix

    assert any(
        "test-status-poll-task" in record.getMessage() and "RuntimeError" in record.getMessage()
        for record in caplog.records
    ), "the failure must be logged by error TYPE and the task name, not silently dropped"


def test_a_still_running_task_is_cancelled_and_the_cancellation_is_suppressed():
    async def _sleep_forever() -> None:
        await asyncio.sleep(1000)

    async def scenario() -> None:
        task = asyncio.create_task(_sleep_forever())
        await asyncio.sleep(0)  # let the task actually start
        await _cancel_and_await(task, task_name="test-still-running-task")
        assert task.cancelled()

    asyncio.run(scenario())  # must return normally -- CancelledError stays internal


def test_a_task_that_finished_cleanly_before_being_cancelled_is_a_no_op():
    async def _return_value() -> str:
        return "done"

    async def scenario() -> None:
        task = asyncio.create_task(_return_value())
        await asyncio.sleep(0)  # let it complete before cancel/await
        await _cancel_and_await(task, task_name="test-clean-task")
        assert task.result() == "done"

    asyncio.run(scenario())
