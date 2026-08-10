from datetime import UTC, datetime
from typing import Annotated, TypedDict
from zoneinfo import ZoneInfo

from langchain.tools import ToolRuntime, tool
from langchain_core.prompts import PromptTemplate
from pydantic import Field

from assistant_agent.services.dispatcher import Dispatch, DispatcherService, IntervalUnit

_PROMPT_DESCRIPTION = "エージェントへの指示を入力。具体的な指示を入力してください。"
_AT_DESCRIPTION = (
    "発信する日時。タイムゾーンの指定を省略した場合は JST（Asia/Tokyo）として解釈する。"
)
_RESPONSE_TMPL = """
# {title}
{description}

{response}
"""


class DispatcherContext(TypedDict):
    dispatcher_service: DispatcherService
    agent_id: str


@tool
async def dispatcher_invoke_at(
    prompt: Annotated[str, Field(description=_PROMPT_DESCRIPTION)],
    at: Annotated[datetime, Field(description=_AT_DESCRIPTION)],
    runtime: ToolRuntime[DispatcherContext],
) -> str:
    """指定日時に指示メッセージを発信する予定を登録する."""
    if at.tzinfo is None:
        at = at.replace(tzinfo=ZoneInfo("Asia/Tokyo"))
    at = at.astimezone(UTC)
    service = runtime.context["dispatcher_service"]
    agent_id = runtime.context["agent_id"]
    dispatch = await service.invoke_at(agent_id, prompt, at)
    return PromptTemplate.from_template(_RESPONSE_TMPL).format(
        title="Dispatcher invoke_at", description="The registered dispatch.", response=str(dispatch)
    )


@tool
async def dispatcher_invoke_delay(
    prompt: Annotated[str, Field(description=_PROMPT_DESCRIPTION)],
    runtime: ToolRuntime[DispatcherContext],
    delay_value: int = 10,
    delay_unit: IntervalUnit = IntervalUnit.SECONDS,
    interval_value: int = 0,
    interval_unit: IntervalUnit = IntervalUnit.MINUTES,
) -> str:
    """開始オフセット経過後に指示メッセージを発信する予定を登録する.

    interval_value に 0 以下を指定すると単発発信になる。既定値のまま呼べば
    「10秒後に単発発信」（即時発信相当）になる。
    """
    service = runtime.context["dispatcher_service"]
    agent_id = runtime.context["agent_id"]
    dispatch = await service.invoke_delay(
        agent_id, prompt, delay_value, delay_unit, interval_value, interval_unit
    )
    return PromptTemplate.from_template(_RESPONSE_TMPL).format(
        title="Dispatcher invoke_delay",
        description="The registered dispatch.",
        response=str(dispatch),
    )


@tool
async def dispatcher_get(dispatch_id: str, runtime: ToolRuntime[DispatcherContext]) -> str:
    """dispatch_id を指定して予定1件を取得する. メッセージ内容は全文を返す."""
    service = runtime.context["dispatcher_service"]
    agent_id = runtime.context["agent_id"]
    dispatch = await service.get_dispatch(agent_id, dispatch_id)
    response = str(dispatch) if dispatch is not None else "(dispatch not found)"
    return PromptTemplate.from_template(_RESPONSE_TMPL).format(
        title="Dispatcher get", description="The requested dispatch, if found.", response=response
    )


@tool(response_format="content_and_artifact")
async def dispatcher_cancel(
    dispatch_id: str, runtime: ToolRuntime[DispatcherContext]
) -> tuple[str, bool]:
    """登録済みの予定を dispatch_id を指定して解除する."""
    service = runtime.context["dispatcher_service"]
    agent_id = runtime.context["agent_id"]
    cancelled = await service.cancel_dispatch(agent_id, dispatch_id)
    return "Success" if cancelled else "Failure", cancelled


@tool(response_format="content_and_artifact")
async def dispatcher_list(runtime: ToolRuntime[DispatcherContext]) -> tuple[str, list[Dispatch]]:
    """登録済みの予定一覧を取得する."""
    service = runtime.context["dispatcher_service"]
    agent_id = runtime.context["agent_id"]
    dispatches = await service.list_dispatch(agent_id)
    response = "\n\n".join(str(d)[:200] for d in dispatches) if dispatches else "(no dispatches)"
    formatted = PromptTemplate.from_template(_RESPONSE_TMPL).format(
        title="Dispatcher list",
        description="List of the currently registered dispatches.",
        response=response,
    )
    return formatted, dispatches
