from datetime import datetime
from typing import Annotated, TypedDict

from langchain.tools import ToolRuntime, tool
from langchain_core.messages import HumanMessage
from langchain_core.prompts import PromptTemplate
from pydantic import Field

from assistant_agent.services.dispatcher import Dispatch, DispatcherService, IntervalUnit
from assistant_agent.utils.absclass import AgentInvocation

_PROMPT_DESCRIPTION = "エージェントへの指示を入力。具体的な指示を入力してください。"


class DispatcherContext(TypedDict):
    dispatcher_service: DispatcherService


@tool
async def dispatcher_invoke_at(
    prompt: Annotated[str, Field(description=_PROMPT_DESCRIPTION)],
    at: datetime,
    runtime: ToolRuntime[DispatcherContext],
) -> str:
    """指定日時に指示メッセージを発信する予定を登録する. 過去の日時は指定できない."""
    service = runtime.context["dispatcher_service"]
    invocation = AgentInvocation(input={"messages": [HumanMessage(content=prompt)]})
    try:
        dispatch = service.invoke_at(invocation, at)
        return _format_response(str(dispatch), "The registered dispatch.")
    except ValueError as e:
        return _format_response(str(e), "Failed to register the dispatch.")


@tool
async def dispatcher_invoke_delay(
    prompt: Annotated[str, Field(description=_PROMPT_DESCRIPTION)],
    runtime: ToolRuntime[DispatcherContext],
    delay_value: int = 10,
    delay_unit: IntervalUnit = IntervalUnit.SECONDS,
    interval_value: int = -1,
    interval_unit: IntervalUnit = IntervalUnit.MINUTES,
) -> str:
    """開始オフセット経過後に指示メッセージを発信する予定を登録する.

    interval_value に 0 以下を指定すると単発発信になる。既定値のまま呼べば
    「10秒後に単発発信」（即時発信相当）になる。
    """
    service = runtime.context["dispatcher_service"]
    invocation = AgentInvocation(input={"messages": [HumanMessage(content=prompt)]})
    dispatch = service.invoke_delay(
        invocation, delay_value, delay_unit, interval_value, interval_unit
    )
    return _format_response(str(dispatch), "The registered dispatch.")


@tool
async def dispatcher_cancel(dispatch_id: str, runtime: ToolRuntime[DispatcherContext]) -> str:
    """登録済みの予定を dispatch_id を指定して解除する."""
    service = runtime.context["dispatcher_service"]
    cancelled = service.cancel_dispatch(dispatch_id)
    return _format_response(str(cancelled), "Whether the dispatch was found and cancelled.")


@tool(response_format="content_and_artifact")
async def dispatcher_list(runtime: ToolRuntime[DispatcherContext]) -> tuple[str, list[Dispatch]]:
    """登録済みの予定一覧を取得する."""
    service = runtime.context["dispatcher_service"]
    dispatches = service.list_dispatch()
    response = "\n".join(str(d) for d in dispatches) if dispatches else "(no dispatches)"
    return _format_response(response, "List of the currently registered dispatches."), dispatches


def _format_response(response: str, description: str) -> str:
    response = f"{'\n' if '\n' in response else ''}{response}"
    prompt = PromptTemplate.from_template("Description: {description}\nResponse: {response}")
    return prompt.format(description=description, response=response)
