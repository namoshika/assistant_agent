from typing import Any
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langchain.agents import create_agent
from langgraph.graph.state import CompiledStateGraph
from agent_assistant.context import ContextSchema

SYSTEM_PROMPT = """
You are a helpful assistant.
Respond to the user in Japanese.
"""


def build_graph(
    agent_name: str, llm: BaseChatModel, tools: list[BaseTool]
) -> CompiledStateGraph[Any, ContextSchema, Any, Any]:
    agentic_graph = create_agent(
        model=llm,
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
        context_schema=ContextSchema,
        name=agent_name,
    )
    return agentic_graph
