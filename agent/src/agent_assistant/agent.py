import mlflow.models

# from mlflow.types.responses import ResponsesAgentRequest, ResponsesAgentResponse
from agent_assistant import connector, context, graph
from agent_assistant.utils.mlflow import LangGraphChatAgent

AGENT_NAME = "agent"
ctx = context.build_session()
agent = graph.build_graph(AGENT_NAME, ctx.llm, connector.get_tools())
agent_wrapped = LangGraphChatAgent(agent, ctx)  # pyright: ignore[reportArgumentType]

# MLflow integration (log model)
mlflow.models.set_model(agent_wrapped)
