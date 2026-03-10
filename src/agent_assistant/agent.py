import mlflow.models
from agent_assistant.utils.mlflow import LangGraphWrapper
from agent_assistant import connector, context, graph

AGENT_NAME = "agent"
ctx = context.build_session()
agent = graph.build_graph(AGENT_NAME, connector.get_llm(), connector.get_tools())
agent_wrapped = LangGraphWrapper(agent, ctx)
mlflow.models.set_model(agent_wrapped)
