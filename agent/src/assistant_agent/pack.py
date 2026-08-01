import mlflow.models

from assistant_agent import agents
from assistant_agent.utils.mlflow import LangGraphChatAgent

lc_agent, ctx = agents.build_agent()
agent_wrapped = LangGraphChatAgent(lc_agent, ctx)
mlflow.models.set_model(agent_wrapped)
