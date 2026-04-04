import mlflow.models

from agent_assistant import agents

agent_wrapped = agents.build_agent()
mlflow.models.set_model(agent_wrapped)
