import mlflow.models

from assistant_agent import agents

agent_wrapped = agents.build_agent()
mlflow.models.set_model(agent_wrapped)
