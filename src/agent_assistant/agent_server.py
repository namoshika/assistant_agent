import mlflow
from mlflow.genai.agent_server import AgentServer

# import することで API を FastAPI へ登録
import agent_assistant.agent  # noqa: F401

agent_server = AgentServer("ResponsesAgent")
app = agent_server.app


def serve():
    """AgentServer を起動."""
    mlflow.set_experiment("agent-rag")
    mlflow.autolog()
    agent_server.run("agent_assistant.agent_server:app")


if __name__ == "__main__":
    serve()
