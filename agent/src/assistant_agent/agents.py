from langchain_core.language_models import BaseChatModel
from mlflow.pyfunc.model import ChatAgent

import assistant_agent.services  # noqa: F401  登録発火
from assistant_agent import graph, tools
from assistant_agent.utils import context, mlflow

AGENT_NAME = "agent"


def get_model() -> BaseChatModel:
    """LLM と埋め込みモデルを生成して返す."""
    from databricks_langchain import ChatDatabricks

    llm = ChatDatabricks(model="databricks-claude-haiku-4-5")
    return llm


def build_agent() -> ChatAgent:
    """エージェントがセッション開始した際の初期化を行う.

    Returns:
        初期化済みの ChatAgent インスタンス。

    """
    # コンテキスト初期化
    ctx = context.ContextRegistry.build({"sample_retriever": "qwen3emb06b"})

    # エージェント初期化
    agent = graph.build_graph(AGENT_NAME, get_model(), tools.get_tools())
    agent_wrapped = mlflow.LangGraphChatAgent(agent, ctx)

    return agent_wrapped
