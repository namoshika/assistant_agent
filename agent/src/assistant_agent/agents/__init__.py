import importlib
import importlib.util

from deepagents.backends.store import StoreBackend
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.store.base import BaseStore

from assistant_agent.utils.context import CommonContext
from assistant_agent.utils.workflow import Agent, DefaultRolloverStrategy


def module_exists(module_name: str) -> bool:
    """agents/ 配下に module_name.py が存在するか、import せずに判定する."""
    return importlib.util.find_spec(f"{__name__}.{module_name}") is not None


def get_agent(
    module_name: str,
    agent_id: str,
    thread_id: str | None,
    llm: BaseChatModel,
    context: CommonContext,
    checkpointer: BaseCheckpointSaver,
    store: BaseStore,
) -> Agent:
    """module_name に対応するモジュールの build_lc_agent を使い Agent を組み立てて返す.

    agent_id は thread_id prefix・backend namespace・Agent の識別子に使う。
    """
    if not module_exists(module_name):
        raise ValueError(f"Unknown agent module: {module_name!r}")
    module = importlib.import_module(f"{__name__}.{module_name}")
    lc_agent = module.build_lc_agent(checkpointer, store, llm, agent_id)

    backend = StoreBackend(store=store, namespace=lambda _rt: (agent_id, "filesystem"))
    rollover_strategy = DefaultRolloverStrategy(llm, backend)
    return Agent(
        lc_agent,
        context=context,
        agent_id=agent_id,
        thread_id=thread_id,
        rollover_strategy=rollover_strategy,
    )
