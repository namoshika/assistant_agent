import importlib
import importlib.util
import os
from pathlib import Path

from deepagents.backends import (
    BackendProtocol,
    CompositeBackend,
    FilesystemBackend,
    LocalShellBackend,
)
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_tavily import (
    TavilyCrawl,
    TavilyExtract,
    TavilyGetResearch,
    TavilyMap,
    TavilyResearch,
    TavilySearch,
)
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.store.base import BaseStore

from assistant_agent.tools import discord, dispatcher, obsidian, utils
from assistant_agent.utils.context import CommonContext
from assistant_agent.utils.workflow import Agent

COMMON_TOOLS = [
    # Utils
    utils.get_weather,
    utils.get_datetime_now,
    utils.web_fetch,
    dispatcher.dispatcher_get,
    dispatcher.dispatcher_list,
    dispatcher.dispatcher_invoke_at,
    dispatcher.dispatcher_invoke_delay,
    dispatcher.dispatcher_cancel,
    # Retriever (Tavily)
    TavilySearch(topic="general", country="japan"),
    TavilyExtract(),
    TavilyCrawl(),
    TavilyMap(),
    TavilyResearch(),
    TavilyGetResearch(),
    # Retriever (Obsidian)
    obsidian.obsidian_vault_search,
    obsidian.obsidian_vault_get,
    # Channel (Discord)
    discord.discord_get_messages,
    discord.discord_send_message,
    discord.discord_mention_user,
    discord.discord_reply_message,
]


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
    """module_name に対応するモジュールの build_agent を使い Agent を組み立てて返す.

    agent_id は thread_id prefix・backend namespace・Agent の識別子に使う。
    """
    if not module_exists(module_name):
        raise ValueError(f"Unknown agent module: {module_name!r}")
    module = importlib.import_module(f"{__name__}.{module_name}")
    return module.build_agent(llm, context, checkpointer, store, agent_id, thread_id)


def build_backend(agent_id: str) -> BackendProtocol:
    """エージェント用の backend を構築する."""
    profile_dir = Path(os.path.expanduser(os.getenv("AA_PROFILE_DIR", "~/.assistant_agent/")))
    profile_dir = profile_dir.resolve()

    agent_dir = profile_dir / f"agent_{agent_id}"
    agent_dir.mkdir(parents=True, exist_ok=True)
    skills_dir = profile_dir / "skills"
    skills_dir.mkdir(parents=True, exist_ok=True)

    return CompositeBackend(
        default=LocalShellBackend(agent_dir, inherit_env=True),
        routes={
            "/skills/": FilesystemBackend(skills_dir),
        },
    )
