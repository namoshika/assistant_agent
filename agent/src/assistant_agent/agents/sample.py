from typing import Any

import deepagents
from deepagents.backends.store import StoreBackend
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph.state import CompiledStateGraph
from langgraph.store.base import BaseStore

from assistant_agent import subagents
from assistant_agent.tools import discord, dispatcher, obsidian, sample
from assistant_agent.utils.context import CommonContext

AGENT_ID = "assistant-agent-1"  # thread_id の prefix・StoreBackend の namespace に使う識別子

SYSTEM_PROMPT = """
# Instruction
あなたは親身なエージェントで、名前は assistant_agent_1 です。
ユーザの必要に対してツールを使いながら応えてください。

# Guideline
エージェントはいくつかのチャンネルから外界のイベントを入力されます。
イベントから外界の状態を理解し、行動してください。行動後、以下を出力してください。

## 入力例:

```
# (Input Channel Name)
## Guideline
(チャンネルに関する説明)

## (Input Item 1)
(チャンネルからの入力)

## (Input Item 2)
...
```

## 出力例:

```
# Activity Log

## Observe
(観測した事実)

## Understand
(現在の状況・目的)

## Consider
(判断・方針)

## Act
(実行したこと)

## Verify
(結果・確認)

## Report
(ユーザーへの報告)
```
"""


def build_lc_agent(
    checkpointer: BaseCheckpointSaver,
    store: BaseStore,
    llm: BaseChatModel,
) -> CompiledStateGraph[Any, CommonContext, Any, Any]:
    """ツール・checkpointer/store を束ねたグラフを構築する（LLM は呼び出し元から受け取る）."""
    backend = StoreBackend(store=store, namespace=lambda _rt: (AGENT_ID, "filesystem"))
    lc_agent = deepagents.create_deep_agent(
        model=llm,
        tools=[
            sample.get_weather,
            sample.get_datetime_now,
            dispatcher.dispatcher_invoke_at,
            dispatcher.dispatcher_invoke_delay,
            dispatcher.dispatcher_cancel,
            dispatcher.dispatcher_list,
            obsidian.obsidian_vault_search,
            obsidian.obsidian_vault_get,
            discord.discord_get_messages,
            discord.discord_send_message,
            discord.discord_mention_user,
            discord.discord_reply_message,
        ],
        subagents=[
            subagents.web_researcher,
        ],
        backend=backend,
        system_prompt=SYSTEM_PROMPT,
        context_schema=CommonContext,
        name="SampleAgent",
        checkpointer=checkpointer,
        store=store,
    )
    return lc_agent
