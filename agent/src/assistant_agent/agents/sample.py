from pathlib import Path

import deepagents
from langchain.agents.middleware import AgentMiddleware
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.store.base import BaseStore

from assistant_agent.agents import COMMON_TOOLS, build_backend
from assistant_agent.utils.context import CommonContext
from assistant_agent.utils.workflow import Agent, DefaultRolloverStrategy

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

## コード生成・実行時
python を優先し使用せよ。環境構築には uv が使用可能。
実行する場合はルートディレクトリ直下にディレクトリを作成し、仮想環境を構築して実行してください。
"""


class _NoopSummarizationMiddleware(AgentMiddleware):
    """`create_deep_agent()` のベーススタックの SummarizationMiddleware を無効化する no-op 実装."""

    @property
    def name(self) -> str:
        """`.name` の一致で `_apply_custom_middleware()` に置き換えさせるための固定名."""
        return "SummarizationMiddleware"


def build_agent(
    llm: BaseChatModel,
    context: CommonContext,
    checkpointer: BaseCheckpointSaver,
    store: BaseStore,
    agent_id: str,
    thread_id: str | None,
) -> Agent:
    """ツール・checkpointer/store を束ねたグラフから Agent を構築する.

    LLM・agent_id は呼び出し元から受取。
    """
    backend = build_backend(agent_id)
    prof_dir = Path(__file__) / "../../../../.assistant_agent/"
    prof_dir = prof_dir.resolve()
    cwd = prof_dir / f"agent_{agent_id}"
    cwd = str(cwd)

    lc_agent = deepagents.create_deep_agent(
        llm,
        COMMON_TOOLS,
        skills=["/skills"],
        backend=backend,
        system_prompt=SYSTEM_PROMPT.format(cwd=cwd),
        context_schema=CommonContext,
        checkpointer=checkpointer,
        store=store,
        name="SampleAgent",
    )
    rollover_strategy = DefaultRolloverStrategy(llm, backend)
    return Agent(
        lc_agent,
        context=context,
        agent_id=agent_id,
        thread_id=thread_id,
        rollover_strategy=rollover_strategy,
    )
