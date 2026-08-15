from pathlib import Path

import deepagents
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

## ファイル読み書き・シェルコマンド実行
ファイルパス指定はできる限り、相対パスを使用せよ。
カレントディレクトリは可能な限り `{cwd}` に設定せよ。
"""


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
