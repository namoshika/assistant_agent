from unittest.mock import MagicMock

from langchain_core.messages import HumanMessage

from assistant_agent.utils.absclass import Channel


class _DummyChannel(Channel):
    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        pass


class TestChannel:
    def test_publish_01(self):
        """subscribe/publish の配信挙動を確認.

        観点1: 同一コールバックを複数回登録しても publish で1回しか呼ばれない
        観点2: 異なるコールバックを複数登録すると publish で全員が呼ばれる
        """
        # 試験準備
        channel = _DummyChannel()
        cb1 = MagicMock()
        cb2 = MagicMock()
        msg = HumanMessage(content="hello")

        # 試験実施
        channel.subscribe(cb1)
        channel.subscribe(cb1)
        channel.subscribe(cb2)
        channel.publish(msg)

        # 結果検証
        # 観点1
        cb1.assert_called_once_with(msg)
        # 観点2
        cb2.assert_called_once_with(msg)
