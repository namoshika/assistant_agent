import json
import uuid

from fastapi import FastAPI
from fastapi.testclient import TestClient
from mlflow.types.agent import ChatAgentChunk, ChatAgentMessage, ChatAgentRequest, ChatAgentResponse
from mlflow.types.chat import ChatMessage, TextContentPart

from agent_assistant.utils.serving import (
    ChatCompletion,
    from_chat_agent_chunk,
    from_chat_agent_response,
    to_chat_agent_messages,
)


def test_to_chat_agent_messages_01():
    """to_chat_agent_messages に対しテストすること.

    観点1: string content と role が正しく変換されること (user・system ロール含む)
    観点2: list[TextContentPart] content がテキスト抽出・スペース結合されること
    """
    # 試験準備
    string_messages = [
        ChatMessage(role="user", content="こんにちは"),
        ChatMessage(role="system", content="あなたはアシスタントです"),
    ]
    part_messages = [
        ChatMessage(
            role="user",
            content=[
                TextContentPart(type="text", text="先月の"),
                TextContentPart(type="text", text="売上は？"),
            ],
        )
    ]

    # 試験実施
    result_string = to_chat_agent_messages(string_messages)
    result_parts = to_chat_agent_messages(part_messages)

    # 結果検証
    # 観点1
    assert result_string[0].role == "user"
    assert result_string[0].content == "こんにちは"
    assert result_string[1].role == "system"
    assert result_string[1].content == "あなたはアシスタントです"
    # 観点2
    assert result_parts[0].content == "先月の 売上は？"


def test_from_chat_agent_response_01():
    """from_chat_agent_response に対しテストすること.

    観点1: 単一 assistant メッセージから全フィールドが正しく生成されること (既定値フィールドを含む)
    観点2: 複数 assistant メッセージがスペース結合されること
    """
    # 試験準備
    single_response = ChatAgentResponse(
        messages=[ChatAgentMessage(id=str(uuid.uuid4()), role="assistant", content="テスト応答")]
    )
    multi_response = ChatAgentResponse(
        messages=[
            ChatAgentMessage(id=str(uuid.uuid4()), role="assistant", content="前半"),
            ChatAgentMessage(id=str(uuid.uuid4()), role="assistant", content="後半"),
        ]
    )

    # 試験実施
    result_single = from_chat_agent_response(single_response, "test-model")
    result_multi = from_chat_agent_response(multi_response, "test-model")

    # 結果検証
    # 観点1
    assert result_single.id.startswith("chatcmpl-")
    assert result_single.object == "chat.completion"
    assert isinstance(result_single.created, int)
    assert result_single.model == "test-model"
    assert result_single.choices[0].index == 0
    assert result_single.choices[0].message.role == "assistant"
    assert result_single.choices[0].message.content == "テスト応答"
    assert result_single.choices[0].finish_reason == "stop"
    assert result_single.choices[0].logprobs is None
    assert result_single.usage is None
    # 観点2
    assert result_multi.choices[0].message.content is not None
    assert "前半" in result_multi.choices[0].message.content
    assert "後半" in result_multi.choices[0].message.content


def test_from_chat_agent_chunk_01():
    """from_chat_agent_chunk に対しテストすること.

    観点1: SSE 形式で、パース後の全フィールドが正しいこと (既定値フィールドを含む)
    観点2: 日本語テキストが Unicode エスケープされないこと
    """
    # 試験準備
    chunk = ChatAgentChunk(
        delta=ChatAgentMessage(id=str(uuid.uuid4()), role="assistant", content="テスト")
    )
    chunk_ja = ChatAgentChunk(
        delta=ChatAgentMessage(id=str(uuid.uuid4()), role="assistant", content="日本語テキスト")
    )
    chunk_id = "chatcmpl-abc"

    # 試験実施
    line = from_chat_agent_chunk(chunk_id, chunk, "test-model")
    line_ja = from_chat_agent_chunk(chunk_id, chunk_ja, "test-model")

    # 結果検証
    # 観点1
    assert line.startswith("data: ")
    assert line.endswith("\n\n")
    body = json.loads(line.removeprefix("data: ").strip())
    assert body["id"] == chunk_id
    assert body["object"] == "chat.completion.chunk"
    assert isinstance(body["created"], int)
    assert body["model"] == "test-model"
    assert body["choices"][0]["index"] == 0
    assert body["choices"][0]["delta"]["content"] == "テスト"
    assert body["choices"][0]["finish_reason"] is None

    # 観点2
    assert "日本語テキスト" in line_ja
    assert "\\u" not in line_ja


class TestChatCompletion:
    def test_invoke_handler_01(self):
        """ChatCompletion._invoke_handler に対しテストすること.

        (o: ストリーム対応あり、x: ストリーム対応なし)
        観点1: client/agent のストリーム対応の有無の組み合わせ (o/x) ごとに正しく動作すること
               - client: o, agent: o → SSE ストリームレスポンスの全フィールドが正しいこと
               - client: o, agent: x → 同期フォールバックで通常レスポンスが返ること
               - client: x, agent: o → 通常レスポンスが返ること
               - client: x, agent: x → 通常レスポンスの全フィールドが正しいこと
        観点2: 未登録モデルへのリクエストが 503 を返すこと (stream の有無を問わず)
        """
        # 試験準備
        base_messages = [{"role": "user", "content": "質問"}]
        client_s = self._make_client(allow_stream=True)
        client_ns = self._make_client(allow_stream=False)
        stream_payload = {"model": "test-model", "messages": base_messages, "stream": True}
        normal_payload = {"model": "test-model", "messages": base_messages}

        # 試験実施
        resp_oo = client_s.post("/v1/chat/completions", json=stream_payload)  # case1: o, o
        resp_ox = client_ns.post("/v1/chat/completions", json=stream_payload)  # case2: o, x
        resp_xo = client_s.post("/v1/chat/completions", json=normal_payload)  # case3: x, o
        resp_xx = client_ns.post("/v1/chat/completions", json=normal_payload)  # case4: x, x

        # 結果検証
        # 観点1
        # case1: client: o, agent: o → SSE ストリーム
        assert resp_oo.status_code == 200
        assert "text/event-stream" in resp_oo.headers["content-type"]
        lines = [line for line in resp_oo.text.splitlines() if line]
        data_lines = [line for line in lines if line.startswith("data: ")]
        assert any(line == "data: [DONE]" for line in data_lines)
        chunk_line = next(line for line in data_lines if line != "data: [DONE]")
        chunk = json.loads(chunk_line.removeprefix("data: "))
        assert chunk["id"].startswith("chatcmpl-")
        assert chunk["object"] == "chat.completion.chunk"
        assert isinstance(chunk["created"], int)
        assert chunk["model"] == "test-model"
        assert chunk["choices"][0]["index"] == 0
        assert chunk["choices"][0]["delta"]["content"] == "ストリーム"
        assert chunk["choices"][0]["finish_reason"] is None

        # case2: client: o, agent: x → 同期フォールバック
        assert resp_ox.status_code == 200
        assert resp_ox.json()["object"] == "chat.completion"

        # case3: client: x, agent: o → 通常レスポンス
        assert resp_xo.status_code == 200
        assert resp_xo.json()["object"] == "chat.completion"

        # case4: client: x, agent: x → 通常レスポンス全フィールド
        assert resp_xx.status_code == 200
        body = resp_xx.json()
        assert body["id"].startswith("chatcmpl-")
        assert body["object"] == "chat.completion"
        assert isinstance(body["created"], int)
        assert body["model"] == "test-model"
        assert body["choices"][0]["index"] == 0
        assert body["choices"][0]["message"]["role"] == "assistant"
        assert body["choices"][0]["finish_reason"] == "stop"
        assert body["choices"][0]["logprobs"] is None
        assert body["usage"] is None

        # 観点2: 未登録モデル → 503
        app = FastAPI()
        ChatCompletion.bind(app)
        client_bare = TestClient(app)
        unknown_payload = {"model": "unknown-model", "messages": base_messages}
        assert client_bare.post("/v1/chat/completions", json=unknown_payload).status_code == 503
        assert (
            client_bare.post(
                "/v1/chat/completions", json={**unknown_payload, "stream": True}
            ).status_code
            == 503
        )

    def test_list_models_01(self):
        """ChatCompletion._list_models に対しテストすること.

        観点1: GET /v1/models のレスポンスに登録 model_id が含まれること
        """
        # 試験準備
        client = self._make_client(allow_stream=True)

        # 試験実施
        resp = client.get("/v1/models")

        # 結果検証
        # 観点1
        assert resp.status_code == 200
        model_ids = {m["id"] for m in resp.json()["data"]}
        assert "test-model" in model_ids

    @staticmethod
    def _make_client(allow_stream: bool) -> TestClient:
        """TestClient を生成するヘルパーメソッド.

        allow_stream=True のとき regist と regist_stream 両方を登録する。
        allow_stream=False のとき regist のみ登録する。
        """
        app = FastAPI()
        endpoint = ChatCompletion.bind(app)

        @endpoint.regist(model_id="test-model")
        def _predict(req: ChatAgentRequest) -> ChatAgentResponse:
            return ChatAgentResponse(
                messages=[ChatAgentMessage(id=str(uuid.uuid4()), role="assistant", content="応答")]
            )

        if allow_stream:

            @endpoint.regist_stream(model_id="test-model")
            def _stream(req: ChatAgentRequest):
                yield ChatAgentChunk(
                    delta=ChatAgentMessage(
                        id=str(uuid.uuid4()), role="assistant", content="ストリーム"
                    )
                )

        return TestClient(app)
