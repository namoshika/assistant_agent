import json

import pytest
from fastapi.testclient import TestClient


@pytest.mark.integration
def test_chat_completions_01():
    """クライアントからのリクエストを正しく応答できるか確認.

    観点1: GET /api/models でモデル一覧が返ること (model_id が含まれる)
    観点2: POST /api/chat/completions (非ストリーム) でエージェントが呼び出されレスポンスが返ること
    """
    # 試験準備
    from assistant_agent.agent_server import app

    with TestClient(app) as client:
        # 観点1: モデル一覧
        # 試験実施
        models_resp = client.get("/api/models")

        # 結果検証 (観点1)
        assert models_resp.status_code == 200
        model_ids = {m["id"] for m in models_resp.json()["data"]}
        assert "assistant_agent_v1" in model_ids

        # 観点2: 非ストリーミング
        # 試験実施
        resp = client.post(
            "/api/chat/completions",
            json={
                "model": "assistant_agent_v1",
                "messages": [{"role": "user", "content": "こんにちは"}],
            },
        )

        # 結果検証 (観点2)
        assert resp.status_code == 200
        body = resp.json()
        assert body["object"] == "chat.completion"
        assert body["model"] == "assistant_agent_v1"
        assert body["choices"][0]["message"]["role"] == "assistant"
        assert body["choices"][0]["message"]["content"]

        # 観点2: 会話履歴を含む複数メッセージ
        # 試験実施
        history_resp = client.post(
            "/api/chat/completions",
            json={
                "model": "assistant_agent_v1",
                "messages": [
                    {"role": "user", "content": "私の好きな色は青です。覚えておいてください。"},
                    {"role": "assistant", "content": "承知しました。"},
                    {"role": "user", "content": "先ほど伝えた好きな色は何でしたか？"},
                ],
            },
        )

        # 結果検証 (観点2)
        assert history_resp.status_code == 200
        history_body = history_resp.json()
        assert history_body["choices"][0]["message"]["role"] == "assistant"
        assert history_body["choices"][0]["message"]["content"]


@pytest.mark.integration
def test_chat_completions_stream_01():
    """ストリーミングでクライアントからのリクエストを正しく応答できるか確認.

    観点1: stream=true でリクエストすると text/event-stream のレスポンスが返ること
    観点2: SSE の各行が ChatCompletionChunk としてパース可能で、[DONE] で終端すること
    """
    # 試験準備
    from assistant_agent.agent_server import app

    with TestClient(app) as client:
        # 試験実施
        with client.stream(
            "POST",
            "/api/chat/completions",
            json={
                "model": "assistant_agent_v1",
                "messages": [{"role": "user", "content": "こんにちは"}],
                "stream": True,
            },
        ) as resp:
            # 結果検証 (観点1)
            assert resp.status_code == 200
            assert resp.headers["content-type"].startswith("text/event-stream")
            lines = [line for line in resp.iter_lines() if line]

        # 結果検証 (観点2)
        assert lines[-1] == "data: [DONE]"
        chunks = [json.loads(line.removeprefix("data: ")) for line in lines[:-1]]
        assert all(chunk["object"] == "chat.completion.chunk" for chunk in chunks)
        assert any(chunk["choices"][0]["delta"]["content"] for chunk in chunks)
