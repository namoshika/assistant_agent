import pytest
from fastapi.testclient import TestClient

from agent_assistant.agent_server import app


@pytest.fixture
def _client():
    """本番 app を使った TestClient を返すフィクスチャ."""
    return TestClient(app)


@pytest.mark.integration
def test_chat_completions_01(_client):
    """ChatCompletion エンドポイント エンドツーエンドテスト.

    観点1: GET /v1/models でモデル一覧が返ること (model_id が含まれる)
    観点2: POST /v1/chat/completions (非ストリーム) でエージェントが呼び出されレスポンスが返ること
    観点3: POST /v1/chat/completions (stream=True) でエージェント関数が呼び出され、SSE が返ること
    """
    # 観点1: モデル一覧
    # 試験実施
    models_resp = _client.get("/v1/models")

    # 結果検証 (観点1)
    assert models_resp.status_code == 200
    model_ids = {m["id"] for m in models_resp.json()["data"]}
    assert "agent_assistant_v1" in model_ids

    # 観点2: 非ストリーミング
    # 試験実施
    resp = _client.post(
        "/v1/chat/completions",
        json={
            "model": "agent_assistant_v1",
            "messages": [{"role": "user", "content": "こんにちは"}],
        },
    )

    # 結果検証 (観点2)
    assert resp.status_code == 200
    body = resp.json()
    assert body["object"] == "chat.completion"
    assert body["model"] == "agent_assistant_v1"
    assert body["choices"][0]["message"]["role"] == "assistant"
    assert body["choices"][0]["message"]["content"]

    # 観点3: ストリーミング
    # 試験実施
    stream_resp = _client.post(
        "/v1/chat/completions",
        json={
            "model": "agent_assistant_v1",
            "messages": [{"role": "user", "content": "こんにちは"}],
            "stream": True,
        },
    )

    # 結果検証 (観点3)
    assert stream_resp.status_code == 200
    assert "text/event-stream" in stream_resp.headers["content-type"]
    data_lines = [line for line in stream_resp.text.splitlines() if line.startswith("data: ")]
    assert len(data_lines) >= 1
    assert any(line == "data: [DONE]" for line in data_lines)
