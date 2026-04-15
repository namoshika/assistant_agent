"""
title: Databricks App ChatCompletion Pipe.
author: namoshika
version: 1.0.0
"""  # noqa: D205, D212, D400, D415

import logging
from datetime import datetime, timedelta, timezone
from typing import Generator, Optional, Union

import requests
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class Pipe:
    """Databricks Apps ChatCompletion 連携 Pipe."""

    class Valves(BaseModel):
        """Open WebUI 管理画面から設定する接続パラメータ."""

        databricks_host: str = Field(
            default="your-workspace.azuredatabricks.net",
            description="Databricks ワークスペースのホスト名 (https:// なし)",
        )
        databricks_app_url: str = Field(
            default="https://agent-assistant-xxxx.databricksapps.com",
            description="Databricks Apps のエンドポイント URL",
        )
        oauth_client_id: str = Field(
            default="",
            description="サービスプリンシパルのクライアント ID",
        )
        oauth_client_secret: str = Field(
            default="",
            description="サービスプリンシパルの OAuth シークレット",
        )

    def __init__(self):
        """Construct Pipe."""
        self.name = "DbxApp"
        self.valves = self.Valves()
        self._access_token: Optional[str] = None
        self._token_expires_at: Optional[datetime] = None  # timezone-aware (UTC)

    def _is_token_expired(self) -> bool:
        if not self._token_expires_at:
            return True
        return datetime.now(timezone.utc) + timedelta(minutes=5) >= self._token_expires_at

    def _get_oauth_token(self) -> str:
        if self._access_token and not self._is_token_expired():
            return self._access_token

        url = f"https://{self.valves.databricks_host.strip()}/oidc/v1/token"
        data = {
            "grant_type": "client_credentials",
            "scope": "all-apis",
            "client_id": self.valves.oauth_client_id.strip(),
            "client_secret": self.valves.oauth_client_secret.strip(),
        }
        response = requests.post(
            url,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            data=data,
        )
        response.raise_for_status()
        token_data = response.json()
        self._access_token = token_data["access_token"]
        expires_in = token_data.get("expires_in", 3600)
        self._token_expires_at = datetime.now(timezone.utc) + timedelta(seconds=expires_in)
        logger.info("[%s] OAuth token refreshed (expires_in=%ds)", self.name, expires_in)
        assert self._access_token is not None
        return self._access_token

    def _get_headers(self) -> dict:
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._get_oauth_token()}",
        }

    def pipes(self) -> list:
        """Return available models from the agent server."""
        url = f"{self.valves.databricks_app_url.rstrip('/')}/api/models"
        response = requests.get(url, headers=self._get_headers())
        response.raise_for_status()
        # Open WebUI は各エントリに "name" キーを要求するが、エージェントサーバーの
        # ModelInfo には "name" がないため補完する。
        # Open WebUI はモデル名を "{self.name}{name}" として表示するため、
        # セパレータ " " を挟んで "{self.name} {id}" 形式にする。
        return [{"name": f" {m['id']}", **m} for m in response.json()["data"]]

    def pipe(self, body: dict) -> Union[dict, Generator]:
        """Forward chat completion request to the Databricks Apps agent."""
        # body["model"] は "pipe名.model_id" 形式で渡る。"." 以降を model_id として抽出する。
        # "." が含まれない場合、find() は -1 を返し元の文字列がそのまま使われる。
        raw_model = body.get("model", "")
        model_id = raw_model[raw_model.find(".") + 1 :]
        payload = {**body, "model": model_id}
        headers = self._get_headers()
        url = f"{self.valves.databricks_app_url.rstrip('/')}/api/chat/completions"

        if body.get("stream", False):
            return self._stream_response(url, headers, payload)
        else:
            return self._non_stream_response(url, headers, payload)

    def _stream_response(self, url: str, headers: dict, payload: dict) -> Generator:
        payload["stream"] = True
        response = requests.post(url, headers=headers, json=payload, stream=True)
        response.raise_for_status()
        for line in response.iter_lines():
            if line:
                yield line.decode("utf-8") + "\n\n"

    def _non_stream_response(self, url: str, headers: dict, payload: dict) -> dict:
        payload["stream"] = False
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()
