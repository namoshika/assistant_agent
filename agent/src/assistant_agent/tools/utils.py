import urllib.error
import urllib.request
from datetime import datetime
from typing import Annotated
from zoneinfo import ZoneInfo

from langchain.tools import tool
from langchain_core.prompts import PromptTemplate
from pydantic import Field

_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"
)

_WEB_FETCH_TMPL = """\
## HTTP Response (url: {{ url }}, status: {{ status }})
{{ body }}
"""


# --------------------------------
# Tool: get_weather
# --------------------------------
@tool
def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"


# --------------------------------
# Tool: get_datetime_now
# --------------------------------
@tool
def get_datetime_now() -> str:
    """現在時刻を JST の ISO8601 文字列で取得する."""
    return datetime.now(ZoneInfo("Asia/Tokyo")).isoformat()


# --------------------------------
# Tool: web_fetch
# --------------------------------
@tool
def web_fetch(url: Annotated[str, Field(description="取得対象の URL")]) -> str:
    """指定 URL に HTTP GET を行い、レスポンス本文をそのまま返す.

    Tavily 系ツールによる抽出・要約を経由しないため、RSS フィード等
    構造化されたコンテンツをそのままの形式で取得したい場合に使う。
    """
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            charset = resp.headers.get_content_charset(failobj="utf-8")
            status = resp.status
            body = resp.read().decode(charset, errors="replace")
    except urllib.error.HTTPError as ex:
        charset = ex.headers.get_content_charset(failobj="utf-8")
        status = ex.code
        body = ex.read().decode(charset, errors="replace") or str(ex)
    except urllib.error.URLError as ex:
        status, body = "error", str(ex)

    return PromptTemplate.from_template(_WEB_FETCH_TMPL, template_format="jinja2").format(
        url=url, status=status, body=body
    )
