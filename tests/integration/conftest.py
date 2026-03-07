import os
import pytest
from sqlalchemy import create_engine
from langchain_postgres import PGEngine


@pytest.fixture(scope="session")
def pg_engine():
    """langchain_postgres PGEngine (セッション全体で共有)。"""
    conn_str = os.environ.get("ENV_PG_CONNECTION_STRING")
    if not conn_str:
        pytest.fail("ENV_PG_CONNECTION_STRING が未設定のため失敗")
    return PGEngine.from_connection_string(conn_str)


@pytest.fixture(scope="session")
def sa_engine():
    """SQLAlchemy Engine (セッション全体で共有)。"""
    conn_str = os.environ.get("ENV_PG_CONNECTION_STRING")
    if not conn_str:
        pytest.fail("ENV_PG_CONNECTION_STRING が未設定のため失敗")
    engine = create_engine(conn_str)
    yield engine
    engine.dispose()
