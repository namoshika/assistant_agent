import os
import dotenv
import mlflow
import mlflow.models
from typing import Annotated
from langchain.agents import create_agent
from langchain_core.documents import Document
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from langchain_postgres import PGEngine
from langchain.tools import tool
from sqlalchemy import create_engine

from utils.mlflow import LangGraphWrapper
from utils.chunker.markdown import MarkdownHeaderChunker
from utils.chunker.text import TextChunker
from utils.documentstore.markdown import MarkdownDocumentStore
from utils.documentstore.obsidian import ObsidianDocumentStore
from utils.retriever import MonthlyNewsChunkStore, ObsidianChunkStore

AGENT_NAME = "agent"

dotenv.load_dotenv()
CONNECTION_STRING = os.getenv("ENV_PG_CONNECTION_STRING")
engine = PGEngine.from_connection_string(url=CONNECTION_STRING, pool_size=5)
doc_chunk = MonthlyNewsChunkStore(engine)
doc_store = MarkdownDocumentStore("monthly_news", doc_chunk, doc_chunk, MarkdownHeaderChunker())

sa_engine = create_engine(CONNECTION_STRING)
obsidian_chunk = ObsidianChunkStore(engine)
obsidian_store = ObsidianDocumentStore("obsidian_vault", obsidian_chunk, sa_engine, TextChunker())


@tool
def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"


@tool(response_format="content_and_artifact")
def search_knowledge(search_query: Annotated[str, "検索期間 yyyy/MM"]) -> list[Document]:
    """Search monthly news for a given query."""
    doc_store.connect()
    tmpl = (
        " title: {doc_name}  \n"
        + "===  \n"
        + "{doc_content}  \n\n"
    )

    results = doc_store.search_documents(search_query, top_k=5)
    contents = [
        tmpl.format(
            doc_name=item.metadata["source"],
            doc_content=item.page_content,
        )
        for item in results
    ]
    contents = "".join(contents)
    return contents, results


@tool(response_format="content_and_artifact")
def obsidian_vault_search(search_query: Annotated[str, "検索クエリ"]) -> list[Document]:
    """Obsidian vault を検索し、マッチしたノートの原文を返す。"""
    obsidian_store.connect()
    tmpl = (
        " title: {doc_name}  \n"
        + "===  \n"
        + "{doc_content}  \n\n"
    )
    results = obsidian_store.search_documents(search_query, top_k=5)
    contents = [
        tmpl.format(
            doc_name=item.metadata.get("path", ""),
            doc_content=item.page_content,
        )
        for item in results
    ]
    return "".join(contents), results


@tool(response_format="content_and_artifact")
def obsidian_vault_get(ids: Annotated[list[str], "取得するノート ID のリスト"]) -> list[Document]:
    """ファイル名または相対パスで指定した Obsidian ノートの原文を返す。"""
    obsidian_store.connect()
    tmpl = (
        " title: {doc_name}  \n"
        + "===  \n"
        + "{doc_content}  \n\n"
    )
    results = obsidian_store.get_documents(ids)
    contents = [
        tmpl.format(
            doc_name=item.metadata.get("path", ""),
            doc_content=item.page_content,
        )
        for item in results
    ]
    return "".join(contents), results

def init_agent():
    GEMINI_MODEL_ID = os.environ.get("GEMINI_MODEL_ID", "gemini-3-flash-preview")
    GEMINI_API_KEY = os.environ.get("ENV_GEMINI_API_KEY")
    llm = ChatGoogleGenerativeAI(model=GEMINI_MODEL_ID, api_key=GEMINI_API_KEY)

    agent = create_agent(
        model=llm,
        tools=[get_weather, search_knowledge, obsidian_vault_search, obsidian_vault_get],
        system_prompt="You are a helpful assistant",
    )
    agent_wrapped = LangGraphWrapper(agent)
    return agent_wrapped

mlflow.models.set_model(agent_wrapped)
