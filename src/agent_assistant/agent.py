import os
import dotenv
import mlflow.models
from langchain.agents import create_agent
from langchain_aws import ChatBedrock
from pydantic import SecretStr

from agent_assistant.utils.mlflow import LangGraphWrapper
from agent_assistant.context import ContextSchema, build_session
from agent_assistant import tool

AGENT_NAME = "agent"

dotenv.load_dotenv(override=True)
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_DEFAULT_REGION = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
assert AWS_ACCESS_KEY_ID is not None
assert AWS_SECRET_ACCESS_KEY is not None
AWS_ACCESS_KEY_ID = SecretStr(AWS_ACCESS_KEY_ID)
AWS_SECRET_ACCESS_KEY = SecretStr(AWS_SECRET_ACCESS_KEY)

# from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
# llm = ChatGoogleGenerativeAI(
#     model=os.environ.get("ENV_GEMINI_MODEL_ID", "gemini-3.1-pro-preview"),
#     api_key=os.environ.get("ENV_GEMINI_API_KEY"),
# )

llm = ChatBedrock(
    model="global.anthropic.claude-haiku-4-5-20251001-v1:0",
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    region=AWS_DEFAULT_REGION,
)

SYSTEM_PROMPT = """
You are a helpful assistant.
Respond to the user in Japanese.
"""
agent = create_agent(
    model=llm,
    tools=[
        tool.get_weather,
        tool.obsidian_vault_search,
        tool.obsidian_vault_get,
    ],
    system_prompt=SYSTEM_PROMPT,
    context_schema=ContextSchema,
    name=AGENT_NAME,
)
context = build_session()

agent_wrapped = LangGraphWrapper(agent, context)
mlflow.models.set_model(agent_wrapped)
