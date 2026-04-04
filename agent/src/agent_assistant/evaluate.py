import mlflow
import mlflow.genai
from mlflow.genai.scorers import Correctness, Guidelines  # noqa: F401
from mlflow.pyfunc.model import ChatAgent, ResponsesAgent
from mlflow.types.agent import ChatAgentResponse
from mlflow.types.responses import ResponsesAgentResponse
from mlflow.types.responses_helpers import Message, OutputItem


def eval_responses(model: ResponsesAgent, eval_dataset: list):
    """評価用データセットを用いて ResponsesAgent の評価を行う. 実装内容は要件に応じて要書き換え."""

    # 評価基準を定義
    @mlflow.genai.scorer(description="出力を評価 (観点: 回答が10文字以上であること)")
    def custom_check(outputs: str) -> bool:
        return len(outputs) > 10

    scorers = [
        # 出力が期待値と一致していること
        # Correctness(model="bedrock:/global.anthropic.claude-haiku-4-5-20251001-v1:0"),
        Correctness(model="bedrock:/google.gemma-3-4b-it"),
        # # 出力が日本語であること
        Guidelines(
            model="bedrock:/google.gemma-3-4b-it",
            name="is_japanese",
            guidelines="The answer must be in Japanese",
        ),
        # 任意の評価基準を満たす事
        custom_check,
    ]

    # モデルを評価
    @mlflow.trace
    def predict_fn(messages: list[Message | OutputItem]):
        res = model.predict({"input": messages})  # pyright: ignore[reportArgumentType]

        # mlflow.pyfunc.log_model() すると ResponsesAgent も PythonModel になる
        # ResponsesAgent と PythonModel では predict 時の戻り値が異なるため、辞書に揃える
        if isinstance(res, ResponsesAgentResponse):
            res = res.model_dump()

        res = "".join([item["text"] for item in res["output"][-1]["content"]])
        return res

    res = mlflow.genai.evaluate(eval_dataset, scorers, predict_fn)
    return res


def eval_responses_cc(model: ChatAgent, eval_dataset: list):
    """評価用データセットを用いて ChatAgent の評価を行う. 実装内容は要件に応じて要書き換え."""

    # 評価基準を定義
    @mlflow.genai.scorer(description="出力を評価 (観点: 回答が10文字以上であること)")
    def custom_check(outputs: str) -> bool:
        return len(outputs) > 10

    scorers = [
        # Correctness(model="bedrock:/global.anthropic.claude-haiku-4-5-20251001-v1:0"),
        Correctness(model="bedrock:/google.gemma-3-4b-it"),
        # 出力が日本語であること
        Guidelines(
            model="bedrock:/google.gemma-3-4b-it",
            name="is_japanese",
            guidelines="The answer must be in Japanese",
        ),
        # 任意の評価基準を満たす事
        custom_check,
    ]

    # モデルを評価
    @mlflow.trace
    def predict_fn(messages: list):
        res = model.predict({"messages": messages})  # pyright: ignore[reportArgumentType]

        # mlflow.pyfunc.log_model() すると ChatAgent も PythonModel になる
        # ChatAgent と PythonModel では predict 時の戻り値が異なるため、辞書に揃える
        if isinstance(res, ChatAgentResponse):
            res = res.model_dump(exclude_none=True)

        return res["messages"][-1].get("content", "")

    res = mlflow.genai.evaluate(eval_dataset, scorers, predict_fn)
    return res
