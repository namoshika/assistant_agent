from collections.abc import Callable
from typing import Any, ClassVar, TypedDict, cast


class CommonContext(TypedDict, total=False):
    """エージェント共通コンテキスト。将来の共通フィールド追加の置き場."""


class ContextRegistry:
    _factories: ClassVar[dict[str, dict[str, Callable[..., Any]]]] = {}

    @classmethod
    def register(cls, name: str, variant: str = "default") -> Callable:
        """Register factory 関数を name / variant で登録する."""

        def decorator(fn: Callable) -> Callable:
            if variant in cls._factories.get(name, {}):
                raise ValueError(f"variant '{variant}' is already registered for '{name}'.")
            cls._factories.setdefault(name, {})[variant] = fn
            return fn

        return decorator

    @classmethod
    def build(cls, variants: dict[str, str] | None = None, **kwargs: Any) -> CommonContext:
        """登録済み factory を呼び出して context dict を返す."""
        resolved = variants or {}
        result = {}
        for name, impls in cls._factories.items():
            variant = resolved.get(name, "default")
            if variant not in impls:
                available = list(impls.keys())
                raise ValueError(
                    f"variant '{variant}' is not registered for '{name}'. "
                    f"Available variants: {available}"
                )
            result[name] = impls[variant](**kwargs)
        return cast(CommonContext, result)
