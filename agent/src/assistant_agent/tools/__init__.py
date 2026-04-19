from . import obsidian, sample


def get_tools():
    """エージェントに使用させるツールを返す."""
    return [
        sample.get_weather,
        sample.sample_search,
        obsidian.obsidian_vault_search,
        obsidian.obsidian_vault_get,
    ]
