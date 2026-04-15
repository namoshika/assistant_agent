from . import databricks, obsidian, sample

__all__ = [
    "databricks",
    "obsidian",
    "sample",
]


def get_tools():
    """エージェントに使用させるツールを返す."""
    return [
        sample.get_weather,
        databricks.databricks_search,
        # sample.sample_search,
        # obsidian.obsidian_vault_search,
        # obsidian.obsidian_vault_get,
    ]
