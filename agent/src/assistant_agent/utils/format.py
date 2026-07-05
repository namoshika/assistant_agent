from dataclasses import dataclass
from typing import Any, Sequence, TypeAlias

import yaml
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate

# --------------------------------
# Utils: format_doc_ids
# --------------------------------
_DOCUMENT_IDS_TMPL_STR = """\
Search results ({{ documents | length }} documents found):
{% for doc in documents -%}
- { document_id: "{{ doc.id }}", file_path: "{{ doc.metadata['file_path'] }}" }
{% endfor %}
Use obsidian_vault_get with document_ids to retrieve full content.
"""


def format_doc_ids(documents: Sequence[Document]) -> str:
    """Document リストから document_id と file_path の一覧文字列を返す."""
    return (
        PromptTemplate.from_template(_DOCUMENT_IDS_TMPL_STR, template_format="jinja2")
        .invoke({"documents": documents})
        .to_string()
    )


# --------------------------------
# Utils: format_doc_list
# --------------------------------
_DOCS_OBS_TMPL_STR = """\
{% for doc in docs -%}
title: {{ doc.title }}
===

{% if doc.frontmatter -%}
```yaml
{{ to_yaml(doc.frontmatter) -}}
```
{%- endif %}

{{ doc.contents }}
{%- if not loop.last %}

---

{% endif %}
{%- endfor -%}
"""


TPayload: TypeAlias = str | int | float


@dataclass
class ContentsWithFrontmatter:
    title: str
    contents: str
    frontmatter: dict[str, Any] | None


def format_doc_list(documents: Sequence[ContentsWithFrontmatter]) -> str:
    """Document オブジェクトのリストを、エージェントが読みやすいテキスト形式に整形する."""
    return (
        PromptTemplate.from_template(_DOCS_OBS_TMPL_STR, template_format="jinja2")
        .invoke(
            {
                "docs": documents,
                "to_yaml": lambda text: yaml.safe_dump(text, allow_unicode=True),
            }
        )
        .to_string()
    )
