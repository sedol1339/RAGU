# Variable DEFAULT_COMMUNITY_REPORT_PROMPT (defined in ragu/common/prompts/default_templates.py at lines 61-83)

DEFAULT_COMMUNITY_REPORT_PROMPT = """
**Goal**
Generate a detailed community report using entities, their relationships, and any additional statements.

**Instructions**
1. Create a clear title and summary.
2. Provide an impact rating with justification.
3. Produce 5–10 key findings with short summaries and detailed explanations.

Input text:
{% for entity in community.entities -%}
Entity: {{ entity.entity_name }}, description: {{ entity.description }}{% if not loop.last %}, {% endif %}
{% endfor %}

Relations
{% for relation in community.relations -%}
{{ relation.subject_name }} -> {{ relation.object_name }}, relations description: {{ relation.description }}
{% endfor %}

Provide the answer in the following language: {{ language }}
Return the result as valid JSON matching the provided schema.
"""