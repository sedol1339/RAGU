# Variable DEFAULT_ENTITY_SUMMARIZER_PROMPT (defined in ragu/common/prompts/default_templates.py at lines 95-105)

DEFAULT_ENTITY_SUMMARIZER_PROMPT = """
**Goal**
From the given entity and multiple phrases, produce one concise, consistent entity description.

Data:
Entity: {{ entity.entity_name }}, Description: {{ entity.description }}

Provide the answer in the following language: {{ language }}
Return the result as valid JSON matching the provided schema.
"""