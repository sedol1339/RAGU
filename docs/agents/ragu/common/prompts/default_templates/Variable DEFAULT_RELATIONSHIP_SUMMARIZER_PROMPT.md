# Variable DEFAULT_RELATIONSHIP_SUMMARIZER_PROMPT (defined in ragu/common/prompts/default_templates.py at lines 84-94)

DEFAULT_RELATIONSHIP_SUMMARIZER_PROMPT = """
**Goal**
From the given entity pair and multiple phrases, produce one concise, consistent relationship description.

Data:
Subject: {{ relation.subject_name }}, Object: {{ relation.object_name }}, Relationship description: {{ relation.description }}

Provide the answer in the following language: {{ language }}
Return the result as valid JSON matching the provided schema.
"""