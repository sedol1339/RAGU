# Variable DEFAULT_ARTIFACTS_VALIDATOR_PROMPT (defined in ragu/common/prompts/default_templates.py at lines 38-60)

DEFAULT_ARTIFACTS_VALIDATOR_PROMPT = """
**Goal**
Validate correctness and completeness of entities and relationships against the given text.

**Instructions**
1. Add missing entities with correct types and descriptions.
2. Add missing relationships with descriptions and strength.
3. Return full updated lists.

{% if entity_types -%}
The entity type must be one of the following: {{ entity_types }}
{% endif %} 

Triplets for validation:
{{ artifacts }}

Text for validation:
{{ context }}

Provide the answer in the following language: {{ language }}
Return the result as valid JSON matching the provided schema.
"""