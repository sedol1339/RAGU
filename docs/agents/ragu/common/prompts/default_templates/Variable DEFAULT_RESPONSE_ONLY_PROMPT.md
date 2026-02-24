# Variable DEFAULT_RESPONSE_ONLY_PROMPT (defined in ragu/common/prompts/default_templates.py at lines 136-150)

DEFAULT_RESPONSE_ONLY_PROMPT = """
**Goal**
Answer the query by summarizing relevant information from the context and, if necessary, well-known facts.

**Instructions**
1. If you do not know the correct answer, explicitly state that.
2. Do not include unsupported information.

Query: {{ query }}
Context: {{ context }}

Provide the answer in the following language: {{ language }}
Return the result as valid JSON matching the provided schema.
"""