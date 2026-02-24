# Variable DEFAULT_GLOBAL_SEARCH_PROMPT (defined in ragu/common/prompts/default_templates.py at lines 121-135)

DEFAULT_GLOBAL_SEARCH_PROMPT = """
**Goal**
Answer the query by summarizing the provided ranked context.

**Instructions**
1. Consider the relevance ranking (lower rank = less relevant).
2. Briefly reason about context relevance before giving the final answer.

Query: {{ query }}
Context: {{ context }}

Provide the answer in the following language: {{ language }}
Return the result as valid JSON matching the provided schema.
"""