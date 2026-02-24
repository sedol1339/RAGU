# Variable DEFAULT_GLOBAL_SEARCH_CONTEXT_PROMPT (defined in ragu/common/prompts/default_templates.py at lines 106-120)

DEFAULT_GLOBAL_SEARCH_CONTEXT_PROMPT = """
**Goal**
Answer the query by summarizing relevant information from the context and, if needed, well-known facts.

**Instructions**
1. Reason about context relevance.
2. Provide a usefulness rating from 0 to 10 (0 = useless, 10 = direct answer).

Query: {{ query }}
Context: {{ context }}

Provide the answer in the following language: {{ language }}
Return the result as valid JSON matching the provided schema.
"""