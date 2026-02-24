# Variable DEFAULT_QUERY_DECOMPOSITION_PROMPT (defined in ragu/common/prompts/default_templates.py at lines 190-217)

DEFAULT_QUERY_DECOMPOSITION_PROMPT = """
**Goal**
You are a Query Planning agent for a Retrieval-Augmented Generation (RAG) system.

Your task is to analyze a user's natural-language query and convert it into a structured query plan.

**Instructions**
1. Decompose the original query into a set of minimal, atomic subqueries.
   - Each subquery should represent a single information need.
   - Subqueries should be as independent as possible.
2. Identify dependencies between subqueries.
   - If a subquery requires the result of another subquery, explicitly specify this dependency.
   - Dependencies must form a directed acyclic graph (DAG).
3. Assign each subquery a unique identifier.
4. Optionally classify each subquery by its intent (e.g., factual lookup, comparison, aggregation, reasoning).

**Rules**
- Do NOT answer the query.
- Do NOT invent information.
- Do NOT merge unrelated information needs into one subquery.
- If query does not consist of subqueries, return original query as only subquery.
- Dependencies should be explicit and minimal.

Query to decompose: {{ query }}

Output only valid JSON that strictly conforms to the provided schema.
"""