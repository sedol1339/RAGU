# Variable DEFAULT_CLUSTER_SUMMARIZER_PROMPT (defined in ragu/common/prompts/default_templates.py at lines 151-159)

DEFAULT_CLUSTER_SUMMARIZER_PROMPT = """
**Goal**
You are given a list of descriptions.  
Summarize them into a single concise description.  

Texts to summarize:  
{{ content }}
"""