# Variable DEFAULT_RAGU_LM_ENTITY_EXTRACTION_PROMPT (defined in ragu/common/prompts/default_templates.py at lines 162-167)

DEFAULT_RAGU_LM_ENTITY_EXTRACTION_PROMPT = """
Распознайте все именованные сущности в тексте и выпишите их список с новой строки.
Текст: {{ text }}
Именованные сущности:
"""