# Variable DEFAULT_RAGU_LM_ENTITY_NORMALIZATION_PROMPT (defined in ragu/common/prompts/default_templates.py at lines 168-174)

DEFAULT_RAGU_LM_ENTITY_NORMALIZATION_PROMPT = """
Выполните нормализацию именованной сущности, встретившейся в тексте.
Исходная (ненормализованная) именованная сущность: {{ source_entity }}
Текст: {{ source_text }}
Нормализованная именованная сущность:
"""