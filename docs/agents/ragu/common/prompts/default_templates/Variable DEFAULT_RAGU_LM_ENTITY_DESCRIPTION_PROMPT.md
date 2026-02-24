# Variable DEFAULT_RAGU_LM_ENTITY_DESCRIPTION_PROMPT (defined in ragu/common/prompts/default_templates.py at lines 175-181)

DEFAULT_RAGU_LM_ENTITY_DESCRIPTION_PROMPT = """
Напишите, что означает именованная сущность в тексте, то есть раскройте её смысл относительно текста.
Именованная сущность: {{ normalized_entity }}
Текст: {{ source_text }}
Смысл именованной сущности:
"""