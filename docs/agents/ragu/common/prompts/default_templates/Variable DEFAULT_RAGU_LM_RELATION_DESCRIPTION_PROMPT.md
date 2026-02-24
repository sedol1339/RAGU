# Variable DEFAULT_RAGU_LM_RELATION_DESCRIPTION_PROMPT (defined in ragu/common/prompts/default_templates.py at lines 182-189)

DEFAULT_RAGU_LM_RELATION_DESCRIPTION_PROMPT = """
Напишите, что означает отношение между двумя именованными сущностями в тексте, то есть раскройте смысл этого отношения относительно текста (либо напишите прочерк, если между двумя именованными сущностями отсутствует отношение).
Первая именованная сущность: {{ first_normalized_entity }}
Вторая именованная сущность: {{ second_normalized_entity }}
Текст: {{ source_text }}
Смысл отношения между двумя именованными сущностями:
"""